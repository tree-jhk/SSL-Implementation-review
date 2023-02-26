import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import os
import contextlib
from train_utils import AverageMeter

from .mixmatch_utils import consistency_loss, Get_Scalar, one_hot, mixup_one_target
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy


class MixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, lambda_u, \
                 t_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Mixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(MixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function # T값 그대로 반환
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict): # set DataLoader on MixMatch
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register() # model의 파라미터들(model.named_parameters())을 dict로 저장
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        # Pytorch 에서 CUDA 호출이 비동기식이기 때문에 
        # 타이머를 시작 또는 중지 하기 전에 torch.cuda.synchronize() 를 통해 코드를 동기화 시켜주어야 한다. 
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record() # 시간 기록 시작
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)
        """
        mixmatch 기준으로,
            x_lb, y_lb: labeled image, target
            x_ulb_w1, x_ulb_w2: augmented unlabeled image1, augmented unlabeled image2
                augmented unlabeled image1와 augmented unlabeled image2는 
                모두 같은 image를 augement했지만, 적용된 augmentation이 random 기반이기 때문에 이미지 자체는 다르다.
                그러나 '라벨'은 같다!
        """
        for (_, x_lb, y_lb), (_, x_ulb_w1, x_ulb_w2) in zip(self.loader_dict['train_lb'],
                                                            self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w1.shape[0]
            assert num_ulb == x_ulb_w2.shape[0] # 같지 않으면 에러 발생

            x_lb, x_ulb_w1, x_ulb_w2 = x_lb.cuda(args.gpu), x_ulb_w1.cuda(args.gpu), x_ulb_w2.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                with torch.no_grad():
                    # 흠... bn freezing 왜하는거지?
                    self.bn_controller.freeze_bn(self.model)
                    logits_x_ulb_w1 = self.model(x_ulb_w1)
                    logits_x_ulb_w2 = self.model(x_ulb_w2)
                    self.bn_controller.unfreeze_bn(self.model)
                    # Temperature sharpening
                    T = self.t_fn(self.it)
                    # avg == 배치별 unlabeled data의 예측 확률들의 평균
                    # (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) == 같은 데이터에 각기 다른 augment 수행했는데 그것들을 다 더함. 평균을 내기 위해.
                    avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
                    avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
                    # sharpening == 배치별 unlabeled data의 예측 확률들의 평균을 shaprening해서 entropy minimize(더 확실한 확률로 만들기)
                    sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / T)
                    sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                    # Pseudo Label
                    # one_hot(y_lb, args.num_classes, args.gpu) == labeled data에 대해서는 라벨에 해당하는 부분만 1인 one-hot으로 표현해서 logit 표현
                    # input_labels == labeled data의 logit + unlabeled data의 logit + unlabeled data의 logit
                    input_labels = torch.cat(
                        [one_hot(y_lb, args.num_classes, args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)

                    # Mix up
                    # labeled data와 unlabeled data의 augmentation_1와 unlabeled data의 augmentation_2 concat
                    inputs = torch.cat([x_lb, x_ulb_w1, x_ulb_w2])
                    # len(mixed_x) == len(inputs)
                    # len(mixed_y) == len(input_labels)
                    # inputs: labeled와 unlabeled 둘을 concat만함. (num_lb를 기준으로 나뉘어짐)
                    # mixed_x[:num_lb]: labeled data들과 그 외의 데이터가 섞임(그 외의 데이터에는 labeled와 unlabeled가 포함됨)
                    # mixed_x[num_lb:]: unlabeled data들과 그 외의 데이터가 섞임(그 외의 데이터에는 labeled와 unlabeled가 포함됨)
                    mixed_x, mixed_y, _ = mixup_one_target(inputs, input_labels,
                                                        args.gpu,
                                                        args.alpha,
                                                        is_bias=True)

                    # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                    mixed_x = list(torch.split(mixed_x, num_lb)) # mixed_x가 num_lb 단위로 나뉘어짐.
                    mixed_x = self.interleave(mixed_x, num_lb) # rule-based로 labeled data와 unlabeled data 섞기
                    # interleave 부분 더 봐야할 듯
                    
                logits = [self.model(mixed_x[0])]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                # BN layer freeze 후 남은 배치들 학습
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt))

                # put interleaved samples back
                logits = self.interleave(logits, num_lb)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)
                self.bn_controller.unfreeze_bn(self.model)

                # 지도학습 loss == ce_loss
                sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
                sup_loss = sup_loss.mean()
                # 비지도학습 loss == mse_loss
                unsup_loss = consistency_loss(logits_u, mixed_y[num_lb:])

                # set ramp_up for lambda_u
                # rampup: self.it / (args.ramp_up * args.num_train_iter) 내의 element에 대해 0.0 미만인 값들을 min값으로 바꿔주고, 1.0 초과인 값들을 max값들로 바꿔주는 함수
                rampup = float(np.clip(self.it / (args.ramp_up * args.num_train_iter), 0.0, 1.0))
                lambda_u = self.lambda_u * rampup # iter를 반복할 때마다 lambda_u가 커짐(왜냐면 다른 값은 고정임.) 커지는 것이 아마도, iter를 반복할수록 unsupervised loss를 더 반영해도 되기 때문일듯

                total_loss = sup_loss + lambda_u * unsup_loss

            # parameter updates
            # amp는 빠른 연산을 도와주는 함수라고 대강 알고 있음
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0): # clipping을 한다면,
                    # clipping을 하는 이유:
                        # gradient exploding을 방지하여 학습의 안정화를 도모하기 위해 사용하는 방법
                        # gradient가 일정 threshold(args.clip)를 넘어가면 clipping을 해준다.
                        # clipping은 gradient의 L2norm(norm이지만 보통 L2 norm사용)으로 나눠주는 방식으로 수행
                        # threshold의 경우 gradient가 가질 수 있는 최대 L2norm을 뜻하고 이는 하이퍼파라미터로 사용자가 설정
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else: # args.amp는 아직 잘 모르니까, False로 두고 쓰자.
                total_loss.backward()
                if (args.clip > 0): # clipping을 한다면,
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record() # 1회 run 끝내기
            # synchronize는 GPU 연산이 끝날 때까지 기다리는 함수
            # GPU 연산의 결과가 필요한 시점 이전에 synchronize를 사용하지 않으면 딥러닝 결과의 올바름을 보장할 수 없음.
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            # total_loss = sup_loss + lambda_u * unsup_loss
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path):
        if self.it < 1000000:
            return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = deepcopy(self.model)
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model.state_dict()},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.print_fn('model loaded')

    # batch: num_lb, 
    # nu == len(xy) - 1 == mixed_x // num_lb - 1 == 길이가 num_lb인 토막들의 수 (마지막 토막은 num_lb보다 작을 수 있음)
    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1) # 마지막 토막이 num_lb보다 작을 수 있는데, 그거 다 채웠을 때의 크기를 담은 list
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets # [0, batch // (nu + 1), batch // (nu + 1) * 2, ..., batch // (nu + 1) * len(nu + 1)]

    # xy: mixed_x의 토막들, batch: num_lb
    # len(xy) == mixed_x // num_lb
    # len(xy) - 1 == 길이가 num_lb인 토막들의 수 (마지막 토막은 num_lb보다 작을 수 있음)
    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
