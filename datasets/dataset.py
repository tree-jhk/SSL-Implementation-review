from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb # 데이터셋이 labeled인지 unlabeled인지
        self.onehot = onehot

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
# unlabled data에 대해서는 k번의 random한 augmentation이 들어가는 코드
                self.strong_transform.transforms.insert(0, RandAugment(3, 5)) # (3, 5)에서 3만 실제로 쓰이고, 3이 곧 k번의 augmentation을 의미함.
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_) # target을 onehot으로 만들지

        # set augmented images

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else: # mixmatch, fixmatch, flextmatch 모두 transform을 필요로 함.
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img) # numpy 배열로 되어있는 이미지 배열을 PIL 이미지로 변환
            img_w = self.transform(img)
            if not self.is_ulb: # 데이터셋이 labeled data이면
                return idx, img_w, target # idx, img_w, target == 이미지의 index, augmented 이미지, 라벨
            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'flexmatch':
                    return idx, img_w, self.strong_transform(img) # idx, img_w, self.strong_transform(img) == 이미지의 index, augmented 이미지, strongly augmented 이미지,
                elif self.alg == 'softmatch' or self.alg == 'freematch' or self.alg == 'freematch_entropy':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img) # idx, img_w, self.transform(img) == 이미지의 index, augmented 이미지, augmented 이미지 -> img_x와 self.transform(img) 다를 수 있음 왜냐면 aumgentation이 random하게 적용되기 때문에.
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)
