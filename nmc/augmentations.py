import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for transform in self.transforms:
            img = transform(img)
        return img

class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Tensor) -> Tensor:
        if self.brightness > 0:
            img = TF.adjust_brightness(img, self.brightness)
        if self.contrast > 0:
            img = TF.adjust_contrast(img, self.contrast)
        if self.saturation > 0:
            img = TF.adjust_saturation(img, self.saturation)
        if self.hue > 0:
            img = TF.adjust_hue(img, self.hue)
        return img

class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor) -> Tensor:
        return TF.adjust_gamma(img, self.gamma, self.gain)

class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, self.sharpness)
        return img

class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            img = TF.autocontrast(img)
        return img

class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            img = TF.gaussian_blur(img, self.kernel_size)
        return img

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            return TF.hflip(img)
        return img

class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            return TF.vflip(img)
        return img

class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img

class Equalize:
    def __call__(self, image):
        return TF.equalize(image)

class Posterize:
    def __init__(self, bits=2):
        self.bits = bits
        
    def __call__(self, image):
        return TF.posterize(image, self.bits)

class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0]):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        
    def __call__(self, img):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.BILINEAR, 0)

class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, expand: bool = False) -> None:
        self.p = p
        self.angle = degrees
        self.expand = expand

    def __call__(self, img: Tensor) -> Tensor:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            img = TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
        return img

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor) -> Tensor:
        return TF.center_crop(img, self.size)

class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
        return img

class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int]) -> None:
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return TF.pad(img, padding)

class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        H, W = img.shape[1:]
        tH, tW = self.size

        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)

        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        return img 

class Resize:
    def __init__(self, size: int = 1024) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape

        # 긴 쪽을 기준으로 스케일 팩터 계산
        scale_factor = self.size / max(H, W)
        
        # 새로운 높이와 너비 계산
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        
        # 이미지 리사이즈
        img = TF.resize(img, (new_H, new_W), interpolation=TF.InterpolationMode.BILINEAR)

        # 패딩 계산
        pad_H = max(0, self.size - new_H)
        pad_W = max(0, self.size - new_W)
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        # 패딩 적용 (검정색으로)
        img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return img

class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0)) -> None:
        self.size = size
        self.scale = scale

    def __call__(self, img: Tensor) -> Tensor:
        H, W = img.shape[1:]
        tH, tW = self.size

        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        scale = int(tH*ratio), int(tW*4*ratio)

        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)

        margin_h = max(img.shape[1] - tH, 0)
        margin_w = max(img.shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        img = img[:, y1:y2, x1:x2]

        if img.shape[1:] != self.size:
            padding = [0, 0, tW - img.shape[2], tH - img.shape[1]]
            img = TF.pad(img, padding, fill=0)
        return img 

def get_train_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        Resize(size[0]),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        Resize(size[0]),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

if __name__ == '__main__':
    h = 230
    w = 420
    img = torch.randn(3, h, w)
    aug = Compose([
        RandomResizedCrop((512, 512)),
    ])
    img = aug(img)
    print(img.shape)