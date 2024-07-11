import torch
from torch import Tensor
from typing import Tuple, Dict


class Metrics:
    def __init__(self, num_classes: int, device) -> None:
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).to(self.device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
        
    def update_epi(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        keep = target != self.ignore_label

        target_flat = target[keep].flatten().to(torch.int64)
        pred_flat = pred[keep].flatten().to(torch.int64)

        bincount_input = target_flat * self.num_classes + pred_flat
        hist_update = torch.bincount(bincount_input, minlength=self.num_classes**2)
        self.hist += hist_update.view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
