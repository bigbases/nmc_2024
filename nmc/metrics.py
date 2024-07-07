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
<<<<<<< HEAD
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
=======
        pred = pred.argmax(dim=1)
        for t, p in zip(target.view(-1), pred.view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1

    def compute_accuracy(self) -> float:
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return (correct / total * 100).item()

    def compute_precision_recall_f1(self) -> Dict[str, Dict[int, float]]:
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        results = {
            'precision': {},
            'recall': {},
            'f1': {}
        }

        for i in range(self.num_classes):
            results['precision'][i] = precision[i].item() * 100
            results['recall'][i] = recall[i].item() * 100
            results['f1'][i] = f1[i].item() * 100

        return results

    def compute_metrics(self) -> Dict[str, float]:
        accuracy = self.compute_accuracy()
        class_metrics = self.compute_precision_recall_f1()

        avg_precision = sum(class_metrics['precision'].values()) / self.num_classes
        avg_recall = sum(class_metrics['recall'].values()) / self.num_classes
        avg_f1 = sum(class_metrics['f1'].values()) / self.num_classes

        return {
            'accuracy': round(accuracy, 2),
            'avg_precision': round(avg_precision, 2),
            'avg_recall': round(avg_recall, 2),
            'avg_f1': round(avg_f1, 2),
            'class_metrics': class_metrics
        }
>>>>>>> 63a9dcf9ea8282e31f104a66d5facd6eaebb8020
