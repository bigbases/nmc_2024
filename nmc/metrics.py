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