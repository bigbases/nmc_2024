import torch
from torch import Tensor
from typing import Tuple, Dict

class MultiLabelMetrics:
    def __init__(self, num_classes: int, device) -> None:
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrices = [torch.zeros((2, 2), dtype=torch.int).to(self.device) for _ in range(num_classes)]

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.sigmoid(pred) > 0.5
        for i in range(target.size(0)):
            for j in range(self.num_classes):
                actual = target[i, j].item()
                predicted = pred[i, j].item()
                self.confusion_matrices[j][int(actual), int(predicted)] += 1

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

    def compute_accuracy(self) -> float:
        total_correct = sum([cm[1, 1] for cm in self.confusion_matrices])
        total_correct += sum([cm[0, 0] for cm in self.confusion_matrices])

        total = sum([cm.sum() for cm in self.confusion_matrices])
        print(f"total: {total}")
        return (total_correct / total * 100).item() if total != 0 else 0

    def compute_precision_recall_f1(self) -> Dict[str, Dict[int, float]]:
        results = {'precision': {}, 'recall': {}, 'f1': {}}

        for i, cm in enumerate(self.confusion_matrices):
            tp = cm[1, 1].float()
            fp = cm[0, 1].float()
            fn = cm[1, 0].float()

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

            results['precision'][i] = precision.item() * 100 if torch.is_tensor(precision) else precision * 100
            results['recall'][i] = recall.item() * 100 if torch.is_tensor(recall) else recall * 100
            results['f1'][i] = f1.item() * 100 if torch.is_tensor(f1) else f1 * 100

        return results

    
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
            
    def update_epi(self, pred: torch.Tensor, target: torch.Tensor) -> None:
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
