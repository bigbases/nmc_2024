import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from pathlib import Path
#from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from nmc.models import *
from nmc.models.heads import MultiHeadEmbedding
from nmc.datasets import * 
from nmc.augmentations import get_train_augmentation, get_val_augmentation
from nmc.losses import get_loss
from nmc.schedulers import get_scheduler
from nmc.optimizers import get_optimizer
from nmc.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate_epi
from nmc.utils.episodic_utils import * 
from torch import optim, nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 데이터셋에서 레이블 추출
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.from_numpy(self.labels)
        elif hasattr(dataset, 'targets'):
            self.labels = dataset.targets
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.from_numpy(self.labels)
        else:
            try:
                self.labels = [sample[1] for sample in dataset]
                if isinstance(self.labels[0], np.ndarray):
                    self.labels = torch.from_numpy(np.array(self.labels))
                else:
                    self.labels = torch.tensor(self.labels)
            except:
                raise ValueError("Cannot access labels from dataset")
        
        self.n_classes = self.labels.shape[1] if len(self.labels.shape) > 1 else len(torch.unique(self.labels))
        self.samples_per_class = batch_size // self.n_classes
        
        # 클래스별 인덱스 저장
        self.class_indices = []
        for i in range(self.n_classes):
            if len(self.labels.shape) > 1:
                idx = torch.where(self.labels[:, i] == 1)[0]
            else:
                idx = torch.where(self.labels == i)[0]
            self.class_indices.append(idx)
        
        self.n_batches = len(self.dataset) // batch_size
        if len(self.dataset) % batch_size != 0:
            self.n_batches += 1
    
    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            for class_idx in range(self.n_classes):
                class_samples = self.class_indices[class_idx]
                if len(class_samples) == 0:
                    continue
                
                # 랜덤 선택
                selected = class_samples[torch.randint(len(class_samples), 
                                                     (self.samples_per_class,))]
                batch_indices.extend(selected.tolist())
            
            # 배치 크기에 맞게 자르기
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            
            # 중요: 리스트로 yield
            yield batch_indices
    
    def __len__(self):
        return self.n_batches

def save_distance_analysis(distances_per_class, labels, thresholds, n_classes, output_dir='output'):
    """
    거리 분포 분석 및 시각화를 저장하는 함수
    
    Args:
        distances_per_class: 클래스별 거리 리스트
        labels: 실제 레이블
        thresholds: 클래스별 계산된 임계값 딕셔너리
        n_classes: 클래스 수
        output_dir: 출력 저장 디렉토리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 거리 분포 시각화
    plt.figure(figsize=(12, 6))
    for class_idx in range(n_classes):
        distances = distances_per_class[class_idx]
        plt.subplot(1, n_classes, class_idx + 1)
        
        # 정답 레이블별로 거리 분리
        positive_distances = [d for d, l in zip(distances, labels[:, class_idx]) if l == 1]
        negative_distances = [d for d, l in zip(distances, labels[:, class_idx]) if l == 0]
        
        # 히스토그램 그리기
        plt.hist(positive_distances, bins=50, alpha=0.5, label='Positive', color='blue')
        plt.hist(negative_distances, bins=50, alpha=0.5, label='Negative', color='red')
        
        # 임계값 표시
        threshold = thresholds[class_idx]
        plt.axvline(x=threshold, color='green', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})')
        
        plt.title(f'Class {class_idx} Distance Distribution')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 거리 통계 저장
    with open(os.path.join(output_dir, 'distance_stats.txt'), 'w') as f:
        for class_idx in range(n_classes):
            distances = distances_per_class[class_idx]
            positive_distances = [d for d, l in zip(distances, labels[:, class_idx]) if l == 1]
            negative_distances = [d for d, l in zip(distances, labels[:, class_idx]) if l == 0]
            threshold = thresholds[class_idx]
            
            f.write(f"\nClass {class_idx} Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Adaptive Threshold: {threshold:.4f}\n\n")
            
            if positive_distances:
                f.write("Positive Samples:\n")
                f.write(f"Mean: {np.mean(positive_distances):.4f}\n")
                f.write(f"Std: {np.std(positive_distances):.4f}\n")
                f.write(f"Min: {np.min(positive_distances):.4f}\n")
                f.write(f"Max: {np.max(positive_distances):.4f}\n")
                # 임계값 기준 오분류 비율 추가
                misclassified = sum(1 for d in positive_distances if d >= threshold)
                misclassification_rate = misclassified / len(positive_distances)
                f.write(f"False Negative Rate: {misclassification_rate:.4f}\n")
            
            if negative_distances:
                f.write("\nNegative Samples:\n")
                f.write(f"Mean: {np.mean(negative_distances):.4f}\n")
                f.write(f"Std: {np.std(negative_distances):.4f}\n")
                f.write(f"Min: {np.min(negative_distances):.4f}\n")
                f.write(f"Max: {np.max(negative_distances):.4f}\n")
                # 임계값 기준 오분류 비율 추가
                misclassified = sum(1 for d in negative_distances if d < threshold)
                misclassification_rate = misclassified / len(negative_distances)
                f.write(f"False Positive Rate: {misclassification_rate:.4f}\n")
            
            # 임계값 선택 근거 추가
            f.write("\nThreshold Selection Analysis:\n")
            f.write(f"Selected Threshold: {threshold:.4f}\n")
            if positive_distances and negative_distances:
                pos_95th = np.percentile(positive_distances, 95)
                neg_5th = np.percentile(negative_distances, 5)
                f.write(f"Positive 95th percentile: {pos_95th:.4f}\n")
                f.write(f"Negative 5th percentile: {neg_5th:.4f}\n")
                f.write(f"Gap between classes: {neg_5th - pos_95th:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score + self.min_delta:  # < 를 > 로 수정
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
                
def get_train_augmentation(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Lambda(lambda x: x.float() if x.dtype == torch.uint8 else x),
        transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_test_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.Lambda(lambda x: x.float() if x.dtype == torch.uint8 else x),
        transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
def dot_similarity(embeddings, temperature=0.07, eps=1e-8):
    """
    Calculate similarity matrix using dot product with temperature scaling and numerical stability improvements.
    
    Args:
    embeddings: Tensor of shape [batch, n_class, embedding_dim]
    temperature: Float, temperature parameter for scaling
    eps: Float, small value to avoid division by zero

    Returns:
    log_similarity: Tensor of shape [n_class, batch, batch] in log space
    """
    batch_size, n_class, embedding_dim = embeddings.shape
    
    # L2 normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Permute to [n_class, batch, embedding_dim]
    embeddings = embeddings.permute(1, 0, 2)
    
    # Calculate dot product similarity
    similarity = torch.matmul(embeddings, embeddings.transpose(1, 2))
    
    # Apply temperature scaling
    similarity = similarity / temperature
    
    # Apply log-sum-exp trick for numerical stability
    max_sim = similarity.max(dim=-1, keepdim=True)[0]
    exp_sim = torch.exp(similarity - max_sim)
    
    # Compute log of normalization term
    log_normalization = torch.log(exp_sim.sum(dim=-1, keepdim=True) + eps) + max_sim
    
    # Compute log probabilities
    log_similarity = similarity - log_normalization
    
    # Clip values to prevent any remaining instability
    log_similarity = torch.clamp(log_similarity, min=-50, max=50)
    
    return log_similarity

def improved_adaptive_threshold(distances, labels, margin=0.3, alpha=0.2):
    """
    향상된 적응형 임계값 계산 (수정된 버전)
    
    Args:
        distances: 현재 배치의 거리값들 (torch.Tensor)
        labels: 실제 레이블 (torch.Tensor)
        margin: 기본 margin 값
        alpha: 적응 계수 (0에 가까울수록 margin에 가까운 값 사용)
    
    Returns:
        float: 계산된 임계값
    """
    # tensor로 변환
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
        
    positive_distances = distances[labels == 1]
    negative_distances = distances[labels == 0]
    
    if len(positive_distances) > 0 and len(negative_distances) > 0:
        # positive samples의 95 퍼센타일 계산
        pos_threshold = torch.quantile(positive_distances.float(), 0.95)
        
        # negative samples의 5 퍼센타일 계산
        neg_threshold = torch.quantile(negative_distances.float(), 0.05)
        
        # positive와 negative 샘플 간의 간격 계산
        gap = neg_threshold - pos_threshold
        
        # 최종 임계값 계산: margin을 기준으로 데이터 분포에 따라 조정
        adaptive_component = pos_threshold + (gap * 0.5)
        threshold = (1 - alpha) * margin + alpha * adaptive_component.item()
        
        # 임계값의 범위를 제한
        threshold = max(margin * 0.5, min(threshold, margin * 1.5))
        
        return threshold
    
    return margin  # 충분한 데이터가 없을 경우 기본 margin 사용

def compute_robust_prototype(embeddings, k=None):
    """더 강건한 프로토타입 계산"""
    if k is None:
        k = max(len(embeddings) // 2, 1)  # 상위 50% 샘플 사용
    
    # 초기 중심점 계산
    initial_centroid = embeddings.mean(0)
    
    # 중심점과의 거리 계산
    distances = torch.norm(embeddings - initial_centroid, dim=1)
    
    # 가장 가까운 k개의 샘플 선택
    _, indices = distances.topk(k, largest=False)
    selected_embeddings = embeddings[indices]
    
    # 최종 프로토타입 계산
    prototype = selected_embeddings.mean(0)
    return F.normalize(prototype, p=2, dim=0)

def train_epoch(model, dataloader, optimizer, scaler, device, margin=0.3, temperature=0.1):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # 전체 배치에 대한 임베딩을 한 번에 계산
        with autocast(enabled=scaler is not None):
            all_embeddings = model(images)  # [batch, n_class, embedding_dim]
            all_embeddings = F.normalize(all_embeddings, p=2, dim=2)
            
            batch_loss = 0
            optimizer.zero_grad()
            
            for class_idx in range(labels.shape[1]):
                class_labels = labels[:, class_idx]
                embeddings = all_embeddings[:, class_idx, :]
                
                positive_mask = class_labels == 1
                negative_mask = class_labels == 0
                
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    pos_embeddings = embeddings[positive_mask]
                    prototype = compute_robust_prototype(pos_embeddings)
                    
                    # 코사인 거리 계산
                    pos_sim = F.cosine_similarity(pos_embeddings, prototype.unsqueeze(0))
                    neg_sim = F.cosine_similarity(embeddings[negative_mask], prototype.unsqueeze(0))
                    
                    pos_dist = 1 - pos_sim
                    neg_dist = 1 - neg_sim
                    
                    # 하드 네거티브 마이닝
                    k = min(3, len(neg_dist))
                    hardest_neg_dist, _ = neg_dist.topk(k, largest=False)
                    
                    # Loss 계산
                    class_loss = calculate_class_loss(pos_dist, hardest_neg_dist, margin, temperature, device)
                    batch_loss += class_loss
            
            if batch_loss > 0:
                if scaler is not None:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()
                
                total_loss += batch_loss.item()
        
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def compute_class_prototypes(class_embeddings, device):
    """
    각 클래스별 프로토타입을 계산하는 함수
    
    Args:
        class_embeddings: defaultdict, 클래스별 임베딩을 담고 있는 사전
        device: torch.device, 계산을 수행할 디바이스
    
    Returns:
        prototypes: torch.Tensor, 각 클래스의 프로토타입 [n_classes, embedding_dim]
    """
    prototypes = []
    n_classes = max(class_embeddings.keys()) + 1
    
    for class_idx in range(n_classes):
        if class_embeddings[class_idx]:
            # 클래스의 모든 임베딩을 연결
            class_all_embeddings = torch.cat(class_embeddings[class_idx], dim=0)
            
            # 강건한 프로토타입 계산
            prototype = compute_robust_prototype(class_all_embeddings)
            prototypes.append(prototype)
        else:
            # 해당 클래스의 임베딩이 없는 경우
            # 첫 번째 있는 클래스의 임베딩 차원을 사용
            for idx in range(n_classes):
                if class_embeddings[idx]:
                    embedding_dim = class_embeddings[idx][0].shape[-1]
                    break
            prototypes.append(torch.zeros(embedding_dim, device=device))
    
    return torch.stack(prototypes)  # [n_classes, embedding_dim]

def evaluate_with_precomputed_prototypes(model, val_loader, prototypes, device, margin=0.3):
    """
    미리 계산된 프로토타입을 사용하여 평가를 수행하는 함수 (수정된 버전)
    """
    predictions_list = []
    labels_list = []
    distances_per_class = [[] for _ in range(len(prototypes))]
    
    # 클래스별 임계값을 저장할 딕셔너리
    class_thresholds = {}
    
    # 첫 번째 패스: 거리 수집
    for images, labels in tqdm(val_loader, desc="Collecting distances"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        for class_idx in range(len(prototypes)):
            class_emb = embeddings[:, class_idx, :]
            similarities = F.cosine_similarity(class_emb, prototypes[class_idx].unsqueeze(0))
            distances = 1 - similarities
            distances_per_class[class_idx].extend(distances.cpu().numpy())  # numpy로 변환
    
    # 클래스별 임계값 계산
    for class_idx in range(len(prototypes)):
        class_distances = torch.tensor(distances_per_class[class_idx])
        class_labels = torch.cat([labels[:, class_idx] for _, labels in val_loader])
        
        # 향상된 임계값 계산
        threshold = improved_adaptive_threshold(
            class_distances, 
            class_labels, 
            margin=margin,
            alpha=0.2
        )
        class_thresholds[class_idx] = float(threshold)  # float로 변환
    
    # 두 번째 패스: 실제 예측
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        batch_predictions = []
        
        for class_idx in range(len(prototypes)):
            class_emb = embeddings[:, class_idx, :]
            similarities = F.cosine_similarity(class_emb, prototypes[class_idx].unsqueeze(0))
            distances = 1 - similarities
            
            # 클래스별 계산된 임계값 사용
            pred = (distances < class_thresholds[class_idx]).float()
            batch_predictions.append(pred)
        
        batch_predictions = torch.stack(batch_predictions, dim=1)
        predictions_list.append(batch_predictions.cpu())
        labels_list.append(labels.cpu())
    
    # 결과 분석 및 메트릭 계산
    all_predictions = torch.cat(predictions_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # 거리 분포 분석 저장
    save_distance_analysis(
        distances_per_class=distances_per_class,
        labels=all_labels.numpy(),
        thresholds=class_thresholds,
        n_classes=len(prototypes),
        output_dir='output'
    )
    
    metrics = calculate_metrics(all_predictions, all_labels, distances_per_class, prototypes)
    
    # 임계값 정보를 메트릭에 추가
    metrics['thresholds'] = class_thresholds
    
    return metrics

def calculate_metrics(predictions, labels, distances_per_class, prototypes):
    """
    예측 결과에 대한 메트릭을 계산하는 함수
    """
    metrics = {
        'overall': {
            'accuracy': accuracy_score(labels.numpy(), predictions.numpy()),
            'precision': precision_score(labels.numpy(), predictions.numpy(), average='macro'),
            'recall': recall_score(labels.numpy(), predictions.numpy(), average='macro'),
            'f1': f1_score(labels.numpy(), predictions.numpy(), average='macro')
        },
        'per_class': {}
    }
    
    # 클래스별 메트릭 계산
    for class_idx in range(len(prototypes)):
        class_pred = predictions[:, class_idx]
        class_label = labels[:, class_idx]
        
        metrics['per_class'][f'class_{class_idx}'] = {
            'f1': f1_score(class_label.numpy(), class_pred.numpy()),
            'precision': precision_score(class_label.numpy(), class_pred.numpy()),
            'recall': recall_score(class_label.numpy(), class_pred.numpy()),
            'support': class_label.sum().item(),
            'avg_distance': np.mean(distances_per_class[class_idx])
        }
    
    return metrics

def calculate_class_loss(pos_dist, neg_dist, margin, temperature, device):
    """클래스별 loss 계산을 위한 헬퍼 함수"""
    # InfoNCE loss
    logits_pos = -pos_dist / temperature
    logits_neg = -neg_dist / temperature
    labels_contrastive = torch.zeros(pos_dist.size(0), device=device)
    contrastive_loss = nn.CrossEntropyLoss()(
        torch.cat([logits_pos.unsqueeze(1), logits_neg.expand(pos_dist.size(0), -1)], dim=1),
        labels_contrastive.long()
    )
    
    # Triplet loss
    triplet_loss = torch.clamp(
        pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin,
        min=0
    ).mean()
    
    return triplet_loss + 0.5 * contrastive_loss

@torch.no_grad()
def evaluate_with_prototypes(model, train_loader, val_loader, device, margin=0.3):
    model.eval()
    
    # 프로토타입 계산을 배치 단위로 처리
    prototypes = calculate_prototypes(model, train_loader, device)
    
    # 평가 로직
    metrics = evaluate_with_precomputed_prototypes(
        model, val_loader, prototypes, device, margin)
    
    return metrics

def calculate_prototypes(model, loader, device):
    """프로토타입 계산을 위한 헬퍼 함수"""
    class_embeddings = defaultdict(list)
    
    for images, labels in tqdm(loader, desc="Computing prototypes"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=2)
        
        for class_idx in range(labels.shape[1]):
            class_mask = labels[:, class_idx] == 1
            if class_mask.sum() > 0:
                class_emb = embeddings[class_mask, class_idx, :]
                class_embeddings[class_idx].append(class_emb)
    
    return compute_class_prototypes(class_embeddings, device)

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, scaler, device, epochs, temperature=0.07):
    best_f1 = 0.0
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # criterion 제거 (우리는 custom loss를 사용)
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, temperature)
        
        # 새로운 평가 함수 사용
        metrics = evaluate_with_prototypes(model, train_loader, val_loader, device)
        val_f1 = metrics['overall']['f1']
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # 클래스별 성능 출력
        print("\nPer-class Performance:")
        for class_name, class_metric in metrics['per_class'].items():
            print(f"{class_name}:")
            print(f"  F1: {class_metric['f1']:.4f}")
            print(f"  Precision: {class_metric['precision']:.4f}")
            print(f"  Recall: {class_metric['recall']:.4f}")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 모델 저장 시 필요한 정보들도 함께 저장
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'temperature': temperature,
                'class_metrics': metrics['per_class']
            }
            torch.save(save_dict, 'output/best_model.pth')
            print("New best model saved!")
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        print()
    
    # 최종 결과 출력
    print("\nTraining completed!")
    print(f"Best Validation F1 Score: {best_f1:.4f}")
    
    # best model 불러오기
    checkpoint = torch.load('output/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 최종 성능 평가
    final_metrics = evaluate_with_prototypes(model, train_loader, val_loader, device)

    # 클래스별 성능 출력
    print("\nPer-class Performance:")
    for class_name, class_metric in final_metrics['per_class'].items():
        print(f"{class_name}:")
        print(f"  F1: {class_metric['f1']:.4f}")
        print(f"  Precision: {class_metric['precision']:.4f}")
        print(f"  Recall: {class_metric['recall']:.4f}")
    
    print("\nFinal Model Performance:")
    print(f"Overall F1: {final_metrics['overall']['f1']:.4f}")
    print(f"Overall Precision: {final_metrics['overall']['precision']:.4f}")
    print(f"Overall Recall: {final_metrics['overall']['recall']:.4f}")
    
    return best_f1, final_metrics

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mf1 = 0.0
    device = torch.device(cfg['DEVICE'])
    train_cfg, sched_cfg = cfg['TRAIN'], cfg['SCHEDULER']
    dataset_cfg = cfg['DATASET']  
    num_workers = mp.cpu_count()
    
    # print("Episodic dataset is generated")

    # Model definition (changed to binary classification)
    efficientnet = models.efficientnet_v2_m(pretrained=True)
    num_ftrs = efficientnet.classifier[1].in_features
    num_classes = 7
    embedding_dim = 256
    efficientnet.classifier = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        MultiHeadEmbedding(num_ftrs, num_classes, embedding_dim)
    )
    efficientnet = efficientnet.to(device)
    print("Model is initialized")
    
    # dataset
    image_size = [256,256]
    image_dir = Path(dataset_cfg['ROOT']) / 'train_images'
    train_transform = get_train_augmentation(image_size)
    val_test_transform = get_val_test_transform(image_size)
    batch_size = 32


    dataset = eval(dataset_cfg['NAME'])(
        dataset_cfg['ROOT'] + '/cropped_images_1424x1648',
        dataset_cfg['TRAIN_RATIO'],
        dataset_cfg['VALID_RATIO'],
        dataset_cfg['TEST_RATIO'],
        transform=None
    )
    trainset, valset, testset = dataset.get_splits()
    trainset.transform = train_transform
    valset.transform = val_test_transform
    testset.transform = val_test_transform

    trainloader = DataLoader(
        trainset, 
        batch_sampler=BalancedBatchSampler(trainset, batch_size=batch_size),
        num_workers=num_workers,
        pin_memory=True
    )
    #trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)
    testloader = DataLoader(testset, batch_size=1, num_workers=1, pin_memory=True)
    
    sampler = None
    # criterion_cls = get_loss(cfg['LOSS_CLS']['NAME'])
    # criterion_dist_loss = get_loss('DistContrastive')
    # criterion_bce_cls = get_loss('BCEWithLogitsLoss')
    
    # L2 regularization
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(efficientnet.parameters(), lr=0.0001, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=train_cfg['AMP'])
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler(enabled=train_cfg['AMP'])

    epoch = 100
    print("Start Training ...")
    
    best_f1, final_metrics = train_and_evaluate(
        model=efficientnet,  # model -> efficientnet
        train_loader=trainloader,  # train_loader -> trainloader
        val_loader=valloader,  # val_loader -> valloader
        optimizer=optimizer,
        scheduler=scheduler,  # scheduler 추가
        scaler=scaler,
        device=device,
        epochs=epoch,  # epochs -> epoch
        temperature=0.07
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    print(cfg)
    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()