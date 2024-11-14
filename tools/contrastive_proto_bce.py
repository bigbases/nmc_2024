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
from nmc.models.heads import MultiHeadEmbedding, MultiHeadEmbeddingBCE
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

def train_epoch(model, dataloader, optimizer, scaler, criterion_bce, device, temperature=0.1):
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_contra_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        num_classes = labels.shape[1]
        
        # Head별 gradient 계산 및 업데이트
        head_grads = defaultdict(list)
        optimizer.zero_grad()
        
        # 각 클래스별 loss 계산 및 head의 gradient 계산
        for class_idx in range(num_classes):
            with autocast(enabled=scaler is not None):
                # Forward pass with gradient tracking
                embeddings, logits = model(images)
                embeddings = F.normalize(embeddings, p=2, dim=2)
                
                # 현재 클래스의 loss 계산
                class_logits = logits[:, class_idx:class_idx+1]
                class_labels = labels[:, class_idx:class_idx+1]
                bce_loss = criterion_bce(class_logits, class_labels)
                
                class_embeddings = embeddings[:, class_idx, :]
                class_labels_single = labels[:, class_idx]
                
                positive_mask = class_labels_single == 1
                negative_mask = class_labels_single == 0
                
                contra_loss = torch.tensor(0.0).to(device)
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    pos_embeddings = class_embeddings[positive_mask]
                    neg_embeddings = class_embeddings[negative_mask]
                    
                    contra_loss = calculate_class_contra_loss(
                        pos_embeddings=pos_embeddings,
                        neg_embeddings=neg_embeddings,
                        temperature=temperature
                    )
                
                class_loss = bce_loss + 0.1 * contra_loss
                
                # Backward pass for current head
                if scaler is not None:
                    scaler.scale(class_loss).backward(retain_graph=(class_idx < num_classes-1))
                else:
                    class_loss.backward(retain_graph=(class_idx < num_classes-1))
                
                # Store gradients for heads
                for name, param in model.named_parameters():
                    if 'classifier' in name and f'classifier.{class_idx}' in name:
                        if param.grad is not None:
                            head_grads[name].append(param.grad.clone())
                
                # 현재 head의 gradient 초기화
                for name, param in model.named_parameters():
                    if 'classifier' in name:
                        if param.grad is not None:
                            param.grad.zero_()
                
                total_loss += class_loss.item()
                total_bce_loss += bce_loss.item()
                total_contra_loss += contra_loss.item()
        
        # Backbone과 head의 gradient 업데이트
        with torch.no_grad():
            # Head gradients 복원
            for name, param in model.named_parameters():
                if 'classifier' in name and name in head_grads:
                    param.grad = head_grads[name][0]
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / (n_batches * num_classes),
        'bce_loss': total_bce_loss / (n_batches * num_classes),
        'contra_loss': total_contra_loss / (n_batches * num_classes)
    }


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

def calculate_class_contra_loss(pos_embeddings, neg_embeddings, temperature):
    """클래스별 대조 손실 계산"""
    # Positive pair similarities
    pos_sim = torch.matmul(pos_embeddings, pos_embeddings.t()) / temperature
    
    # Negative pair similarities
    neg_sim = torch.matmul(pos_embeddings, neg_embeddings.t()) / temperature
    
    # Hard negative mining
    k = min(3, len(neg_sim))
    hardest_neg_sim, _ = neg_sim.topk(k, largest=True, dim=1)
    
    # Loss calculation
    pos_loss = -torch.log(
        torch.exp(pos_sim.diag()) / 
        (torch.exp(pos_sim.diag()) + torch.sum(torch.exp(hardest_neg_sim), dim=1))
    )
    
    return pos_loss.mean()

@torch.no_grad()
def evaluate_with_prototypes(model, train_loader, val_loader, device, margin=0.3):
    model.eval()
    predictions_list = []
    labels_list = []
    
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        embeddings, logits = model(images)  # logits도 받아옴
        
        # BCE로 학습된 logits로 예측
        predictions = torch.sigmoid(logits) > 0.5
        
        predictions_list.append(predictions.cpu())
        labels_list.append(labels.cpu())
    
    all_predictions = torch.cat(predictions_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    metrics = {
        'overall': {
            'accuracy': accuracy_score(all_labels.numpy(), all_predictions.numpy()),
            'precision': precision_score(all_labels.numpy(), all_predictions.numpy(), average='macro'),
            'recall': recall_score(all_labels.numpy(), all_predictions.numpy(), average='macro'),
            'f1': f1_score(all_labels.numpy(), all_predictions.numpy(), average='macro')
        },
        'per_class': {}
    }
    
    # 클래스별 메트릭 계산
    for class_idx in range(all_labels.shape[1]):
        class_pred = all_predictions[:, class_idx]
        class_label = all_labels[:, class_idx]
        
        metrics['per_class'][f'class_{class_idx}'] = {
            'f1': f1_score(class_label.numpy(), class_pred.numpy()),
            'precision': precision_score(class_label.numpy(), class_pred.numpy()),
            'recall': recall_score(class_label.numpy(), class_pred.numpy()),
            'support': class_label.sum().item()
        }
    
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

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, scaler,
                      criterion_bce, device, epochs, temperature=0.07):
    best_f1 = 0.0
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 현재 learning rates 출력
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Learning rates - Backbone: {current_lrs[0]:.6f}, Head: {current_lrs[1]:.6f}")
        
        # Training
        loss_dict = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion_bce=criterion_bce,
            device=device,
            temperature=temperature
        )
        
        # Validation
        metrics = evaluate_with_prototypes(model, train_loader, val_loader, device)
        val_f1 = metrics['overall']['f1']
        
        print(f"Training BCE Loss: {loss_dict['bce_loss']:.4f}")
        print(f"Training Contrastive Loss: {loss_dict['contra_loss']:.4f}")
        print(f"Training Total Loss: {loss_dict['total_loss']:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Learning rate 조정
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'temperature': temperature,
                'class_metrics': metrics['per_class'],
                'loss_dict': loss_dict
            }
            torch.save(save_dict, 'output/best_model.pth')
            print("New best model saved!")
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return best_f1, metrics


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
    efficientnet.classifier = MultiHeadEmbeddingBCE(num_ftrs, num_classes, embedding_dim)
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
    
    # 파라미터 그룹 분리
    backbone_params = []
    head_params = []
    
    for name, param in efficientnet.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    # Optimizer 설정
    optimizer = torch.optim.AdamW([
        {
            'params': backbone_params,
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        {
            'params': head_params,
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
    ])
    
    
    
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=train_cfg['AMP'])
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler(enabled=train_cfg['AMP'])

    epoch = 100
    print("Start Training ...")
    
    best_f1, final_metrics = train_and_evaluate(
        model=efficientnet,
        train_loader=trainloader,
        val_loader=valloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criterion_bce=criterion_bce,  # criterion_bce를 직접 전달
        device=device,
        epochs=epoch,
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