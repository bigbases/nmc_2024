import torch 
import argparse
import yaml
import time
import multiprocessing as mp
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
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

def train_epoch(model, dataloader, optimizer, scaler, device, margin=0.3, temperature=0.1):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=scaler is not None):
            for class_idx in range(labels.shape[1]):
                class_labels = labels[:, class_idx]
                
                with torch.set_grad_enabled(True):
                    embeddings = model(images)[:, class_idx:class_idx+1, :].clone().squeeze(1)  # [batch, embedding_dim]
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    positive_mask = class_labels == 1
                    negative_mask = class_labels == 0
                    
                    if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                        # 프로토타입 계산 (anchor)
                        prototype = embeddings[positive_mask].mean(0)
                        prototype = F.normalize(prototype, p=2, dim=0)
                        
                        # Positive와 Negative samples와의 거리 계산
                        pos_dist = torch.sum((embeddings[positive_mask] - prototype) ** 2, dim=1)
                        neg_dist = torch.sum((embeddings[negative_mask] - prototype) ** 2, dim=1)
                        
                        # 트리플렛 로스 계산
                        triplet_loss = torch.clamp(
                            pos_dist.unsqueeze(1) - neg_dist.unsqueeze(0) + margin,
                            min=0
                        ).mean()
                        
                        # print(f"\nclass_idx: {class_idx}")
                        # print(f"triplet_loss: {triplet_loss.item():.4f}")
                        # print(f"pos_dist mean: {pos_dist.mean().item():.4f}")
                        # print(f"neg_dist mean: {neg_dist.mean().item():.4f}")
                        
                        if scaler is not None:
                            scaler.scale(triplet_loss).backward()
                        else:
                            triplet_loss.backward()
                        
                        total_loss += triplet_loss.item()
                
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_with_prototypes(model, train_loader, val_loader, device, margin=0.3):
    model.eval()
    
    # 1. 학습 데이터로부터 각 클래스의 prototype 계산
    n_classes = None
    all_embeddings = None
    
    # 첫 배치로 n_classes와 embedding 차원 확인
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        n_classes = labels.shape[1]
        all_embeddings = [[] for _ in range(n_classes)]
        break
    
    # prototype 계산을 위한 embedding 수집
    print("Computing prototypes...")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)  # [batch, n_class, embedding_dim]
        
        for class_idx in range(n_classes):
            class_mask = labels[:, class_idx] == 1
            if class_mask.sum() > 0:
                class_emb = embeddings[class_mask, class_idx, :]
                all_embeddings[class_idx].append(class_emb)
    
    # 각 클래스의 prototype 계산
    prototypes = []
    for class_idx in range(n_classes):
        if all_embeddings[class_idx]:
            class_embeddings = torch.cat(all_embeddings[class_idx], dim=0)
            class_embeddings = F.normalize(class_embeddings, p=2, dim=1)
            prototype = class_embeddings.mean(0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)
        else:
            # embedding_dim 찾기
            for emb_list in all_embeddings:
                if emb_list:
                    embedding_dim = emb_list[0].shape[-1]
                    break
            prototypes.append(torch.zeros(embedding_dim, device=device))
    
    prototypes = torch.stack(prototypes)  # [n_class, embedding_dim]
    
    # 2. 검증 데이터 평가
    predictions_list = []
    labels_list = []
    
    print("Evaluating...")
    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)  # [batch, n_class, embedding_dim]
        
        batch_predictions = []
        for class_idx in range(n_classes):
            class_emb = embeddings[:, class_idx, :]  # [batch, embedding_dim]
            class_emb = F.normalize(class_emb, dim=1)
            
            # prototype과의 거리 계산
            distances = torch.sum((class_emb - prototypes[class_idx]) ** 2, dim=1)
            # 거리가 임계값보다 작으면 해당 클래스로 분류
            pred = (distances < margin).float()
            batch_predictions.append(pred)
        
        batch_predictions = torch.stack(batch_predictions, dim=1)  # [batch, n_class]
        
        predictions_list.append(batch_predictions.cpu())
        labels_list.append(labels.cpu())
    
    # 결과 평가
    all_predictions = torch.cat(predictions_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    metrics = {
        'accuracy': accuracy_score(all_labels.numpy(), all_predictions.numpy()),
        'precision': precision_score(all_labels.numpy(), all_predictions.numpy(), average='macro'),
        'recall': recall_score(all_labels.numpy(), all_predictions.numpy(), average='macro'),
        'f1': f1_score(all_labels.numpy(), all_predictions.numpy(), average='macro')
    }
    
    # 클래스별 메트릭
    class_metrics = {}
    for class_idx in range(n_classes):
        class_pred = all_predictions[:, class_idx]
        class_label = all_labels[:, class_idx]
        
        class_metrics[f'class_{class_idx}'] = {
            'f1': f1_score(class_label.numpy(), class_pred.numpy()),
            'precision': precision_score(class_label.numpy(), class_pred.numpy()),
            'recall': recall_score(class_label.numpy(), class_pred.numpy()),
            'support': class_label.sum().item()
        }
        
        # prototype과의 평균 거리도 추가
        with torch.no_grad():
            class_emb = embeddings[:, class_idx, :]
            class_emb = F.normalize(class_emb, dim=1)
            distances = torch.sum((class_emb - prototypes[class_idx]) ** 2, dim=1)
            class_metrics[f'class_{class_idx}']['avg_distance'] = distances.mean().item()
    
    return {
        'overall': metrics,
        'per_class': class_metrics
    }

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

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=True)
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