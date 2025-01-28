import torch
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from collections import deque
from typing import Dict, Optional, Tuple
import random
def add_contrastive_loss(outputs, model, memory_bank, info_nce_loss, logger, step):
    # 获取特征
    anc_features = outputs[0][0]  # anchor sample
    pos_features = outputs[0][1]  # 正样本特征
    neg_features = outputs[0][2]  # 负样本特征
    '''print(f"Anchor Features type: {type(anc_features)}")
    print(f"Anchor Features shape: {anc_features.shape}")
    print(f"Positive Features type:  {type(pos_features)}")
    print(f"Positive Features shape: {pos_features.shape}")
    print(f"Negative Features type:  {type(neg_features)}")
    print(f"Negative Features shape: {neg_features.shape}")'''
    # 计算损失
    adaptive_weight = model.sigmoid(model.weight)
    contrastive_loss = info_nce_loss(anc_features, pos_features, neg_features)
    ortho_loss = margin_orthogonal_loss(pos_features, neg_features)
    
    return contrastive_loss, adaptive_weight, ortho_loss


def margin_orthogonal_loss(pos_features, neg_features, margin=0.1, epsilon=1e-8):
   # neg_features is already [batch_size, num_neg, feature_dim]
   # pos_features is [batch_size, feature_dim]
   
   diffs = pos_features.unsqueeze(1) - neg_features  # [batch_size, num_neg, feature_dim]
   
   # 归一化
   norms = torch.norm(diffs, dim=2, keepdim=True)
   diffs_normalized = diffs / (norms + epsilon)
   
   # 批量计算内积 [batch_size, num_neg, num_neg]
   inner_products = torch.bmm(
       diffs_normalized,
       diffs_normalized.transpose(1, 2)
   )
   
   # 只取上三角部分
   mask = torch.triu(torch.ones_like(inner_products), diagonal=1)
   loss = torch.mean(torch.relu(torch.abs(inner_products) * mask - margin))
   
   return loss

def check_pairs_labels(anchor_features, positive_features, negative_features, anchor_labels, memory_bank=None):
    batch_size = anchor_features.shape[0]
    
    for i in range(batch_size):
        # 检查positive pair
        if anchor_labels[i] != anchor_labels[i]:  # 使用anchor_labels因为positive sample应该与anchor同类
            print(f"错误: anchor {i} (label={anchor_labels[i]}) 和 positive sample (label={anchor_labels[i]}) 标签不同")
        
        # 检查negative pairs
        for j in range(negative_features.shape[1]):
            if anchor_labels[i] == anchor_labels[i]:  # 同理使用anchor_labels
                print(f"错误: anchor {i} (label={anchor_labels[i]}) 和 negative sample {j} 标签相同")
    
    return True

def construct_pairs(features, labels, memory_bank, n_negatives=4, device='cuda'):
    """
    构造对比学习的样本对，每个anchor对应一个positive和多个negative samples
    Args:
        features: 当前batch的特征 [batch_size, feature_dim]
        labels: 当前batch的标签
        memory_bank: 存储历史特征的memory bank
        n_negatives: 每个anchor选择的负样本数量
    Returns:
        anchors: [N, feature_dim]
        positives: [N, feature_dim]
        negatives: [N, n_negatives, feature_dim]  # 关键改动：增加一个维度
    """
    anchors = []
    pos_features = []
    neg_features_groups = []  # 存储每个anchor的多个negative samples
    anchor_labels = []
    pos_labels = []
    neg_labels_groups = []
    
    for i, (feat, label) in enumerate(zip(features, labels)):
        memory_feats, memory_labels = memory_bank.get_samples(
            current_labels=[label],
            exclude_labels=None
        )
        
        if len(memory_feats) == 0:
            continue
            
        pos_indices = [i for i, l in enumerate(memory_labels) if l == label]
        neg_indices = [i for i, l in enumerate(memory_labels) if l != label]
        
        if len(pos_indices) == 0 or len(neg_indices) < n_negatives:
            continue
        
        # 为每个anchor选择一个positive
        pos_idx = random.choice(pos_indices)
        # 选择多个negatives
        neg_idxs = random.sample(neg_indices, n_negatives)
        
        # 收集样本
        anchors.append(feat.to(device))
        pos_features.append(memory_feats[pos_idx].to(device))
        # 收集这个anchor的所有negative samples
        curr_neg_features = [memory_feats[idx].to(device) for idx in neg_idxs]
        neg_features_groups.append(torch.stack(curr_neg_features))
        
        # 收集标签
        anchor_labels.append(label)
        pos_labels.append(memory_labels[pos_idx])
        neg_labels_groups.append([memory_labels[idx] for idx in neg_idxs])
    
    if not anchors:
        return torch.empty(0,features.size(1)), torch.empty(0,features.size(1)), \
               torch.empty(0,features.size(1),0), [], [], []
    
    # 将列表转换为tensor，注意negative samples多一个维度
    anchors = torch.stack(anchors)  # [N, feature_dim]
    positives = torch.stack(pos_features)  # [N, feature_dim]
    negatives = torch.stack(neg_features_groups)  # [N, n_negatives, feature_dim]
    
    return (anchors, positives, negatives, 
            anchor_labels, pos_labels, neg_labels_groups)

class MemoryBank:
    def __init__(self, feature_dim: int, max_size: int = 8192, momentum: float = 0.9):
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.momentum = momentum
        
        # 为每个类别维护独立的deque
        self.features_per_class = {}  # Dict[label, deque]
        self.class_counts = {}
        
        # 计算每个类别的最大样本数
        self.samples_per_class = max_size // 10  # 假设最多10个类别，可以根据实际情况调整
        self.contrastive_loss_activated = False
    
    def get_samples(self, current_labels: list, n_samples: Optional[int] = None, exclude_labels: list = None):
        if not self.features_per_class:
            return torch.empty(0, self.feature_dim), []
            
        # 收集所有可用样本
        all_features = []
        all_labels = []
        
        exclude_labels = exclude_labels or []
        available_classes = [label for label in self.features_per_class.keys() 
                           if label not in exclude_labels]
        
        if not available_classes:
            return torch.empty(0, self.feature_dim), []
            
        # 从每个类别中收集样本
        for label in available_classes:
            features = list(self.features_per_class[label])
            all_features.extend(features)
            all_labels.extend([label] * len(features))
            
        # 转换为tensor
        features_tensor = torch.tensor(all_features)
        
        # 如果需要限制样本数量
        if n_samples is not None and len(features_tensor) > n_samples:
            # 确保每个类别都有代表
            samples_per_class = n_samples // len(available_classes)
            selected_features = []
            selected_labels = []
            
            for label in available_classes:
                class_indices = [i for i, l in enumerate(all_labels) if l == label]
                selected_indices = class_indices[:samples_per_class]
                selected_features.append(features_tensor[selected_indices])
                selected_labels.extend([label] * len(selected_indices))
            
            features_tensor = torch.cat(selected_features)
            all_labels = selected_labels
            
        return features_tensor, all_labels
        
    def __len__(self) -> int:
        return sum(self.class_counts.values())
    
    def has_sufficient_samples(self, min_samples_per_class: int = 4):
        sufficient = all(count >= min_samples_per_class for count in self.class_counts.values())
        
        # 一旦达到条件就永久激活
        if sufficient and not self.contrastive_loss_activated:
            self.contrastive_loss_activated = True
            print("Contrastive learning activated!")
            
        return self.contrastive_loss_activated or sufficient
    
    def update(self, features: torch.Tensor, labels: list, update_existing: bool = True):
        features = features.detach().cpu()

        for feat, label in zip(features, labels):
            label_key = label.item() if isinstance(label, torch.Tensor) else label  # 转换tensor为普通数字

            if label_key not in self.features_per_class:
                self.features_per_class[label_key] = deque(maxlen=self.samples_per_class)
                self.class_counts[label_key] = 0

            self.features_per_class[label_key].append(feat.tolist())
            self.class_counts[label_key] = len(self.features_per_class[label_key])

    def get_status(self):
        """获取memory bank的当前状态"""
        return {
            'total_samples': len(self),
            'samples_per_class': dict(self.class_counts),
            'max_size_per_class': self.samples_per_class,
            'is_activated': self.contrastive_loss_activated
        }

def move_to_cuda(batch):
    for k, v in batch['anchor'].items():
        if v is not None:
            batch['anchor'][k] = v.cuda()
    for k, v in batch["positives"].items():
        if v is not None:
            batch["positives"][k] = v.cuda()

    for k, v in batch["negatives"].items():
        if v is not None:
            batch["negatives"][k] = v.cuda()
    return batch

def tokenize_and_batch(sample, tokenizer, max_length=512):
    def tokenize(text):
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized_anchor = tokenize(sample["anchor"])
    tokenized_positive = tokenize(sample['positive'])
    tokenized_negatives = [tokenize(neg) for neg in sample["negatives"]]

    return {
        "anchor": tokenized_anchor,
        "positive": tokenized_positive,
        "negatives": tokenized_negatives,
    }

def main_training_loop(dataloader, model, checkpoint_dir, alpha, num_epochs=3, patience=3, temperature=0.1, 
                       contrastive_lr=1e-5, delta=5e-3, trained_contrastive_checkpoint_dir=None):
    # writer = SummaryWriter(log_dir='test_contrastive_log_dir')

    info_nce_loss = InfoNCE(negative_mode='paired', temperature=temperature)
    print_interval = 100
    optimizer = optim.Adam(model.parameters(), lr=contrastive_lr,weight_decay=1e-4)
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    start_epoch = 0
    epochs_no_improve = 0

    if trained_contrastive_checkpoint_dir is not None:
        checkpoint = torch.load(os.path.join(trained_contrastive_checkpoint_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        # optimizer = optim.Adam(model.parameters(), lr=contrastive_lr,weight_decay=1e-4)
        print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")


    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_cuda(batch)
            anchor_enc = batch["anchor"]
            positives_enc = batch["positives"]
            negatives_enc = batch["negatives"]
            
            '''print("Anchor shape:", anchor_enc["input_ids"].shape)
            print("Positives shape:", positives_enc["input_ids"].shape)
            print("Negatives shape:", negatives_enc["input_ids"].shape)'''
            
            anchor_emb, pos_emb, neg_emb = model.contrastive_forward(anchor_enc, positives_enc, negatives_enc)


            batch_size = len(anchor_enc['input_ids'])
            negatives_num = len(neg_emb)
            # print(f"Number of negatives: {negatives_num}")
            negatives_num = int(negatives_num / batch_size)

            # Disentanglement Loss
            if alpha > 0:
                neg_emb_dis = neg_emb.view(batch_size, negatives_num, 768)  # [8, 5, 768]
                # print(f"Disentanglement Negative shape: {neg_emb_dis.shape}")
                pos_emb_dis = pos_emb.unsqueeze(1)
                # print(f"Disentanglement Positive shape: {pos_emb_dis.shape}")

                pos_emb_dis = F.normalize(pos_emb_dis, p=2, dim=2)      # [8, 1, 768]
                neg_emb_dis = F.normalize(neg_emb_dis, p=2, dim=2)      # [8, 5, 768]


                dot_products = torch.bmm(pos_emb_dis, neg_emb_dis.transpose(1, 2))  # [8, 1, 5]
                dot_products = dot_products.view(batch_size, negatives_num)          # [8, 5]
                
                # 选择损失类型（推荐使用平方值）
                orthogonal_loss = torch.mean(dot_products ** 2) 
            else:
                orthogonal_loss = 0
            # print(f"Orthogonal loss: {orthogonal_loss}")  
            
            '''print("Anchor_emb shape:", anchor_emb.shape)
            print("Positives_emb shape:", pos_emb.shape)
            print("Negatives_emb shape:", neg_emb.shape)'''
            

            neg_emb = neg_emb.view(batch_size, int(negatives_num), 768)
            # 3) 计算 InfoNCE loss
            # print(f"shape of pos_emb in train: {pos_emb.shape}")
            
            loss = info_nce_loss(
                anchor_emb,
                pos_emb,
                neg_emb  # (N, M, hidden_dim)
            )
            # print(f"InfoNCE Loss: {loss}")
            loss = loss + alpha * orthogonal_loss
            '''print(f"Combined Loss: {combined_loss}")
            exit(0)'''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # 定期打印当前 step 的 loss
            if (batch_idx + 1) % print_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        # current_lr = scheduler.get_last_lr()[0]  # 如果只有一个参数组
        # writer.add_scalar('Learning Rate', current_lr, epoch * len(dataloader) + batch_idx)

        scheduler.step()
        

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

        '''avg_val_loss = evaluate(model, val_dataloader, info_nce_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Eval Loss: {avg_val_loss:.4f}')'''
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
        }, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt'))

        # 保存最佳模型
        if avg_loss < best_loss - delta:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"New best model saved with training loss: {best_loss:.4f}")
            epochs_no_improve = 0  # 重置早停计数
        else:
            epochs_no_improve += 1
            print(f"No improvement in training loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement in training loss for {patience} consecutive epochs.")
                break


def evaluate(model, val_dataloader, info_nce_loss, alpha=0.1):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in val_dataloader:
            batch = move_to_cuda(batch)
            anchor_enc = batch["anchor"]
            positives_enc = batch["positives"]
            negatives_enc = batch["negatives"]

            anchor_emb, pos_emb, neg_emb = model.contrastive_forward(anchor_enc, positives_enc, negatives_enc)
            # print(f"Shape of pos_emb: {pos_emb.shape}")
            batch_size = len(anchor_enc['input_ids'])
            negatives_num = len(neg_emb)
            # print(f"Number of negatives: {negatives_num}")
            negatives_num = int(negatives_num / batch_size)

            # Disentanglement Loss
            neg_emb_dis = neg_emb.view(batch_size, negatives_num, 768)  # [8, 5, 768]
            # print(f"Disentanglement Negative shape: {neg_emb_dis.shape}")
            pos_emb_dis = pos_emb.unsqueeze(1)
            # print(f"Disentanglement Positive shape: {pos_emb_dis.shape}")

            pos_emb_dis = F.normalize(pos_emb_dis, p=2, dim=2)      # [8, 1, 768]
            neg_emb_dis = F.normalize(neg_emb_dis, p=2, dim=2)      # [8, 5, 768]


            dot_products = torch.bmm(pos_emb_dis, neg_emb_dis.transpose(1, 2))  # [8, 1, 5]
            dot_products = dot_products.view(batch_size, negatives_num)          # [8, 5]
            
            # 选择损失类型（推荐使用平方值）
            orthogonal_loss = torch.mean(dot_products ** 2)
    
            # 重塑 negative embeddings
            neg_emb = neg_emb.view(batch_size, negatives_num, 768)  # [batch_size, num_negatives, 768]
            
            # 计算 InfoNCE loss
            loss = info_nce_loss(
                anchor_emb,
                pos_emb,
                neg_emb  # (N, M, hidden_dim)
            )
            loss = loss + alpha * orthogonal_loss
            total_loss += loss.item()
    
    avg_val_loss = total_loss / len(val_dataloader)
    return avg_val_loss

def collate_fn(batch_list):
    def stack_tokenized(tokenized_list):
        return {
            "input_ids": torch.cat([x["input_ids"] for x in tokenized_list], dim=0),
            "attention_mask": torch.cat([x["attention_mask"] for x in tokenized_list], dim=0),
            "token_type_ids": torch.cat([x["token_type_ids"] for x in tokenized_list], dim=0)
            if "token_type_ids" in tokenized_list[0]
            else None,
        }

    batch = {
        "anchor": stack_tokenized([b["anchor"] for b in batch_list]),
        "positives": stack_tokenized([b['positive'] for b in batch_list]),
        "negatives": stack_tokenized([n for b in batch_list for n in b["negatives"]]),
    }
    return batch