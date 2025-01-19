import torch
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

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
                       contrastive_lr=1e-5, delta=5e-3, contrastive_checkpoint_dir=None):
    # writer = SummaryWriter(log_dir='test_contrastive_log_dir')
    info_nce_loss = InfoNCE(negative_mode='paired', temperature=temperature)
    print_interval = 100
    optimizer = optim.Adam(model.parameters(), lr=contrastive_lr,weight_decay=1e-4)
    best_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    start_epoch = 0
    epochs_no_improve = 0

    if contrastive_checkpoint_dir is not None:
        checkpoint = torch.load(os.path.join(contrastive_checkpoint_dir, 'best_model.pt'))
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