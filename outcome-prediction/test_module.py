import random
from transformers import BertTokenizerFast
import torch
from data_loader import EHRDataset
from transformers import AdamW, BertConfig, BertTokenizer, BertForSequenceClassification, \
        AutoTokenizer, AutoConfig, AutoModel, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from info_nce import InfoNCE, info_nce
import torch.optim as optim
from outcome_models import ContextAwareContrastiveEmbeddingGenerator, BertWithContrastiveLearning
import os
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
def section_preprocess_function(batch, tokenizer, max_length=512):
        """
        对一个 batch 的数据进行分词和编码，优化以利用 tokenizer 的批量处理能力。
        
        参数:
        - batch: list，每个元素是一个 EHR 的信息 [id, sections, label]。
        - tokenizer: 分词器，用于对 section 文本进行编码。
        - max_pos: int，模型的最大输入长度。
        - section_names: list，所有可能的 section 名称。
        
        返回:
        - batch_result: dict，包含分词后的批次数据，包括对缺失部分的处理。
        """
        # 初始化用于保存所有 EHR 分词结果的结构
        '''section_names = [
            'discharge diagnosis', 'major surgical or invasive procedure', 'history of present illness',
            'past medical history', 'brief hospital course', 'chief complaint', 'family history',
            'physical exam', 'admission date', 'discharge date', 'service', 'date of birth',
            'sex', 'allergies', 'social history', 'discharge disposition', 'discharge medications',
            'medications on admission', 'attending', 'discharge condition', 'discharge instructions',
            'followup instructions', 'pertinent results'
        ]'''
        section_names = [
                        'discharge diagnosis', 'major surgical or invasive procedure', 'history of present illness',
                        'past medical history'
                        ]
        batch_result = {
            "ehr_id": [],
            "labels": [],
            "sections": {}
        }

        # 初始化 sections 的结果字典，并创建用于收集每个 section 内容的列表
        section_content = {section_name: [] for section_name in section_names}
        missing_mask = {section_name: [] for section_name in section_names}  # 用于记录哪些文档有缺失的 section

        # Step 1: 遍历 batch，收集各个 section 的内容
        for unit in batch:
            ehr_id = unit[0]
            sections = unit[1]
            label = unit[2]

            # 保存 id 和 label
            batch_result["ehr_id"].append(ehr_id)
            batch_result["labels"].append(label)

            # 遍历所有 section_names 以收集文本内容
            for section_name in section_names:
                if section_name in sections and sections[section_name].strip():
                    # 如果 section 存在且有内容，保存其内容
                    section_content[section_name].append(sections[section_name])
                    missing_mask[section_name].append(False)  # 该 section 存在
                else:
                    # 如果 section 缺失，记录为空
                    section_content[section_name].append("")  # 这里我们仍然添加空字符串以保持顺序
                    missing_mask[section_name].append(True)  # 标记该 section 缺失
        
        
        # Step 2: 使用 tokenizer 对每个 section 的列表进行批量处理
        for section_name in section_names:
            # 过滤掉空字符串进行 batch 分词

            tokenized_output = tokenizer(section_content[section_name], 
                                        padding='max_length', 
                                        max_length=max_length, 
                                        truncation=True,
                                        return_tensors='pt')
            # 初始化 sections 结果
            batch_result["sections"][section_name] = {
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": []
            }

            # Step 3: 遍历编码后的结果，并处理缺失的部分
            for idx, is_missing in enumerate(missing_mask[section_name]):
                if is_missing:
                    # 如果该 section 缺失，填充为全零向量
                    missing_length = max_length
                    batch_result["sections"][section_name]["input_ids"].append(torch.zeros(missing_length, dtype=torch.long))
                    batch_result["sections"][section_name]["attention_mask"].append(torch.zeros(missing_length, dtype=torch.long))
                    batch_result["sections"][section_name]["token_type_ids"].append(torch.zeros(missing_length, dtype=torch.long))
                else:
                    # 如果 section 存在，使用 tokenizer 的输出
                    batch_result["sections"][section_name]["input_ids"].append(tokenized_output["input_ids"][idx])
                    batch_result["sections"][section_name]["attention_mask"].append(tokenized_output["attention_mask"][idx])
                    if "token_type_ids" in tokenized_output:
                        batch_result["sections"][section_name]["token_type_ids"].append(tokenized_output["token_type_ids"][idx])
                    else:
                        # 如果没有 `token_type_ids`，使用全零占位
                        batch_result["sections"][section_name]["token_type_ids"].append(torch.zeros(missing_length, dtype=torch.long))
        
        # Step 4: 将列表转换为张量
        for section_name in section_names:
            for key in batch_result["sections"][section_name]:
                batch_result["sections"][section_name][key] = torch.stack(batch_result["sections"][section_name][key])

        
        # 转换 labels 为张量
        batch_result["labels"] = torch.LongTensor(batch_result["labels"])
        # batch_result = to_cuda(batch_result)
        # verify_cuda_transfer(batch_result)
        return batch_result

def section_batch_and_tokenizer_data(examples, tokenizer, batch_size, split):
        example_list = []
        for file in list(examples.keys()):
            example = examples[file]
            
            example_list.append([file, example['ehr'], example['outcome']])
    
        batches = []

        for i in range(0, len(example_list), batch_size):
            start = i
            end = min(start + batch_size, len(example_list))
            
            # print(len(example_list[start:end]))
            batch = section_preprocess_function(example_list[start:end], tokenizer)
            batches.append(batch)

            if len(batches) % 100 == 0:
                print('Created {} batches'.format(len(batches)), end="\r", flush=True)
        return batches

def test_add_contrastive_data():
        # 模拟一个小的训练集
        train_data = {
            1: {'outcome': 1, 'section_name': 'diagnosis', 'content': 'Patient diagnosed with diabetes.'},
            2: {'outcome': 1, 'section_name': 'diagnosis', 'content': 'Type 2 diabetes confirmed.'},
            3: {'outcome': 1, 'section_name': 'diagnosis', 'content': 'No evidence of diabetes.'},
            4: {'outcome': 1, 'section_name': 'treatment', 'content': 'Started metformin treatment.'},
            5: {'outcome': 0, 'section_name': 'treatment', 'content': 'No treatment prescribed.'},
            6: {'outcome': 0, 'section_name': 'diagnosis', 'content': 'No treatment drug.'},
            7: {'outcome': 0, 'section_name': 'diagnosis', 'content': 'Patient shows no signs of disease.'},
            8: {'outcome': 1, 'section_name': 'diagnosis', 'content': 'Pre-diabetic condition detected.'},
            9: {'outcome': 1, 'section_name': 'treatment', 'content': 'Patient started on insulin.'},
            10: {'outcome': 0, 'section_name': 'treatment', 'content': 'No medication prescribed.'},
            11: {'outcome': 0, 'section_name': 'follow-up', 'content': 'Follow-up visit scheduled in 6 months.'},
            12: {'outcome': 1, 'section_name': 'follow-up', 'content': 'Follow-up visit for glucose levels.'},
            13: {'outcome': 1, 'section_name': 'diagnosis', 'content': 'Gestational diabetes diagnosis.'},
            14: {'outcome': 0, 'section_name': 'diagnosis', 'content': 'Patient cleared of diabetes.'},
            15: {'outcome': 1, 'section_name': 'treatment', 'content': 'Initiated diet and exercise therapy.'},
            16: {'outcome': 0, 'section_name': 'treatment', 'content': 'Patient declined medication.'}
        }



        class MockDataloader:
            def __init__(self, train_data):
                self.train_data = train_data

            def add_contrastive_data(self, task_type='binary', batch_size=7000, k_max=3, max_negatives=10):
                """
                根据任务类型添加对比学习数据，支持动态调整正负样本数量，并增加负样本比例。

                Args:
                    task_type: 任务类型 ('mp', 'los', 'pmv')。
                    batch_size: 每个 batch 的大小。
                    k_max: 每个 anchor 的最大正样本数量。
                    max_negatives: 每个 anchor 最大负样本数量。
                    
                Returns:
                    final_data: 构建好的对比学习数据列表。
                """
                contrastive_data = []
                final_data = []

                # 根据任务类型选择采样逻辑
                if task_type == 'mp':
                    # 分离 majority 和 minority 样本
                    majority_samples = [entry for entry in self.train_data.values() if entry['outcome'] == 0]
                    minority_samples = [entry for entry in self.train_data.values() if entry['outcome'] == 1]

                    # 确保 minority class 样本都参与
                    contrastive_data.extend(minority_samples)

                    # 从 majority class 中采样
                    majority_sample_size = batch_size - len(minority_samples)
                    if len(majority_samples) > majority_sample_size:
                        sampled_majority = random.sample(majority_samples, majority_sample_size)
                    else:
                        sampled_majority = majority_samples
                    contrastive_data.extend(sampled_majority)

                elif task_type in ['los', 'pmv']:
                    # 不分离样本，直接采样整个数据集
                    contrastive_data = random.sample(list(self.train_data.values()), batch_size)

                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                # 遍历采样数据，构建正负样本
                for anchor in contrastive_data:
                    # 过滤掉空的 section
                    if 'content' not in anchor or not anchor['content'].strip():
                        continue  # 跳过空 content 的样本

                    # 提取 anchor 的基本信息
                    anchor_outcome = anchor['outcome']
                    anchor_section_name = anchor['section_name']
                    anchor_content = anchor['content']

                    # 构建正样本候选集
                    positive_candidates = [
                        sample for sample in contrastive_data
                        if sample['outcome'] == anchor_outcome
                        and sample['section_name'] == anchor_section_name
                        and sample != anchor  # 排除自身
                        and sample['content'].strip() != ''
                    ]
                

                    # 动态调整正样本数量
                    k = min(len(positive_candidates), k_max)
                    if k == 0:
                        continue  # 无正样本可选，跳过该 anchor

                    positive_samples = random.sample(positive_candidates, k)
    
                    # 构建负样本类型 1：不同 label 且相同 section_name
                    negative_candidates_different_label = [
                        sample for sample in contrastive_data
                        if sample['outcome'] != anchor_outcome
                        and sample['section_name'] == anchor_section_name
                        and sample['content'].strip() != ''
                    ]

                    # 构建负样本类型 2：相同 label 但不同 section_name
                    negative_candidates_same_label = [
                        sample for sample in contrastive_data
                        if sample['outcome'] == anchor_outcome
                        and sample['section_name'] != anchor_section_name
                        and sample['content'].strip() != ''
                    ]

                    # 合并负样本
                    all_negative_candidates = negative_candidates_different_label + negative_candidates_same_label

                    # 动态调整负样本数量
                    num_negatives = min(len(all_negative_candidates), max_negatives)
                    if num_negatives == 0:
                        continue  # 无负样本可选，跳过该 anchor

                    selected_negatives = random.sample(all_negative_candidates, num_negatives)

                    # 构建最终数据
                    final_data.append({
                        'anchor': anchor_content,
                        'positives': [pos['content'] for pos in positive_samples],
                        'negatives': [neg['content'] for neg in selected_negatives]
                    })
                
                anchors = []
                positives = []
                negatives = []

                for data in final_data:
                    anchors.append(data['anchor'])
                    positives.append(data['positives'])
                    negatives.append(data['negatives'])
                
                final_data = {
                    'anchors':anchors,
                    'positives':positives,
                    'negatives':negatives
                }

                return final_data


def test_loss_function():
    # Case 1
    anchor = torch.tensor([[1.0, 0.0]])
    positives = [[torch.tensor([1.0, 0.0])]]  # 注意这里是 list of list of tensors
    negatives = [[torch.tensor([0.0, 1.0])]]  # 同样是 list of list of tensors
    loss = compute_independent_info_nce_loss(anchor, positives, negatives, temperature=1.0)
    print(f"Case 1 Loss: {loss.item()}")

    # Case 2
    anchor = torch.tensor([[1.0, 0.0]])
    positives = [[torch.tensor([1.0, 0.0]), torch.tensor([0.707, 0.707])]]
    negatives = [[torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]]
    loss = compute_independent_info_nce_loss(anchor, positives, negatives, temperature=1.0)
    print(f"Case 2 Loss: {loss.item()}")

    # Case 3
    anchor = torch.tensor([[1.0, 0.0]])
    positives = [[]]  # 空正样本
    negatives = [[torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]]
    loss = compute_independent_info_nce_loss(anchor, positives, negatives, temperature=1.0)
    print(f"Case 3 Loss: {loss.item()}")



def test_section_segment():
    tokenizer = BertTokenizerFast.from_pretrained(
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        model_max_length=512,
        cache_dir='../cache'
    )
    '''train_path = '/scratch/nkw3mr/sdoh_clinical_outcome_prediction/BEEP/mimic_iii_data/filtered_pmv_data/section/val.json'
    test_path = train_path
    dev_path = train_path
    do_train = True
    do_test = True
    section_segment = True
    dataset = EHRDataset(train_path, dev_path, test_path, do_train, do_test, section_segment)
    print(dataset.train_data['101514'])'''
    example_data = {
        "ehr_1": {
            "ehr": {
                "discharge diagnosis": "Example text for discharge diagnosis",
                "major surgical or invasive procedure": "Example text for major surgical or invasive procedure",
                "history of present illness": "Example text for history of present illness",
                "past medical history": ""
            },
            "outcome": 0
        },
        "ehr_2": {
            "ehr": {
                "discharge diagnosis": "Example text for discharge diagnosis",
                "major surgical or invasive procedure": "Example text for major surgical or invasive procedure",
                "history of present illness": "Example text for history of present illness",
                "past medical history": "Example text for past medical history"
            },
            "outcome": 0
        },
        "ehr_3": {
            "ehr": {
                "discharge diagnosis": "Example text for discharge diagnosis",
                "major surgical or invasive procedure": "Example text for major surgical or invasive procedure",
                "history of present illness": "Example text for history of present illness",
                "past medical history": ""
            },
            "outcome": 1
        },
        "ehr_4": {
            "ehr": {
                "discharge diagnosis": "Example text for discharge diagnosis",
                "major surgical or invasive procedure": "Example text for major surgical or invasive procedure",
                "history of present illness": "Example text for history of present illness",
                "past medical history": "Example text for past medical history"
            },
            "outcome": 1
        },
        "ehr_5": {
            "ehr": {
                "discharge diagnosis": "",
                "major surgical or invasive procedure": "Example text for major surgical or invasive procedure",
                "history of present illness": "Example text for history of present illness",
                "past medical history": "Example text for past medical history"
            },
            "outcome": 0
        }
    }
    train_batches = section_batch_and_tokenizer_data(example_data, tokenizer, 2, 'train')
    print(train_batches[0]['sections']['past medical history']['input_ids'])


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


def main_training_loop(dataloader, model, checkpoint_dir, num_epochs=3, patience=5, alpha=1e-3):
    # writer = SummaryWriter(log_dir='test_contrastive_log_dir')
    info_nce_loss = InfoNCE(negative_mode='paired')
    print_interval = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-5,weight_decay=1e-4)
    best_loss = float('inf')
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs/2, eta_min=1e-6)
    for epoch in range(num_epochs):
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        # current_lr = scheduler.get_last_lr()[0]  # 如果只有一个参数组
        # writer.add_scalar('Learning Rate', current_lr, epoch * len(dataloader) + batch_idx)

        scheduler.step()
        

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
        }, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt'))

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"New best model saved with validation loss: {best_loss:.4f}")
            epochs_no_improve = 0  # 重置早停计数
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
                break
        print("|" * 120)
    # writer.close()


def test_train(model_name_or_path, max_negatives, batch_size, epochs, lr, max_samples_per_outcome, alpha, 
                contrastive_type, section_selection):
    simple_full_ehr_data = {
        "1": {
            "outcome": 1,
            "ehr": {
                "chief complaint": "data-26",
                "discharge instructions": "data-67",
                "family history": "data-24",
                "discharge date": "data-65",
                "history of present illness": "data-42"
            }
        },
        "2": {
            "outcome": 0,
            "ehr": {
                "discharge medications": "data-3",
                "sex": "data-65",
                "chief complaint": "data-69"
            }
        },
        "3": {
            "outcome": 1,
            "ehr": {
                "service": "data-41",
                "brief hospital course": "data-26",
                "admission date": "data-55",
                "allergies": "data-45",
                "discharge disposition": "data-12",
                "past medical history": "data-2"
            }
        },
        "4": {
            "outcome": 1,
            "ehr": {
                "discharge instructions": "data-87",
                "date of birth": "data-50",
                "discharge medications": "data-47",
                "sex": "data-100",
                "service": "data-55"
            }
        },
        "5": {
            "outcome": 0,
            "ehr": {
                "history of present illness": "data-14",
                "discharge diagnosis": "data-39",
                "physical exam": "data-100",
                "attending": "data-34",
                "discharge date": "data-40"
            }
        },
        "6": {
            "outcome": 1,
            "ehr": {
                "admission date": "data-78",
                "followup instructions": "data-31",
                "discharge diagnosis": "data-39",
                "social history": "data-46"
            }
        },
        "7": {
            "outcome": 1,
            "ehr": {
                "discharge instructions": "data-23",
                "medications on admission": "data-52",
                "brief hospital course": "data-100",
                "date of birth": "data-7"
            }
        },
        "8": {
            "outcome": 0,
            "ehr": {
                "admission date": "data-69",
                "past medical history": "data-96",
                "pertinent results": "data-9",
                "social history": "data-25",
                "family history": "data-76"
            }
        },
        "9": {
            "outcome": 1,
            "ehr": {
                "history of present illness": "data-5",
                "physical exam": "data-83",
                "brief hospital course": "data-48",
                "discharge date": "data-78",
                "past medical history": "data-7"
            }
        },
        "10": {
            "outcome": 0,
            "ehr": {
                "discharge date": "data-32",
                "followup instructions": "data-43",
                "date of birth": "data-21"
            }
        },
        "11": {
            "outcome": 0,
            "ehr": {
                "chief complaint": "data-59",
                "brief hospital course": "data-55",
                "discharge date": "data-21",
                "service": "data-19"
            }
        },
        "12": {
            "outcome": 0,
            "ehr": {
                "discharge date": "data-97",
                "service": "data-45",
                "pertinent results": "data-22"
            }
        },
        "13": {
            "outcome": 0,
            "ehr": {
                "family history": "data-86",
                "past medical history": "data-68",
                "pertinent results": "data-85"
            }
        },
        "14": {
            "outcome": 0,
            "ehr": {
                "discharge disposition": "data-39",
                "service": "data-59",
                "physical exam": "data-4",
                "date of birth": "data-78",
                "brief hospital course": "data-55",
                "past medical history": "data-19"
            }
        },
        "15": {
            "outcome": 0,
            "ehr": {
                "admission date": "data-58",
                "social history": "data-3",
                "medications on admission": "data-79",
                "followup instructions": "data-54",
                "discharge diagnosis": "data-96"
            }
        }
    }

    ehr_data = {
            1: {
                'outcome': 0,
                'ehr': {
                    'chief complaint': 'Mild headache for 2 days',
                    'past medical history': 'No known chronic diseases',
                    'physical exam': 'Vitals stable, no apparent abnormalities',
                    'followup instructions': 'Return if headache worsens or new symptoms appear'
                }
            },
            2: {
                'outcome': 1,
                'ehr': {
                    'history of present illness': 'High fever for 3 days with chills',
                    'brief hospital course': 'Admitted for IV antibiotics and observation',
                    'discharge diagnosis': 'Bacterial pneumonia',
                    'discharge medications': 'Oral antibiotics for 7 days'
                }
            },
            3: {
                'outcome': 1,
                'ehr': {
                    'chief complaint': 'Cough and occasional chest pain',
                    'past medical history': 'History of smoking, mild COPD',
                    'physical exam': 'Decreased breath sounds in left lung',
                    'discharge disposition': 'Home with oxygen support',
                    'followup instructions': 'Pulmonary clinic visit in 2 weeks'
                }
            },
            4: {
                'outcome': 0,
                'ehr': {
                    'chief complaint': 'Severe abdominal pain',
                    'major surgical or invasive procedure': 'Appendectomy performed',
                    'brief hospital course': 'Surgery uneventful; pain resolved post-op',
                    'discharge instructions': 'Avoid heavy lifting for 4 weeks'
                }
            },
            5: {
                'outcome': 0,
                'ehr': {
                    'history of present illness': 'Dizziness and fatigue for 1 week',
                    'medications on admission': 'Multivitamins, occasional ibuprofen',
                    'physical exam': 'Blood pressure slightly low, other vitals normal',
                    'discharge medications': 'Oral iron supplements',
                    'discharge disposition': 'Home with dietary advice'
                }
            },
            6: {
                'outcome': 1,
                'ehr': {
                    'past medical history': 'Type 2 Diabetes, hypertension',
                    'chief complaint': 'Blurred vision and elevated blood glucose',
                    'followup instructions': 'Endocrinology consult in 1 week',
                    'discharge diagnosis': 'Poorly controlled diabetes',
                    'medications on admission': 'Metformin, Lisinopril'
                }
            },
            7: {
                'outcome': 1,
                'ehr': {
                    'chief complaint': 'Severe chest pain radiating to left arm',
                    'major surgical or invasive procedure': 'Cardiac catheterization',
                    'discharge instructions': 'Continue dual antiplatelet therapy',
                    'discharge medications': 'Aspirin, Clopidogrel, beta-blockers',
                    'past medical history': 'Hyperlipidemia, CAD'
                }
            },
            8: {
                'outcome': 0,
                'ehr': {
                    'brief hospital course': 'Admitted for observation after fainting episode',
                    'history of present illness': 'Syncope while exercising',
                    'physical exam': 'EKG normal, slight tachycardia noted',
                    'discharge disposition': 'Home after 24-hour monitoring',
                    'followup instructions': 'Cardiology appointment if recurrence'
                }
            }
    }
    dev_data = {
        9: {
            'outcome': 1,
            'ehr': {
                'chief complaint': 'Severe headache and nausea',
                'past medical history': 'Migraine, hypertension',
                'physical exam': 'Neurological exam normal except for photophobia',
                'discharge diagnosis': 'Migraine attack',
                'discharge medications': 'Triptans prescribed',
                'followup instructions': 'Neurology appointment in 2 weeks'
            }
        },
        10: {
            'outcome': 0,
            'ehr': {
                'history of present illness': 'Mild rash on arms and legs',
                'medications on admission': 'Cetirizine for allergies',
                'physical exam': 'Erythematous patches without swelling',
                'discharge disposition': 'Home with antihistamines',
                'followup instructions': 'Monitor for any changes or worsening'
            }
        },
        11: {
            'outcome': 1,
            'ehr': {
                'chief complaint': 'Shortness of breath and wheezing',
                'past medical history': 'Asthma, seasonal allergies',
                'physical exam': 'Decreased breath sounds with wheezing on auscultation',
                'discharge diagnosis': 'Asthma exacerbation',
                'discharge medications': 'Inhaled corticosteroids and bronchodilators',
                'followup instructions': 'Pulmonology consult in 1 week'
            }
        },
        12: {
            'outcome': 0,
            'ehr': {
                'history of present illness': 'Mild lower back pain after lifting heavy objects',
                'medications on admission': 'Ibuprofen as needed',
                'physical exam': 'No neurological deficits, tenderness in lower lumbar region',
                'discharge disposition': 'Home with physical therapy referral',
                'followup instructions': 'Continue pain management and attend physical therapy sessions'
            }
        }
    }


    simple_ehr_data  = {
        1: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '1',
                'past medical history': '2',
                'physical exam': '3'
            }
        },
        2: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '4',
                'past medical history': '5',
                'physical exam': '6'
            }
        },
        3: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '7',
                'past medical history': '8',
                'physical exam': '9'
            }
        },
        4: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '10',
                'past medical history': '11',
                'physical exam': '12'
            }
        }
    }

    
    simple_dev_data = {
        9: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '1',
                'past medical history': '2',
                'physical exam': '3',
                'medications on admission': '4',
                'discharge instructions': '5'
            }
        },
        10: {
            'outcome': 0,
            'ehr': {
                'history of present illness': '6',
                'physical exam': '7',
                'medications on admission': '8',
                'discharge disposition': '9',
                'followup instructions': '10'
            }
        },
        11: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '11',
                'major surgical or invasive procedure': '12',
                'discharge diagnosis': '13',
                'discharge medications': '14',
                'past medical history': '15'
            }
        },
        12: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '16',
                'past medical history': '17',
                'physical exam': '18',
                'discharge instructions': '19',
                'followup instructions': '20'
            }
        },
        13: {
            'outcome': 1,
            'ehr': {
                'history of present illness': '21',
                'brief hospital course': '22',
                'discharge diagnosis': '23',
                'discharge disposition': '24',
                'discharge instructions': '25'
            }
        },
        14: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '26',
                'major surgical or invasive procedure': '27',
                'physical exam': '28',
                'discharge instructions': '29',
                'followup instructions': '30'
            }
        },
        15: {
            'outcome': 1,
            'ehr': {
                'past medical history': '31',
                'chief complaint': '32',
                'followup instructions': '33',
                'discharge diagnosis': '34',
                'medications on admission': '35'
            }
        },
        16: {
            'outcome': 0,
            'ehr': {
                'history of present illness': '36',
                'physical exam': '37',
                'medications on admission': '38',
                'discharge disposition': '39',
                'followup instructions': '40'
            }
        }
    }

    ehr_data_multiclass = {
        1: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '1',
                'past medical history': '2',
                'physical exam': '3'
            }
        },
        2: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '4',
                'past medical history': '5',
                'physical exam': '6'
            }
        },
        3: {
            'outcome': 2,
            'ehr': {
                'chief complaint': '7',
                'past medical history': '8',
                'physical exam': '9'
            }
        },
        4: {
            'outcome': 3,
            'ehr': {
                'chief complaint': '10',
                'past medical history': '11',
                'physical exam': '12'
            }
        },
        5: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '13',
                'past medical history': '14',
                'physical exam': '15'
            }
        },
        6: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '16',
                'past medical history': '17',
                'physical exam': '18'
            }
        },
        7: {
            'outcome': 2,
            'ehr': {
                'chief complaint': '19',
                'past medical history': '20',
                'physical exam': '21'
            }
        },
        8: {
            'outcome': 3,
            'ehr': {
                'chief complaint': '22',
                'past medical history': '23',
                'physical exam': '24'
            }
        },
        9: {
            'outcome': 0,
            'ehr': {
                'chief complaint': '25',
                'past medical history': '26',
                'physical exam': '27'
            }
        },
        10: {
            'outcome': 1,
            'ehr': {
                'chief complaint': '28',
                'past medical history': '29',
                'physical exam': '30'
            }
        }
    }


    out_dir = 'temp'
    tokenizer = BertTokenizerFast.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        cache_dir='../cache'
    )
    config = BertConfig.from_pretrained(
        model_name_or_path,
        cache_dir='../cache'
    )
    model = BertForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config
    )

    if contrastive_type == 'combine':
        model = ContextAwareContrastiveEmbeddingGenerator(config=config, model=model)
    else:
        print("Only Test the Results of the Contrastive Learning")
        model = BertWithContrastiveLearning(config=config, model=model)
    train_path = '/scratch/nkw3mr/sdoh_clinical_outcome_prediction/BEEP/mimic_iii_data/filtered_los_data/section/train.json'
    dev_path = '/scratch/nkw3mr/sdoh_clinical_outcome_prediction/BEEP/mimic_iii_data/filtered_los_data/section/val.json'
    # do_train=True, do_test=True, section_segment=False, do_contrastive=False, task_type='mp'
    dataset = EHRDataset(train_path=train_path, dev_path=dev_path, test_path=train_path,  do_train=True, do_test=False, section_segment=True, task_type='pmv')
    # print(dataset.train_data.keys())
    # print(len(dataset.train_data))
    # processed_data = process_contrastive_samples(final_data)
    # print(dataset.train_data['100018'])
    # dataset.add_contrastive_data()
    


    # important_sections = ['history of present illness', 'physical exam']
    preprocessed_data = dataset.preprocess_ehr_data_with_sampling(simple_full_ehr_data, max_samples_per_outcome, section_selection)
    print(len(preprocessed_data))
    # print(preprocesse)
    # print(preprocessed_data)
    # print(preprocessed_data)
    # dev_preprocessed_data = dataset.preprocess_ehr_data_with_sampling(dataset.dev_data, max_samples_per_outcome)
    # print(dev_preprocessed_data)
    # 构建对比学习数据
    print("Finished Processing the raw data by class")
    print("-"*120)
    contrastive_data = dataset.add_contrastive_data_multiclass_efficient(preprocessed_data=preprocessed_data, max_negatives=max_negatives)
    print(contrastive_data[0])
    print(contrastive_data[1])
    # eval_data = dataset.add_contrastive_data(dev_preprocessed_data, max_negatives=max_negatives)
    print(f"{len(contrastive_data)} train pair, each with {max_negatives} negative data")
    
    exit(0)
    # print(f"{len(eval_data)} eval pair, each with {max_negatives} negative data")
    
    print("Finished Constructing the contrastive data")
    print("-"*120)

    # tokenized_data = [tokenize_and_batch(sample, tokenizer) for sample in contrastive_data]
    tokenized_data = [tokenize_and_batch(sample, tokenizer) for sample in tqdm(contrastive_data)]
    # dev_tokenized_data = [tokenize_and_batch(sample, tokenizer) for sample in tqdm(eval_data)]
    # print(f"Number of tokenized data: {len(dev_tokenized_data)}")
    print("Finished Tokenizing Data")
    print("-"*120)

    out_dir = f"{out_dir}_{batch_size}"
    
    dataloader = DataLoader(tokenized_data, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn,
                            num_workers=0,  # 根据服务器的 CPU 核心数调整
                            pin_memory=True)
    '''eval_dataloader = DataLoader(dev_tokenized_data, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn,
                                num_workers=1,  # 根据服务器的 CPU 核心数调整
                                pin_memory=True)'''
    print("Finished Constructing DataLoader")
    model = model.cuda()
    print("Basic Information")
    print(f"Batch Size: {batch_size}")
    print(f"Number of epoch: {epochs}")
    print(f"{len(contrastive_data)} contrastive pair, each with {max_negatives} negative data")
    print(f"Alpha: {alpha}")
    # print(f"{len(eval_data)} eval pair, each with {max_negatives} negative data")
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("Start Training")
    main_training_loop(dataloader, model, checkpoint_dir, epochs, alpha)

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    print(f"Best Contrastive checkpoint path: {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(model)

    # collated_batch = collate_fn(batches)
    # print(collated_batch)
    
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, action='store', default='/scratch/nkw3mr/sdoh_clinical_outcome_prediction/BEEP/outcome-prediction/UMLS_Bert/umlsbert', help='model path')
    parser.add_argument('--max_negatives', type=int, action='store', default=6, help='number of max negatives used to sample')
    parser.add_argument('--batch_size', type=int, action='store', default=8, help='batch size for contrastive learning')
    parser.add_argument("--epochs", type=int, action='store', default=20, help='epochs used to train')
    parser.add_argument("--lr", type=float, action='store', default=1e-5, help="initial learning rate")
    parser.add_argument("--max_samples_per_outcome", type=int, action="store", default=1000, help="number of samples per outcome")
    parser.add_argument("--alpha", type=float, action='store', default=0.0, help="weight of the disentanglement loss")
    parser.add_argument("--contrastive_type", type=str, action='store', default='single', help='decide if the contrastive module will be combined with anohter')
    parser.add_argument('--section_selection', type=str, action='store', default='full', help='selected sections for contrastive learning')
    args = parser.parse_args()
    args_dict = vars(args)
    test_train(**args_dict)