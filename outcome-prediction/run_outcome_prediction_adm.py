import argparse
import random
import os
import pickle
import copy
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, \
    RocCurveDisplay, PrecisionRecallDisplay, \
    precision_score, recall_score, precision_recall_curve, roc_curve
from matplotlib import pyplot as plt
import pandas as pd
import setproctitle
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import EHRDataset
from transformers import AdamW, BertConfig, BertTokenizer, BertForSequenceClassification, \
        AutoTokenizer, AutoConfig, AutoModel, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from outcome_models import BertLongForSequenceClassification, LitAugPredictorBienc, LitAugPredictorCrossenc, L2RLitAugPredictorBienc, \
        OrthAwareMissingEmbeddingGenerator, MOEOrthAwareMissingEmbeddingGenerator
# from info_nce import InfoNCE, info_nce
# from contrastive_utils import tokenize_and_batch, main_training_loop, move_to_cuda, collate_fn, evaluate, MemoryBank, margin_orthogonal_loss, add_contrastive_loss
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 配置logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import wandb

wandb.login(key="d7c68693ef5c5723e30df705c6b36f60fc48fb85")


def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU. torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def to_cuda(batch):
        """
        递归地将字典中的所有PyTorch张量移动到CUDA设备上
        
        Args:
            batch: 字典、列表、元组或张量
            
        Returns:
            移动到CUDA设备上的相同结构数据
        """
        cuda_batch = {
        "ehr_id": batch["ehr_id"],
        "labels": batch["labels"],
        "sections": {}
        }
    
        # 处理sections中的每个子字典
        for section_name, section_dict in batch["sections"].items():
            cuda_batch["sections"][section_name] = {}
            # 处理每个section中的张量
            for key, value in section_dict.items():
                if isinstance(value, torch.Tensor):
                    cuda_batch["sections"][section_name][key] = value.cuda()
                else:
                    cuda_batch["sections"][section_name][key] = value
                    
        return cuda_batch

def create_long_model(init_model, save_model_to, attention_window, max_pos, num_labels):
    config = BertConfig.from_pretrained(init_model,
                                        num_labels=num_labels,
                                        label2id={x:x for x in range(num_labels)},
                                        id2label={x:x for x in range(num_labels)},
                                        cache_dir='../cache'
    )
    model = BertForSequenceClassification.from_pretrained(init_model, config=config, cache_dir='../cache')
    tokenizer = BertTokenizerFast.from_pretrained(init_model, model_max_length=max_pos, cache_dir='../cache')
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight
        k += step
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed
    model.bert.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)


def train(model, train_data, dev_data, out_dir, epochs, lr, class_weights, acc_steps, strategy,
          use_warmup, warmup_steps, stop_on_roc, dump_test_preds, section_segment, orth_weight):
    # print('Dropout default" {}'.format(model.config.hidden_dropout_prob))

    weights = torch.cuda.FloatTensor([x[1] for x in list(sorted(class_weights.items(), key=lambda x:x[0]))])
    weighted_ce_loss = nn.CrossEntropyLoss(weight=weights)
    print(f"Start Downstream Task Fine Tuning")
    if 'vote' in strategy:
        weighted_ce_loss = nn.NLLLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    if use_warmup:
        # optimizer = AdamW(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        print('Using linear schedule with warmup for {} steps'.format(warmup_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, epochs*len(train_data))
    prev_dev_loss = 10000
    prev_auroc = -10000
    # print(train_data)
    
    batch_size = len(train_data[0]['ehr_id'])
    acc_factor = acc_steps/batch_size
    val_steps = batch_size * (800 // batch_size) # just to be in line with the originally 800 steps (of batch size 1)
    for epoch in range(epochs):
        step = 0
        random.shuffle(train_data)
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()
        num_batches = len(train_data)
        num_train_examples = num_batches * batch_size
        for batch in train_data:

            if section_segment == False:
                gpu_batch = {x:y.cuda() for x,y in batch.items()
                            if x not in ['ehr_id', 'pubmed_docs', 'pubmed_doc_weights',
                                        'ehr_rerank_tokens', 'pubmed_doc_ids']}
            
                if 'pubmed_docs' in batch:
                    gpu_batch['pubmed_docs'] = batch['pubmed_docs']
                    gpu_batch['pubmed_doc_weights'] = batch['pubmed_doc_weights']
                if 'pubmed_doc_ids' in batch:
                    gpu_batch['pubmed_doc_ids'] = batch['pubmed_doc_ids']
                if 'ehr_rerank_tokens' in batch:
                    gpu_batch['ehr_rerank_tokens'] = {x:y.cuda() for x,y in batch['ehr_rerank_tokens'].items()}
        
                outputs = model(**gpu_batch)
            else:
                gpu_batch = to_cuda(batch)
                outputs = model(gpu_batch)


            logits = outputs[1]
            wloss = weighted_ce_loss(logits, gpu_batch["labels"])
            metrics = {
                "CrossEntropyLoss": wloss.item(),
                "epoch": epoch + 1
            }
            '''print(f"Memory Bank Status: {memory_bank.has_sufficient_samples(min_samples_per_class=4)}")
            print(f"Memory Bank class: {memory_bank.class_counts}")'''
            if isinstance(outputs[0], tuple):
                ortho_loss = outputs[0][0]
                if ortho_loss != None:

                    # contrastive_loss = info_nce_loss(outputs[0]) + model.sigmoid(model.weight) * margin_orthogonal_loss(outputs[0][1], outputs[0][2]) 
                    wloss += orth_weight * ortho_loss
                    
                    # 记录额外的指标
                    '''wandb.log({
                        "contrastive_loss": contrastive_loss.item(),
                        "total_samples_in_memory": len(memory_bank),
                        "step": step ,
                        "epoch": epoch + 1,
                        "adaptive_weight": adaptive_weight,
                        "ortho_loss": ortho_loss
                    })'''

                    metrics.update({
                        "orthogonal_loss": ortho_loss,
                        "train_loss": wloss.item() / acc_factor
                    })
            wloss /= acc_factor
            epoch_loss += wloss.item()
            wloss.backward()
            step += batch_size
            if step%acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            metrics.update({
                "train_loss": wloss.item(),
                "step": step
            })
            wandb.log(metrics)


            if step%val_steps == 0:
                print('Completed {}/{} training steps'.format(step, num_train_examples))
                dev_loss, auroc = test(model, dev_data, dump_test_preds, out_dir, epoch, step=step,
                                       return_loss=True, class_weights=class_weights, strategy=strategy, section_segment=section_segment, orth_weight=orth_weight)
                if not stop_on_roc and dev_loss < prev_dev_loss: # stop on loss
                    prev_dev_loss = dev_loss
                    torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
                if stop_on_roc and auroc > prev_auroc: # stops on AUROC
                    prev_auroc = auroc
                    torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
                if not use_warmup:
                    if not stop_on_roc:
                        scheduler.step(dev_loss)
                    else:
                        scheduler.step(auroc)
                else:
                    print('Different step for linear warmup')
                    scheduler.step()
                end_so_far = time.time()
                elapsed_so_far = end_so_far - start_time
                ETA = 1 / (step/num_train_examples) * elapsed_so_far
                print(f"{step} steps + validation took {elapsed_so_far//60} minutes, ETA for epoch: {ETA//60} minutes")
        epoch_loss /= (len(train_data)/acc_factor)
        print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
        '''torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))'''
        dev_loss, auroc = test(model, dev_data, dump_test_preds, out_dir, epoch, step="end",
                               return_loss=True, class_weights=class_weights, strategy=strategy, section_segment=section_segment, orth_weight=orth_weight)
        if dev_loss < prev_dev_loss:
            prev_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
        scheduler.step(dev_loss)


def test(model, dev_data, dump_test_preds, out_dir, epoch, step,
         return_loss=False, class_weights=None, strategy='average', section_segment=False, orth_weight=0.1):
    with torch.no_grad():
        model.eval()
        unique_labels = list(class_weights.keys())
        weights = torch.cuda.FloatTensor([x[1] for x in list(sorted(class_weights.items(), key=lambda x:x[0]))])
        weighted_ce_loss = nn.CrossEntropyLoss(weight=weights)
        if 'vote' in strategy:
            weighted_ce_loss = nn.NLLLoss(weight=weights)
        softmax = nn.Softmax(dim=1)
        dev_loss = 0.0
        all_preds = []
        all_pred_probs = []
        all_labels = []
        all_ids = []
        all_pred_probs_dump = []
        for i, batch in enumerate(dev_data):
            if section_segment == False:
                gpu_batch = {x:y.cuda() for x,y in batch.items() if x not in ['ehr_id', 'pubmed_docs', 'pubmed_doc_weights', 'ehr_rerank_tokens', 'pubmed_doc_ids']}
                if 'pubmed_docs' in batch:
                    gpu_batch['pubmed_docs'] = batch['pubmed_docs']
                    gpu_batch['pubmed_doc_weights'] = batch['pubmed_doc_weights']
                if 'pubmed_doc_ids' in batch:
                    gpu_batch['pubmed_doc_ids'] = batch['pubmed_doc_ids']
                if 'ehr_rerank_tokens' in batch:
                    gpu_batch['ehr_rerank_tokens'] = {x:y.cuda() for x,y in batch['ehr_rerank_tokens'].items()}
                outputs = model(**gpu_batch)
            else:
                gpu_batch = to_cuda(batch)
                outputs = model(gpu_batch)
                
            
            
            logits = outputs[1]
            all_preds += torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            probs = softmax(logits) if 'average' in strategy else torch.exp(logits)
            all_pred_probs_dump += probs.detach().cpu().numpy().tolist()
            probs = probs if len(unique_labels) > 2 else probs[:,1]
            
            wloss = weighted_ce_loss(logits, gpu_batch["labels"])
            if isinstance(outputs[0], tuple):
                ortho_loss = outputs[0][0]
                if ortho_loss != None:
                    wloss += orth_weight * ortho_loss
                    
                    # 记录额外的指标
                    '''wandb.log({
                        "contrastive_loss": contrastive_loss.item(),
                        "total_samples_in_memory": len(memory_bank),
                        "step": step ,
                        "epoch": epoch + 1,
                        "adaptive_weight": adaptive_weight,
                        "ortho_loss": ortho_loss
                    })'''
                    
                    # 记录额外的指标
                    '''wandb.log({
                        "validation_contrastive_loss": contrastive_loss.item(),
                        "total_samples_in_memory": len(memory_bank),
                        "step": step,
                        "epoch": epoch + 1,
                        "adaptive_weight": adaptive_weight,
                        "valiadation_ortho_loss": ortho_loss
                    })'''
            dev_loss += wloss.item()
            all_pred_probs += probs.detach().cpu().numpy().tolist()
            all_labels += gpu_batch["labels"].cpu().numpy().tolist()
            all_ids += batch['ehr_id']
            print(f"completed {i+1}/{len(dev_data)} validation batches", end="\r", flush=True)
    prediction_dict = dict(zip(all_ids, all_preds))
    pred_prob_dict = dict(zip(all_ids, all_pred_probs_dump))
    if not return_loss and dump_test_preds:   # return_loss flag is not used for the test data
        pickle.dump(prediction_dict, open(os.path.join(out_dir, 'test_predictions.pkl'), 'wb'))
        pickle.dump(pred_prob_dict, open(os.path.join(out_dir, 'test_probabilities.pkl'), 'wb'))
    metrics_dict = compute_classification_metrics(all_preds, all_pred_probs, all_labels, epoch, step, out_dir)
    dev_loss /= len(dev_data)
    print('Validation loss after epoch {}: {}'.format(epoch, dev_loss))
    print('------------------Validation Scores for Epoch {}-------------------'.format(epoch))
    for metric_name, metric_value in metrics_dict.items():
        print(f'{metric_name}: {metric_value}')

    # 记录验证指标到 WandB
    # 设置映射逻辑：如果是字符串，则映射为固定的数值
    # 初始化 log_data
    log_data = {}

    # 检查 epoch 和 step 是否为特殊标记
    if epoch == "end" and step == "test":
        # 将 metrics_dict 的结果归类到 test/ 下
        for metric_name, metric_value in metrics_dict.items():
            metric_name = str(metric_name)
            if isinstance(metric_value, dict):
                metric_value = {str(k): float(v) for k, v in metric_value.items()}
            elif hasattr(metric_value, 'item'):
                metric_value = metric_value.item()
            log_data[f"test/{metric_name}"] = metric_value
        
        # 记录验证损失到 test/ 下
        log_data["test/validation_loss"] = float(dev_loss)

    else:
        # 正常记录训练阶段的结果
        log_data = {
            "validation_loss": float(dev_loss),
            "epoch": int(epoch) if isinstance(epoch, int) else str(epoch),
            "step": int(step) if isinstance(step, int) else str(step),
        }

        # 将 metrics_dict 的结果添加到 log_data 中
        for metric_name, metric_value in metrics_dict.items():
            metric_name = str(metric_name)
            if isinstance(metric_value, dict):
                metric_value = {str(k): float(v) for k, v in metric_value.items()}
            elif hasattr(metric_value, 'item'):
                metric_value = metric_value.item()
            log_data[f"validation/{metric_name}"] = metric_value

    # 记录到 wandb
    wandb.log(log_data)



    auroc = metrics_dict["ROC AUC"]
    if return_loss:
        return dev_loss, auroc


def prec_rec_at_k(k, labels, probs):
    from collections import Counter
    if k>=1:
        k = k/100.0
    print(f"if we take only those in top {k} AND P(y(x)==class)>0.5")
    for class_num in set(labels.values()):  # range(len(probs[list(probs.keys())[0]])): ???
        select_count = int(k * len(probs))
        # print(f"Probs: {probs}")
        top_probs = list(reversed(sorted(probs.items(), key=lambda x: x[1][class_num])))[:select_count]
        # print(top_probs)
        eval_probs = [probs[x[0]] for x in top_probs]
        eval_labels = [labels[x[0]] for x in top_probs]
        # print(f"Class_num: {class_num}")
        eval_probs = np.array(eval_probs)
        eval_preds = np.argmax(eval_probs, axis=1)
        # print(f"Eval_probs: {eval_probs}")
        # print(f"Eval_preds: {eval_preds}")
        # print(f"Eval_labels: {eval_labels}")
        # exit(0)
        correct = 0.0
        all_labels = list(labels.values())
        class_counter = Counter(all_labels)
        pred_counter = Counter(eval_preds.tolist())
        for x, y in zip(eval_preds, eval_labels):
            if x == y and x == class_num:
                correct += 1
        print('Class {}'.format(class_num))
        if correct != 0.0:
            print('Precision@10: {}'.format((correct / pred_counter[class_num]) ))
            print('Recall@10: {}'.format((correct / class_counter[class_num]) ))
        else:
            print('Precision and recall are 0.0!!')
        print()
    print()

    print(f"if we take all those in top {k} as belonging to that class")
    for class_num in set(labels.values()):  # range(len(probs[list(probs.keys())[0]])): ???
        select_count = int(k * len(probs))
        top_probs = list(reversed(sorted(probs.items(), key=lambda x: x[1][class_num])))[:select_count]
        eval_probs = [probs[x[0]] for x in top_probs]
        eval_labels = [labels[x[0]] for x in top_probs]
        eval_probs = np.array(eval_probs)
        #eval_preds = np.argmax(eval_probs, axis=1) # todo: replace that...
        eval_preds = class_num * np.ones(select_count) # todo: ... by that
        correct = 0.0
        all_labels = list(labels.values())
        class_counter = Counter(all_labels)
        pred_counter = Counter(eval_preds.tolist())
        for x, y in zip(eval_preds, eval_labels):
            if x == y and x == class_num:
                correct += 1
        print('Class {}'.format(class_num))
        if correct != 0.0:
            print('Precision@10: {}'.format((correct / pred_counter[class_num])))
            print('Recall@10: {}'.format((correct / class_counter[class_num])))
        else:
            print('Precision and recall are 0.0!!')
        print()
    print()

def compute_classification_metrics(preds, probs, labels, epoch, step, out_dir):
    unique_labels, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    p_labels = counts / sum(counts)
    is_binary = len(unique_labels) == 2
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.array(preds)
    print(f"Is the task binary classification: {is_binary}")

    # print("precision/recall @10 using Aakanasha's function:")


    if not is_binary:
        '''unique_labels, counts = np.unique(labels, return_counts=True)
        # probs = np.array(probs)
        # labels = np.array(labels)
        # preds = np.array(preds)
        roc_auc = roc_auc_score(y_true=labels, y_score=probs, average="macro", multi_class="ovo")
        f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        mf1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        weightedf1 = f1_score(y_true=labels, y_pred=preds, average="weighted")
        # print(f"Probs: {probs}")
        probs_for_aakanasha = {i: probs[i] for i in range(n)}
        labels_for_aakanasha = {i: labels[i] for i in range(n)}
        # print(f"Probs for aakanasha: {probs_for_aakanasha}")
        prec_rec_at_k(0.1, labels_for_aakanasha, probs_for_aakanasha)
        
        metric_dict = {}
        # metric_dict["precision (using threshold 0.5)"] = precision_score(y_true=labels, y_pred=preds, average=None)
        # metric_dict["recall (using threshold 0.5)"] = recall_score(y_true=labels, y_pred=preds, average=None)
        metric_dict['ROC AUC'] = roc_auc
        metric_dict["micro_f1"] = f1
        metric_dict["macro_f1"] = mf1
        metric_dict["weighted_f1"] = weightedf1a'''
        unique_labels = set(labels)
        
        probs = np.array(probs)
        labels = np.array(labels)
        preds = np.array(preds)
        roc_auc = roc_auc_score(y_true=labels, y_score=probs, average="macro", multi_class="ovo")
        f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        mf1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        weightedf1 = f1_score(y_true=labels, y_pred=preds, average="weighted")

        probs_for_aakanasha = {i: probs[i] for i in range(n)}
        labels_for_aakanasha = {i: labels[i] for i in range(n)}
        prec_rec_at_k(0.1, labels_for_aakanasha, probs_for_aakanasha)
        

        metric_dict = {}
        # metric_dict["weighted_f1"] = weightedf1a
        metric_dict['ROC AUC'] = roc_auc
        metric_dict["micro_f1"] = f1
        metric_dict["macro_f1"] = mf1
        metric_dict["weighted_f1"] = weightedf1

        # data_distribution = {unique_labels[i]: round(100*p_labels[i], 2) for i in range(len(unique_labels))}
        # metric_dict["true distribution"] = data_distribution

        '''for metric_name, metric_value in metrics_dict.items():
            print(f'{metric_name}: {metric_value}')'''
        return metric_dict
    
    print("precision/recall @10 using Aakanasha's function:")
    probs_for_aakanasha = {i: [1 - probs[i], probs[i]] for i in range(n)}
    labels_for_aakanasha = {i: labels[i] for i in range(n)}
    prec_rec_at_k(0.1, labels_for_aakanasha, probs_for_aakanasha)

    argsort_probs = np.argsort(probs)
    probs = probs[argsort_probs]
    preds = preds[argsort_probs]
    labels = labels[argsort_probs]

    metrics_dict = {}

    # from the documentation: average parameter will be ignored if y_true is binary.
    # in this case it's auprc and roc_auc w.r.t. class 1
    auprc = average_precision_score(y_true=labels, y_score=probs, average="macro")

    metrics_dict["precision (using threshold 0.5)"] = precision_score(y_true=labels, y_pred=preds)
    metrics_dict["recall (using threshold 0.5)"] = recall_score(y_true=labels, y_pred=preds)

    metrics_dict["AUPRC"] = auprc

    roc_auc = roc_auc_score(y_true=labels, y_score=probs, average="macro", multi_class="ovo")
    metrics_dict["ROC AUC"] = roc_auc

    if is_binary:
        f1 = f1_score(y_true=labels, y_pred=preds)
        metrics_dict["f1 (w.r.t. class 1)"] = f1

    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    metrics_dict["micro_f1"] = micro_f1
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    metrics_dict["macro_f1"] = macro_f1
    weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    metrics_dict["weighted_f1"] = weighted_f1

    """
    fig, ax = plt.subplots()
    df = pd.DataFrame(data={"prob": probs, "class": labels})
    df.groupby("class").prob.plot(kind='kde', ax=ax)
    plt.legend()
    plt.xlim((-0.05, 1.05))
    plt.xlabel("p")
    title = f"separation between the classes, epoch: {epoch},  step: {step}"
    plt.title(title)
    separation_path = os.path.join(out_dir, f"separation_epoch_{epoch}_step_{step}.jpg")
    plt.savefig(separation_path)
    plt.show(block=False)
    plt.close()
    """


    data_distribution = {unique_labels[i]: round(100*p_labels[i], 2) for i in range(len(unique_labels))}
    metrics_dict["true distribution"] = data_distribution

    if is_binary:

        true_share_of_1 = np.mean(labels)
        true_share_of_1_size = int(true_share_of_1 * len(labels))
        true_percent_of_1 = np.round(100*true_share_of_1, 2)
        true_share_of_0 = 1-np.mean(labels)
        true_share_of_0_size = int(true_share_of_0 * len(labels))
        true_percent_of_0 = np.round(100*true_share_of_0, 2)

        precision_at_true_percent_class_0 = sum(labels[:true_share_of_0_size]==0) / true_share_of_0_size
        recall_at_true_percent_class_0 = sum(labels[:true_share_of_0_size]==0) / sum(labels==0)
        #metrics_dict[f"precision@{true_percent_of_0} (true percent) (class 0)"] = precision_at_true_percent_class_0
        #metrics_dict[f"recall@{true_percent_of_0} (true percent) (class 0)"] = recall_at_true_percent_class_0

        precision_at_true_percent_class_1 = sum(labels[-true_share_of_1_size:]==1) / true_share_of_1_size
        recall_at_true_percent_class_1 = sum(labels[-true_share_of_1_size:]==1) / sum(labels==1)
        #metrics_dict[f"precision@{true_percent_of_1} (true percent) (class 1)"] = precision_at_true_percent_class_1
        #metrics_dict[f"recall@{true_percent_of_1} (true percent) (class 1)"] = recall_at_true_percent_class_1

        k_percent = 10 # precision/recall@k%
        percent_size = int(k_percent/100 * len(labels))

        precision_at_top_k_class_0 = sum(labels[:percent_size]==0) / percent_size
        recall_at_top_k_class_0 = sum(labels[:percent_size]==0) / sum(labels==0)
        metrics_dict[f"precision@{k_percent} (class 0)"] = precision_at_top_k_class_0
        metrics_dict[f"recall@{k_percent} (class 0)"] = recall_at_top_k_class_0

        precision_at_top_k_class_1 = sum(labels[-percent_size:]==1) / percent_size
        recall_at_top_k_class_1 = sum(labels[-percent_size:]==1) / sum(labels==1)
        metrics_dict[f"precision@{k_percent} (class 1)"] = precision_at_top_k_class_1
        metrics_dict[f"recall@{k_percent} (class 1)"] = recall_at_top_k_class_1



        """
        plt.figure()
        PrecisionRecallDisplay.from_predictions(labels, probs)
        title = f"precision-recall curve, epoch: {epoch}, step: {step}"
        plt.title(title)
        pr_path = os.path.join(out_dir, f"pr_curve_epoch_{epoch}_step_{step}.jpg")
        plt.savefig(pr_path)
        #plt.show(block=False)
        plt.close()


        plt.figure()
        RocCurveDisplay.from_predictions(labels, probs)
        title = f"ROC curve, epoch: {epoch},  step: {step}"
        plt.title(title)
        roc_path = os.path.join(out_dir, f"roc_curve_epoch_{epoch}_step_{step}.jpg")
        plt.savefig(roc_path)
        #plt.show(block=False)
        plt.close()
        """


    return metrics_dict




def run(train_path, dev_path, test_path, lit_ranks, lit_file, init_model,
        rerank_model_path, rerank_checkpoint, longmodel_dir, out_dir,
        do_train, do_test, checkpoint, attention_window, max_pos,
        batch_size, lr, epochs, seed, accumulation_steps, num_top_docs, strategy, enc_strategy,
        use_warmup, warmup_steps, stop_on_roc, dump_test_preds, use_pico, doc_embeds, l2r_top_docs,
        outcome, retrieval_labels, query_proj, query_loss, run_name, num_head=0, section_segment=False,  
        module_type='full', orth_weight=0.1, reduced_dimension=256, delta=0.1, num_experts=8, top_k=2,
        do_dimension_reduction=True, ):

    assert accumulation_steps % batch_size == 0, "accumulation_steps must be a multiple of batch_size"
    
    admission_section_names = [
        "past medical history", "chief complaint", "family history", "physical exam", "allergies", "social history",
        "medications on admission", "present illness"
    ]
    
    selected_sections = admission_section_names

    print(f"Number of sections selected: {len(selected_sections)}")
        


    # 构造 out_dir
    if section_segment == True:
        if module_type == 'orth':
            out_dir = os.path.join(
                out_dir, f"{outcome}/{init_model}/{section_segment}/{module_type}/{orth_weight}/{delta}/{reduced_dimension}/{run_name}"
            )
        else:
            out_dir = os.path.join(
                out_dir, f"{outcome}/{init_model}/{section_segment}/{module_type}/{orth_weight}/{delta}/{top_k}/{num_experts}/{run_name}"
            )
    else:
        if longmodel_dir is not None:
            out_dir = os.path.join(
                out_dir, f"{outcome}/{init_model}/long/{module_type}/{run_name}"
            )
        else:
            out_dir = os.path.join(
                out_dir, f"{outcome}/{init_model}/raw/{module_type}/{run_name}"
            )
    if longmodel_dir is not None and not os.path.exists(longmodel_dir):
        os.makedirs(longmodel_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 创建目录（如果不存在）


    
    seed_torch(seed)
    set_seed(seed)
    setproctitle.setproctitle("python")

    outcome_questions = {'mortality': 'What is the hospital mortality? ',
                         'pmv': 'What is the probability of prolonged mechanical ventilation? ',
                         'los': 'What is the probable length of stay? '}

    dataset = EHRDataset(train_path, dev_path, test_path, do_train, do_test, section_segment,selected_sections=selected_sections)
    
    # print(dataset)
    
    dataset.compute_class_weights()
    
    if lit_ranks is not None:
        dataset.add_relevant_literature(lit_ranks, num_top_docs, lit_file)
        
        missing_lit = 0
        if do_train:
            for doc in list(dataset.train_data.keys()):
                if 'pubmed_docs' not in dataset.train_data[doc]:
                    missing_lit += 1
                    dataset.train_data.pop(doc)
            for doc in list(dataset.dev_data.keys()):
                if 'pubmed_docs' not in dataset.dev_data[doc]:
                    missing_lit += 1
                    dataset.dev_data.pop(doc)
        if do_test:
            for doc in list(dataset.test_data.keys()):
                if 'pubmed_docs' not in dataset.test_data[doc]:
                    missing_lit += 1
                    dataset.test_data.pop(doc)
        print('{} documents do not have PubMed abstracts'.format(missing_lit))
    if doc_embeds:
        dataset.add_literature_matrices(doc_embeds)

    
    num_labels = len(list(dataset.class_weights.keys()))
    print(f"Total Number of labels: {num_labels}")
    if retrieval_labels is not None:
        retrieval_labels = pickle.load(open(retrieval_labels, 'rb'))

    if longmodel_dir is not None:
        create_long_model(
            init_model=init_model,
            save_model_to=longmodel_dir,
            attention_window=attention_window,
            max_pos=max_pos,
            num_labels=num_labels
        )
    if section_segment == True:
        model_path = init_model
    else:
        model_path = longmodel_dir if longmodel_dir is not None else init_model
    config = BertConfig.from_pretrained(model_path,
                                        num_labels=num_labels,
                                        label2id={x: x for x in range(num_labels)},
                                        id2label={x: x for x in range(num_labels)},
                                        cache_dir='../cache')
    tokenizer = BertTokenizerFast.from_pretrained(model_path, cache_dir='../cache') if 'Discharge' not in model_path \
        else AutoTokenizer.from_pretrained(model_path, cache_dir='../cache')
    if section_segment == False:
        model = BertLongForSequenceClassification.from_pretrained(model_path, config=config, cache_dir='../cache') \
            if longmodel_dir is not None \
            else BertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir='../cache')

    rerank_config, rerank_tokenizer, rerank_model = None, None, None
    if rerank_model_path is not None:
        rerank_label_vocab = {'Relevant': 1, 'Irrelevant': 0}
        rerank_config = AutoConfig.from_pretrained(
            rerank_model_path,
            num_labels=len(list(rerank_label_vocab.keys())),
            label2id=rerank_label_vocab,
            id2label={i: l for l, i in rerank_label_vocab.items()},
            cache_dir='../cache',
        )
        rerank_tokenizer = AutoTokenizer.from_pretrained(
            rerank_model_path,
            cache_dir='../cache',
            use_fast=True,
        )
        rerank_model = AutoModel.from_pretrained(
            rerank_model_path,
            from_tf=bool(".ckpt" in rerank_model_path),
            config=rerank_config,
            cache_dir='../cache',
        )
        special_tokens_dict = {'additional_special_tokens': ['[ENTSEP]']}
        num_added_toks = rerank_tokenizer.add_special_tokens(special_tokens_dict)
        rerank_model.resize_token_embeddings(len(rerank_tokenizer))
        if rerank_checkpoint is not None and do_train:  # Only load pretrained reranker if training is to be carried out
            rerank_model.load_state_dict(
                torch.load(rerank_checkpoint))  # Otherwise full model will contain reranker weights too
    if use_pico:
        special_tokens_dict = {'additional_special_tokens': ['<PAR>', '</PAR>', '<INT>', '</INT>', '<OUT>', '</OUT>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        print('Added additional special tokens for PICO highlights')
    print(f"Wether use section segment: {section_segment}")
    if section_segment == True:
        config = BertConfig.from_pretrained(
            init_model,
            num_labels=num_labels,
            label2id={x: x for x in range(num_labels)},
            id2label={x: x for x in range(num_labels)},
            cache_dir='../cache'
        )
        # Step 2: initialize pre-trained model
        model = BertForSequenceClassification.from_pretrained(
            init_model,
            config=config,
            cache_dir='../cache'
        )
        # Step 3: initialize tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(
            init_model,
            model_max_length=512,
            cache_dir='../cache'
        )
        # Step 4: use custom class to extend the base model
        print(f"Create Section-Based Model")
        num_sections = len(selected_sections)

        if module_type == 'orth':
            print(f"Test Orhtogonal Only Performance")
            model = OrthAwareMissingEmbeddingGenerator(config=config, model=model, num_sections=num_sections, 
                                                       reduced_dim=reduced_dimension, delta=delta,
                                                       do_dimension_reduction=do_dimension_reduction)
        elif module_type == 'moe_orth':
            print(f"Test Moe Orthogonal Performance")
            model = MOEOrthAwareMissingEmbeddingGenerator(config=config, model=model, num_sections=num_sections,
                                                          reduced_dim=reduced_dimension, delta=delta,
                                                          do_dimension_reduction=do_dimension_reduction,
                                                          num_experts=num_experts, top_k=top_k)
        # model = ContextAwareMissingEmbeddingGenerator(config=config, model=model,num_sections=len(selected_sections), la_alpha=la_alpha)
    else:
        if lit_ranks is not None and doc_embeds is None: # we're training with literature, and we don't use existing embeddings
            if enc_strategy == 'bienc':
                model = LitAugPredictorBienc(config, model, num_top_docs, strategy, num_head=num_head)
                print("Model is: LitAugPredictorBienc()")
            elif enc_strategy == 'crossenc':
                model = LitAugPredictorCrossenc(config, model, num_top_docs, strategy)
                print("Model is: LitAugPredictorCrossenc()")
        if lit_ranks is not None and doc_embeds is not None:
            if query_proj is None:
                model = L2RLitAugPredictorBienc(config, model, l2r_top_docs, strategy, rerank_model)
            else:
                model = L2RLitAugPredictorBienc(config, model, l2r_top_docs, strategy, rerank_model,
                                                query_proj)
            if query_loss is not None:
                model.query_loss = query_loss
    model = model.cuda()
    print(f"Model: {model}")

    # print('Initialized longformer model with pretrained LM...')

    def preprocess_function(examples, split):
        data_args = (([x[1] for x in examples], None))
        max_length = max_pos if longmodel_dir is not None else 512

        result = tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True,
                           return_tensors='pt')
        result["labels"] = torch.LongTensor([(x[2]) for x in examples])
        result["ehr_id"] = [(x[0]) for x in examples]
        if doc_embeds is not None:
            if rerank_tokenizer is not None:
                data_args = (([outcome_questions[outcome] + x[1] for x in examples], None))
                result["ehr_rerank_tokens"] = rerank_tokenizer(*data_args, padding='max_length', max_length=max_length,
                                                               truncation=True, return_tensors='pt')
            else:
                data_args = (([outcome_questions[outcome] + x[1] for x in examples], None))
                result["ehr_rerank_tokens"] = tokenizer(*data_args, padding='max_length', max_length=max_length,
                                                        truncation=True, return_tensors='pt')
        if lit_ranks is not None and doc_embeds is None:
            result["pubmed_docs"] = []
            result["pubmed_doc_weights"] = []
            k_range = int(num_top_docs) if num_top_docs >= 1 else max([len(x[-1]) for x in examples])
            if k_range > 0:
                if enc_strategy == 'bienc':
                    for k in range(k_range):
                        result["pubmed_doc_weights"].append([x[-1][k][2] if len(x[-1]) > k else 0.0 for x in examples])
                        data_args = (([x[-1][k][1] if len(x[-1]) > k else '' for x in examples], None))
                        result["pubmed_docs"].append(
                            tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True,
                                      return_tensors='pt'))
                if enc_strategy == 'crossenc':
                    for k in range(k_range):
                        result["pubmed_doc_weights"].append([x[-1][k][2] if len(x[-1]) > k else 0.0 for x in examples])
                        data_args = (
                        ([x[1] for x in examples], [x[-1][k][1] if len(x[-1]) > k else '' for x in examples]))
                        result["pubmed_docs"].append(
                            tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True,
                                      return_tensors='pt'))
        if doc_embeds is not None:
            result["pubmed_docs"] = []
            result["pubmed_doc_weights"] = []
            result["pubmed_doc_embeds"] = []
            result["pubmed_doc_ids"] = []
            if retrieval_labels is not None and split == 'train':
                result["pubmed_doc_labels"] = []
            for x in examples:
                result["pubmed_doc_ids"].append([y[0] for y in x[-1]])
                result["pubmed_doc_weights"].append([y[2] for y in x[-1]])
                data_args = (([y[1] for y in x[-1]], None))  # y[0] will be Pubmed ID of doc
                result["pubmed_docs"].append(
                    tokenizer(*data_args, padding='max_length', max_length=max_length, truncation=True,
                              return_tensors='pt'))
                result["pubmed_doc_embeds"].append(np.vstack([x[3][y[0]] for y in x[-1]])[np.newaxis, :, :])
                if retrieval_labels is not None and split == 'train':
                    result["pubmed_doc_labels"].append([retrieval_labels[x[0]][y[0]] for y in x[-1]])
            if retrieval_labels is not None and split == 'train':
                result["pubmed_doc_labels"] = torch.LongTensor(np.vstack(result["pubmed_doc_labels"]))
            result["pubmed_doc_embeds"] = np.vstack(result["pubmed_doc_embeds"])
            result["pubmed_doc_embeds"] = torch.FloatTensor(result["pubmed_doc_embeds"])
        return result

    def batch_and_tokenize_data(examples, batch_size, split):
        
        example_list = []
        for file in list(examples.keys()):
            example = examples[file]
            if lit_ranks is None:
                example_list.append([file, example['ehr'], example['outcome']])
            elif doc_embeds is None:
                example_list.append([file, example['ehr'], example['outcome'], example['pubmed_docs']])
            else:
                example_list.append(
                    [file, example['ehr'], example['outcome'], example['pubmed_doc_embeds'], example['pubmed_docs']])
        batches = []
        # if longmodel_dir is not None and (split == 'dev' or split == 'test'):
        #    batch_size = 1
        #num_tokens_in_ehrs = []
        #num_tokens_in_lits = []
        for i in range(0, len(example_list), batch_size):
            start = i
            end = min(start + batch_size, len(example_list))
            batch = preprocess_function(example_list[start:end], split)
            batches.append(batch)
            #num_tokens_in_ehr = batch["attention_mask"].numpy().sum()
            #num_tokens_in_ehrs.append(num_tokens_in_ehr)
            #num_tokens_in_lit = batch["pubmed_docs"][0]["attention_mask"][0].numpy().sum()
            #num_tokens_in_lits.append(num_tokens_in_lit)
            if len(batches) % 100 == 0:
                print('Created {} batches'.format(len(batches)), end="\r", flush=True)
        return batches

    def section_preprocess_function(batch, tokenizer, max_length=512, section_names=None):
        """
        对一个 batch 的数据进行分词和编码，优化以利用 tokenizer 的批量处理能力。
        
        参数:：
        - batch: list，每个元素是一个 EHR 的信息 [id, sections, label]。
        - tokenizer: 分词器，用于对 section 文本进行编码。
        - max_pos: int，模型的最大输入长度。
        - section_names: list，所有可能的 section 名称。
        
        返回:
        - batch_result: dict，包含分词后的批次数据，包括对缺失部分的处理。
        """
        # 初始化用于保存所有 EHR 分词结果的结构

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

    def section_batch_and_tokenizer_data(examples, tokenizer, batch_size, split, section_names):
        example_list = []
        for file in list(examples.keys()):
            example = examples[file]
            
            example_list.append([file, example['ehr'], example['outcome']])
    
        batches = []

        for i in range(0, len(example_list), batch_size):
            start = i
            end = min(start + batch_size, len(example_list))
            
            # print(len(example_list[start:end]))
            batch = section_preprocess_function(example_list[start:end], tokenizer, section_names=section_names)
            batches.append(batch)

            if len(batches) % 100 == 0:
                print('Created {} batches'.format(len(batches)), end="\r", flush=True)
        return batches

    print('Started batch creation')
    train_batches, dev_batches, test_batches = None, None, None
    
    if do_train:
        if section_segment == False:
            train_batches = batch_and_tokenize_data(dataset.train_data, batch_size, 'train')
            print('Created {} train batches'.format(len(train_batches)))
            dev_batches = batch_and_tokenize_data(dataset.dev_data, batch_size, 'dev')
            print('Created {} dev batches'.format(len(dev_batches)))
        else:
            train_batches = section_batch_and_tokenizer_data(dataset.train_data, tokenizer, batch_size, 'train', section_names=selected_sections)
            print('Created {} Section-based train batches'.format(len(train_batches)))
            # dev_batches = train_batches
            dev_batches = section_batch_and_tokenizer_data(dataset.dev_data, tokenizer, batch_size, 'dev', section_names=selected_sections)
            print('Created {} Section-based dev batches'.format(len(dev_batches)))
    if do_test:
        if section_segment == False:
            test_batches = batch_and_tokenize_data(dataset.test_data, batch_size, 'test')
            print('Created {} test batches'.format(len(test_batches)))
        else:
            test_batches = section_batch_and_tokenizer_data(dataset.test_data, tokenizer, batch_size, 'test', section_names=selected_sections)
            # test_batches = dev_batches
            print('Created {} Section-based test batches'.format(len(test_batches)))
            # print("Pass")
    
    if do_train:
        train(model, train_batches, dev_batches, out_dir, epochs, lr, dataset.class_weights,
            accumulation_steps, strategy, use_warmup, warmup_steps, stop_on_roc, dump_test_preds, section_segment, orth_weight)
    if do_test:
        if checkpoint is not None: # 如果训练中断，加载一个中间段的checkpoint
            if 'checkpoint' in checkpoint:
                full_checkpoint = torch.load(checkpoint)
                model.load_state_dict(full_checkpoint['model_state_dict'])
            else: #如果直接load best_model
                model.load_state_dict(torch.load(checkpoint))
                print('Loaded checkpoint')
        else: # 训练结束后，直接加载best model
            model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
        
        test(model, test_batches, dump_test_preds, out_dir, epoch="end", step="test",
            class_weights=dataset.class_weights, strategy=strategy, section_segment=section_segment, orth_weight=orth_weight)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, action='store', required=True, help='Path to training file')
    parser.add_argument('--dev_path', type=str, action='store', required=True, help='Path to development file')
    parser.add_argument('--test_path', type=str, action='store', required=True, help='Path to test file')
    parser.add_argument('--lit_ranks', type=str, action='store',
                        help='Path to directory containing files of ehr : literature similarity ranks ')
    parser.add_argument('--lit_file', type=str, action='store', help='Path to file containing literature ')
    parser.add_argument('--init_model', type=str, action='store', default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', \
            help='Pretrained model to initialize weights from')
    parser.add_argument('--rerank_model_path', type=str, action='store', help='Pretrained model to initialize reranker weights from')
    parser.add_argument('--rerank_checkpoint', type=str, action='store', help='Checkpoint to load reranker weights from')
    parser.add_argument('--longmodel_dir', type=str, action='store', help='Path to dump longformer version of model')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Provide path to directory to store outputs')
    parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    parser.add_argument('--do_test', action='store_true', default=False, help='Specify if evaluation on test data should be performed')
    parser.add_argument('--checkpoint', type=str, action='store', help='Path to checkpoint to load model weights from')
    parser.add_argument('--attention_window', type=int, action='store', default=512, help='Attention window size')
    parser.add_argument('--max_pos', type=int, action='store', default=4096, help='Maximum position embedding size')
    parser.add_argument('--batch_size', type=int, action='store', default=12, help='Specify batch size')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Specify learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=10, help='Specify number of epochs')
    parser.add_argument('--seed', type=int, action='store', default=0, help='Specify random seed')
    parser.add_argument('--accumulation_steps', type=int, action='store', default=48, help='Specify number of steps for gradient accumulation')
    parser.add_argument('--num_top_docs', type=float, action='store', default=1, help='Number of top ranked abstracts from PubMed to include')
    parser.add_argument('--strategy', type=str, action='store', default='average', help='Strategy to use to combine literature with EHR')
    parser.add_argument('--enc_strategy', type=str, action='store', default='bienc', help='Encoding strategy to use for notes and articles (bienc/crossenc)')
    parser.add_argument('--use_warmup', action='store_true', default=False, help='Choose whether to use LR warmup or not')
    parser.add_argument('--warmup_steps', type=int, action='store', default=5000, help='Choose number of warmup steps')
    parser.add_argument('--stop_on_roc', action='store_true', default=False, help='Use AUROC as early stopping metric')
    parser.add_argument('--dump_test_preds', action='store_true', default=False, help='Dump predictions on test set')
    parser.add_argument('--use_pico', action='store_true', default=False, help='Add PICO highlights to chosen literature docs')
    parser.add_argument('--doc_embeds', type=str, action='store', help='Embeddings of top ranked abstracts for learning to retrieve')
    parser.add_argument('--l2r_top_docs', type=int, action='store', default=5, help='Number of top documents to chosse in learning to retrieve')
    parser.add_argument('--outcome', type=str, action='store', required=True, help='Choose outcome to predict (pmv/los/mortality)')
    parser.add_argument('--retrieval_labels', type=str, action='store', default=None,
                        help='Path to file containing pseudo labels for retrieval training L2R')
    parser.add_argument('--query_proj', type=str, action='store', help='Projection layer to use for queries in L2R')
    parser.add_argument('--query_loss', type=str, action='store', help='Direct loss term for query encoding (pred/reg)')
    parser.add_argument('--run_name', type=str, default="deault run name", action='store', help='name of the run')
    # Context specific augmentation 
    # parser.add_argument('--context_augment', action='store_true', default=False, help='Specify if use the ')
    parser.add_argument('--num_head', type=int, action='store', help='number of head in multi-head attention block')
    parser.add_argument('--section_segment', action='store_true', default=False, help='Decide if segment the EHR by section')
    parser.add_argument('--module_type', type=str, action='store', default='base', help='module type')
    parser.add_argument('--delta', type=float, action='store', default=0.1, help='margin for orthogonal regulation')
    parser.add_argument('--orth_weight', type=float, action='store', default=0.1, help='weight of the orthogonal loss')
    parser.add_argument('--do_dimension_reduction', action='store_true', default=False, help='decide if we use the dimension reduction')
    parser.add_argument('--reduced_dimension', type=int, action='store', default=256, help='reduced dimension')
    parser.add_argument('--num_experts', type=int, action='store', default=0, help='number of experts for MOE module')
    # KNN moe hyperparameter
    parser.add_argument('--top_k', type=int, action='store', default=0, help="Number of experts to be chosen")

    # parser.add_argument('--do_train', action='store_true', default=False, help='Specify if training should be performed')
    args = parser.parse_args()
    args_dict = vars(args)

    # do_long = args.longmodel_dir is not None
    if args.longmodel_dir is not None:
        args.module_type = 'base_long'
        args.delta = None
        args.orth_weight = None
        args.reduced_dimension = None
        args.lambda_residual = None
        args.ssa_softmax_temp = None
        
    elif args.longmodel_dir is None and args.section_segment == False:
        args.module_type = 'base'
        args.delta = None
        args.orth_weight = None
        args.reduced_dimension = None
        args.lambda_residual = None
        args.ssa_softmax_temp = None
        args.pooling_mode = None
        
    if not args.use_warmup:
        args.warmup_steps = None

        
    wandb.init(
            project=f"Structure_Aware_Adm_{args.outcome}",  # 项目名称
            entity="nkw3mr-university-of-virginia",  # 你的 WandB 用户名或团队名
            config={
                "module_type": args.module_type,
                "section_segment": args.section_segment,
                "model": args.init_model,
                "do_long": args.longmodel_dir is not None,
                "margin": args.delta,
                "orth_weight": args.orth_weight,
                'reduced_dimension': args.reduced_dimension,
                "batch_size": args.batch_size,
                "run_name": args.run_name,
                "accumulation_steps": args.accumulation_steps,
                "num_experts": args.num_experts,
                'top_k':args.top_k,
                'do_dimension_reduction':args.do_dimension_reduction,
                'use_warmup': args.use_warmup,
                'warmup_steps': args.warmup_steps,
                'lr':args.lr,
            }
        )  # 直接同步所有超参数


    # 同步 args 到 wandb.config

    print(f"run name: {args_dict['run_name']}")
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    args_dict = vars(args)
    # args_dict.pop("run_name")
    run(**args_dict)
    wandb.finish()
