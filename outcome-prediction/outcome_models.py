import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import BertForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
import torch.nn.functional as F
import contrastive_utils
from contrastive_utils import construct_pairs, check_pairs_labels, margin_based_loss, residual_independent_fusion, residual_independent_fusion_average
section_names = [
            "past medical history", "chief complaint", "family history", "physical exam", "allergies", "social history",
        "medications on admission", "present illness"
        ]
class BertLongSelfAttention(LongformerSelfAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        is_index_masked = attention_mask < 0
        is_index_masked = is_index_masked.squeeze(1).squeeze(1)
        attention_mask = attention_mask.squeeze(1).squeeze(1)
        # print('Running self-attention layer #: {}'.format(self.layer_id))
        return super().forward(hidden_states, \
                attention_mask=attention_mask, \
                is_index_masked=is_index_masked) # output_attentions=output_attentions [Arg not present in v4.1.1]

class SectionDimReducer(nn.Module):
    def __init__(self, orig_dim, reduced_dim):
        super().__init__()
        self.fc1 = nn.Linear(orig_dim, orig_dim // 2)  # 降到 384
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(orig_dim // 2, reduced_dim)  # 再降到 256
        self.act2 = nn.GELU()
        self.layer_norm = nn.LayerNorm(reduced_dim)  # 归一化

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.layer_norm(x)

class SimpleGate(nn.Module):
    def __init__(self, input_dim):
        """
        单层门控网络：
          - 使用一个全连接层，
          - Tanh 激活后乘以系数再加上偏置，
          - 输出范围为 [0,1]，初始值接近 0.5。
        """
        super(SimpleGate, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        # 采用零初始化，使得线性层初始输出接近 0
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        """
        x: Tensor, shape [..., input_dim]
        输出：Tensor, shape [...]，范围在 [0,1]
        公式：g = 0.5 + 0.5 * tanh(fc(x))
        当 fc(x) 取 0 时，输出 0.5；当 fc(x) 取正值时，g 接近 1；取负值时接近 0。
        """
        gate = self.fc(x)
        gate = torch.tanh(gate)  # 输出范围 [-1, 1]
        gate = 0.5 + 0.5 * gate  # 映射到 [0,1]
        return gate




class GatedMultiHeadSectionAttentionImputer(nn.Module):
    def __init__(self, embedding_dim, num_heads, d_k, num_sections):
        super(GatedMultiHeadSectionAttentionImputer, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.total_dim = num_heads * d_k
        
        self.query_layer = nn.Linear(embedding_dim, self.total_dim)
        self.key_layer = nn.Linear(embedding_dim, self.total_dim)
        self.value_layer = nn.Linear(embedding_dim, self.total_dim)
        
        # 多头共现偏置参数，形状为 [num_heads, num_sections, d_k]
        self.section_cooc_bias = nn.Parameter(torch.randn(num_heads, num_sections, d_k))
        
        # 为每个头定义一个简化的门控网络
        # 输入特征为拼接后的 query 与 key，即维度为 2*d_k
        self.gate_nets = nn.ModuleList([
            SimpleGate(2 * d_k) for _ in range(num_heads)
        ])
        

    def forward(self, ehr_embeddings, exist_indices, missing_indices):
        """
        ehr_embeddings: Tensor, shape [num_sections, embedding_dim]
        exist_indices: 1D Tensor，存在 section 的索引
        missing_indices: 1D Tensor，缺失 section 的索引
        
        对缺失部分进行多头 attention imputation，门控网络对共现偏置进行动态加权。
        """
        if len(missing_indices) == 0 or len(exist_indices) == 0:
            return ehr_embeddings

        # 提取缺失和存在的部分
        missing_emb = ehr_embeddings[missing_indices]   # [M, embedding_dim]
        exist_emb = ehr_embeddings[exist_indices]         # [N_exist, embedding_dim]
        M = missing_emb.size(0)
        N_exist = exist_emb.size(0)
        
        # 计算多头 Q, K, V
        q_missing = self.query_layer(missing_emb)  # [M, total_dim]
        k_exist = self.key_layer(exist_emb)        # [N_exist, total_dim]
        v_exist = self.value_layer(exist_emb)        # [N_exist, total_dim]
        
        # reshape 并转置，得到 [num_heads, M, d_k] 和 [num_heads, N_exist, d_k]
        q_missing = q_missing.view(M, self.num_heads, self.d_k).transpose(0, 1)
        k_exist = k_exist.view(N_exist, self.num_heads, self.d_k).transpose(0, 1)
        v_exist = v_exist.view(N_exist, self.num_heads, self.d_k).transpose(0, 1)
        
        # 标准注意力得分： [num_heads, M, N_exist]
        scores = torch.matmul(q_missing, k_exist.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 提取多头共现偏置向量：
        # 缺失部分：[num_heads, M, d_k]；存在部分：[num_heads, N_exist, d_k]
        missing_bias = self.section_cooc_bias[:, missing_indices, :]
        exist_bias = self.section_cooc_bias[:, exist_indices, :]
        # 得到 bias_term: [num_heads, M, N_exist]
        bias_term = torch.matmul(missing_bias, exist_bias.transpose(-2, -1))
        
        # 计算门控值，每个头独立计算：
        gated_bias = []
        for h in range(self.num_heads):
            # 对应头的 query 和 key，形状分别为 [M, d_k] 和 [N_exist, d_k]
            q_h = q_missing[h]  # [M, d_k]
            k_h = k_exist[h]    # [N_exist, d_k]
            # 构造拼接特征: 将每个 (missing, exist) 对进行拼接，结果形状 [M, N_exist, 2*d_k]
            q_expanded = q_h.unsqueeze(1).expand(-1, N_exist, -1)
            k_expanded = k_h.unsqueeze(0).expand(M, -1, -1)
            concat_feat = torch.cat([q_expanded, k_expanded], dim=-1)  # [M, N_exist, 2*d_k]
            # 将拼接特征展平为二维 [M*N_exist, 2*d_k] 输入到门控网络中
            concat_feat_flat = concat_feat.view(-1, 2 * self.d_k)
            gate_vals = self.gate_nets[h](concat_feat_flat)  # [M*N_exist, 1]，输出范围 [0,1]
            gate_vals = gate_vals.view(M, N_exist)  # [M, N_exist]
            gated_bias.append(gate_vals)
        gated_bias = torch.stack(gated_bias, dim=0)  # [num_heads, M, N_exist]
        
        # 用门控值调节共现偏置项
        scores = scores + gated_bias * bias_term
        
        # 计算多头注意力输出
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, M, N_exist]
        attn_output = torch.matmul(attn_weights, v_exist)  # [num_heads, M, d_k]
        
        # 合并各头输出：转置并拼接为 [M, total_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(M, self.total_dim)
        
        # 更新缺失部分的 embedding
        updated_embeddings = ehr_embeddings.clone()
        updated_embeddings[missing_indices] = attn_output
        
        return updated_embeddings


class MultiHeadSectionAttentionImputer(nn.Module):
    def __init__(self, embedding_dim, num_heads, d_k, num_sections):
        """
        Args:
            embedding_dim: 输入 section embedding 的维度
            num_heads: 多头注意力中的头数
            d_k: 每个头的维度（通常 d_k = embedding_dim / num_heads）
            num_sections: 每个样本中 section 的数量
        """
        super(MultiHeadSectionAttentionImputer, self).__init__()
        self.num_heads = num_heads
        print(f"Number of heads: {self.num_heads}")
        self.d_k = d_k
        self.total_dim = num_heads * d_k  # 总输出维度

        # 投影到多头空间
        self.query_layer = nn.Linear(embedding_dim, self.total_dim)
        self.key_layer = nn.Linear(embedding_dim, self.total_dim)
        self.value_layer = nn.Linear(embedding_dim, self.total_dim)
        
        # 多头共现偏置参数，形状为 [num_heads, num_sections, d_k]
        self.section_cooc_bias = nn.Parameter(torch.randn(num_heads, num_sections, d_k))
        

    def forward(self, ehr_embeddings, exist_indices, missing_indices):
        """
        对单个样本中缺失的 section 进行多头注意力 imputation。
        
        Args:
            ehr_embeddings: Tensor, shape [num_sections, embedding_dim]
            exist_indices: 1D Tensor，存在 section 的索引
            missing_indices: 1D Tensor，缺失 section 的索引
        Returns:
            更新后的 ehr_embeddings，缺失部分经过多头 attention imputation
            ，shape: [num_sections, embedding_dim]
        """
        # 若不存在缺失或存在部分为空，直接返回原始 embeddings
        if len(missing_indices) == 0 or len(exist_indices) == 0:
            return ehr_embeddings

        # 提取缺失和存在部分的 embedding
        missing_emb = ehr_embeddings[missing_indices]   # [M, embedding_dim]
        exist_emb = ehr_embeddings[exist_indices]         # [N_exist, embedding_dim]
        M = missing_emb.size(0)
        N_exist = exist_emb.size(0)
        
        # 计算多头 Q, K, V
        q_missing = self.query_layer(missing_emb)  # [M, total_dim]
        k_exist = self.key_layer(exist_emb)          # [N_exist, total_dim]
        v_exist = self.value_layer(exist_emb)          # [N_exist, total_dim]
        
        # Reshape为 [M, num_heads, d_k] 和 [N_exist, num_heads, d_k]
        q_missing = q_missing.view(M, self.num_heads, self.d_k)
        k_exist = k_exist.view(N_exist, self.num_heads, self.d_k)
        v_exist = v_exist.view(N_exist, self.num_heads, self.d_k)
        
        # Transpose到 [num_heads, M, d_k] 和 [num_heads, N_exist, d_k]
        q_missing = q_missing.transpose(0, 1)  # [num_heads, M, d_k]
        k_exist = k_exist.transpose(0, 1)      # [num_heads, N_exist, d_k]
        v_exist = v_exist.transpose(0, 1)      # [num_heads, N_exist, d_k]
        
        # 计算多头注意力得分： [num_heads, M, N_exist]
        scores = torch.matmul(q_missing, k_exist.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 提取多头共现偏置向量
        # 缺失部分： [num_heads, M, d_k]
        missing_bias = self.section_cooc_bias[:, missing_indices, :]
        # 存在部分： [num_heads, N_exist, d_k]
        exist_bias = self.section_cooc_bias[:, exist_indices, :]
        # 计算共现偏置内积，得到 [num_heads, M, N_exist]
        bias_term = torch.matmul(missing_bias, exist_bias.transpose(-2, -1))
        
        # 将 bias_term 加入 scores
        scores = scores + bias_term
        
        # 计算多头注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, M, N_exist]
        # 得到各头的输出： [num_heads, M, d_k]
        attn_output = torch.matmul(attn_weights, v_exist)
        
        # 合并多头输出：先转置为 [M, num_heads, d_k]，再 reshape 为 [M, total_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(M, self.total_dim)
        
        # 用更新后的缺失部分替换原始 embedding 中对应部分
        updated_embeddings = ehr_embeddings.clone()
        updated_embeddings[missing_indices] = attn_output
        
        return updated_embeddings

    
class SectionAttentionImputer(nn.Module):
    def __init__(self, embedding_dim, d_k, num_sections):
        """
        Args:
            embedding_dim: 输入 section embedding 的维度
            d_k: 注意力机制中 query/key/value 的维度
            num_sections: 每个样本中 section 的数量
        """
        super(SectionAttentionImputer, self).__init__()
        self.query_layer = nn.Linear(embedding_dim, d_k)
        self.key_layer = nn.Linear(embedding_dim, d_k)
        self.value_layer = nn.Linear(embedding_dim, d_k)
        # 可学习的共现偏置，每个 section 拥有一个偏置向量
        self.section_cooc_bias = nn.Parameter(torch.randn(num_sections, d_k))
        
        nn.init.xavier_uniform_(self.section_cooc_bias)
    
    def forward(self, ehr_embeddings, exist_indices, missing_indices):
        """
        对单个样本中的缺失 section 进行 imputation

        Args:
            ehr_embeddings: Tensor, shape [num_sections, embedding_dim]
            exist_indices: 1D Tensor, 存在 section 的索引
            missing_indices: 1D Tensor, 缺失 section 的索引

        Returns:
            更新后的 ehr_embeddings, 缺失部分已 impute
        """
        if len(missing_indices) == 0 or len(exist_indices) == 0:
            return ehr_embeddings

        # 获取缺失部分和存在部分的 embedding
        missing_emb = ehr_embeddings[missing_indices]        # [M, embedding_dim]
        exist_emb = ehr_embeddings[exist_indices]              # [n_exist, embedding_dim]

        # 计算 Q（缺失部分）、K 和 V（存在部分）
        q_missing = self.query_layer(missing_emb).unsqueeze(1)   # [M, 1, d_k]
        k_exist = self.key_layer(exist_emb).unsqueeze(0)         # [1, n_exist, d_k]
        v_exist = self.value_layer(exist_emb).unsqueeze(0)         # [1, n_exist, d_k]

        d_k = q_missing.size(-1)
        scores = torch.matmul(q_missing, k_exist.transpose(-2, -1)) / math.sqrt(d_k)  # [M, 1, n_exist]

        # 共现偏置计算
        missing_bias_vecs = self.section_cooc_bias[missing_indices].unsqueeze(1)  # [M, 1, d_k]
        present_bias_vecs = self.section_cooc_bias[exist_indices].unsqueeze(0)      # [1, n_exist, d_k]
        bias_term = torch.matmul(missing_bias_vecs, present_bias_vecs.transpose(-2, -1))  # [M, 1, n_exist]

        scores = scores + bias_term
        attn_weights = F.softmax(scores, dim=-1)  # [M, 1, n_exist]
        attn_output = torch.matmul(attn_weights, v_exist)  # [M, 1, d_k]
        attn_output = attn_output.squeeze(1)  # [M, d_k]

        updated_embeddings = ehr_embeddings.clone()
        updated_embeddings[missing_indices] = attn_output

        return updated_embeddings


class SectionOrthAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8, init_weight=0.5, 
                 delta=0.1):
        super().__init__()
        self.bert = model
        self.config = config
        self.input_size = self.config.hidden_size
        self.output_size = config.num_labels
        # print(f"Number of heads: {num_heads}")

        self.predictor = nn.Linear(embedding_dim, self.output_size)

        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()
        self.delta = delta
        self.softmax = nn.Softmax(dim=1)
        self.device = self.missing_embeddings.weight.device

        # 根据是否降维确定 imputer 输入维度
        input_dim_for_imputer =  embedding_dim
        # 这里采用多头 imputer，需要计算每个头的维度（例如：每头维度 = input_dim_for_imputer // num_heads）
        head_dim = input_dim_for_imputer // num_heads
        

        # print(f"Do Multi-head")
        self.imputer = MultiHeadSectionAttentionImputer(embedding_dim=input_dim_for_imputer,
                                                        num_heads=num_heads,
                                                        d_k=head_dim,
                                                        num_sections=num_sections)
        
        self.to(self.device)
        nn.init.xavier_uniform_(self.missing_embeddings.weight)

    def forward(self, batch):
        """
        完整的 forward 方法，首先构造 section embeddings，
        然后利用多头 imputer 模块基于样本内部的共现偏置对缺失的 section 进行 imputation，
        最后聚合所有 section 得到文档级别的表示并预测。
        """
        device = self.missing_embeddings.weight.device
        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)

        # 初始化 section embeddings 和存在标记
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)

        # 构造每个 section 的初始 embedding（使用 BERT 或缺失默认 embedding）
        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']

            has_section = attention_mask.sum(dim=1) > 0

            if has_section.any():
                present_indices = torch.where(has_section)[0].to(device)
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids,
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [n_present, embedding_dim]
                cls_embeddings = cls_embeddings.to(device)

                for idx_in_present, ehr_idx in enumerate(present_indices):
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1

            missing_indices = torch.where(~has_section)[0].to(device)
            if len(missing_indices) > 0:
                # print(f"section_embeddings[missing_indices, idx, :] shape: {section_embeddings[missing_indices, idx, :].shape}")
                # default_emb = self.missing_embeddings(torch.tensor(idx).to(device))
                # print(f"default_emb shape: {default_emb.shape}")
                # section_embeddings[missing_indices, idx, :] = default_emb
                # print(f"section_embeddings[missing_indices, idx, :] shape: {section_embeddings[missing_indices, idx, :].shape}")
                default_emb = self.missing_embeddings(torch.tensor(idx).to(device))  # shape: [768]
                default_emb = default_emb.unsqueeze(0).expand(len(missing_indices), -1)  # shape: [num_missing, 768]
                # print(f"default_emb shape: {default_emb.shape}")
                section_embeddings[missing_indices, idx, :] = default_emb

        # 利用多头 imputer 模块对每个样本缺失部分进行 imputation
        updated_section_embeddings = section_embeddings.clone()
        for ehr_idx in range(batch_size):
            ehr_embeddings = section_embeddings[ehr_idx]  # [num_sections, input_dim_for_imputer]
            ehr_existence_vector = existing_vectors[ehr_idx]
            exist_indices = torch.where(ehr_existence_vector.bool())[0]
            missing_indices = torch.where(~ehr_existence_vector.bool())[0]

            updated_ehr_embeddings = self.imputer(ehr_embeddings, exist_indices, missing_indices)
            updated_section_embeddings[ehr_idx] = updated_ehr_embeddings

        H_doc = section_embeddings.mean(dim=1)  # (batch_size, reduced_dim or embedding_dim)
        logits = self.predictor(H_doc)

        if self.delta > 0.0:
            loss_orth = margin_based_loss(updated_section_embeddings, delta=self.delta)
            return ((loss_orth, None), logits)
        
        return ((0, None), logits)


class FiLMModule(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super(FiLMModule, self).__init__()
        # 从条件向量生成缩放因子和平移因子
        self.fc_gamma = nn.Linear(condition_dim, feature_dim)
        self.fc_beta = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, features, condition):
        # features: [batch_size, feature_dim]
        # condition: [batch_size, condition_dim]
        gamma = self.fc_gamma(condition)  # [batch_size, feature_dim]
        beta = self.fc_beta(condition)    # [batch_size, feature_dim]
        # FiLM 调制
        modulated = gamma * features + beta
        return modulated
    
class CooccurrenceImputationModule(nn.Module):
    def __init__(self, embedding_dim, condition_dim):
        super(CooccurrenceImputationModule, self).__init__()
        # 初步 imputation 模块
        self.initial_imputer = nn.Linear(embedding_dim, embedding_dim)
        # 将共现向量映射到条件向量的 MLP
        self.condition_fc = nn.Sequential(
            nn.Linear(8, condition_dim),  # 假设共现向量长度为 8
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        # FiLM 模块用于条件归一化
        self.film = FiLMModule(condition_dim, embedding_dim)
        # 新增：门控全连接层，用于计算融合权重
        self.gate_fc = nn.Linear(embedding_dim * 2, 1)
        self.layernorm = nn.LayerNorm(embedding_dim)
    
    def forward(self, context_embed, default_embed, cooccur_vector):
        """
        context_embed: [batch_size, embedding_dim]，通过存在 section 聚合得到的上下文 embedding
        default_embed: [batch_size, embedding_dim]，当前缺失 section 的默认 embedding（从 self.missing_embeddings 得到）
        cooccur_vector: [batch_size, 8]，每个 sample 的共现向量（0/1 表示各 section 的存在情况）
        """
        # 生成初步的 imputed embedding 基于上下文
        imputed_initial = self.initial_imputer(context_embed)  # [batch_size, embedding_dim]
        
        # 将共现向量映射为条件向量
        condition = self.condition_fc(cooccur_vector.float())  # [batch_size, condition_dim]
        
        # 使用 FiLM 对初步 imputed embedding 进行调制
        film_output = self.film(imputed_initial, condition)  # [batch_size, embedding_dim]
        
        # 计算 gating 权重
        # 拼接 default_embed 和 film_output
        concat_feat = torch.cat([default_embed, film_output], dim=-1)  # [batch_size, embedding_dim*2]
        gate = torch.sigmoid(self.gate_fc(concat_feat))  # [batch_size, 1]，范围在 0～1
        
        # 使用残差连接与 gating 融合：默认 embedding + gate * (FiLM 输出)
        imputed_final = self.layernorm(default_embed + gate * film_output)  # [batch_size, embedding_dim]
        
        return imputed_final
    
class FiLMOrthAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8, init_weight=0.5, 
                 reduced_dim=256, delta=0.1,do_dimension_reduction=True, condition_dim=128):
        super().__init__()
        self.bert = model
        self.config = config
        self.input_size = self.config.hidden_size
        self.do_dimension_reduction = do_dimension_reduction
        self.output_size = config.num_labels
        if do_dimension_reduction:
            self.attention_layer = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
            self.reducer = SectionDimReducer(orig_dim=embedding_dim, reduced_dim=reduced_dim) 
            self.predictor = nn.Linear(reduced_dim, self.output_size)
        else:
            self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
            self.predictor = nn.Linear(embedding_dim, self.output_size)

        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()
        self.delta = delta
        self.softmax = nn.Softmax(dim=1)
        self.device = self.missing_embeddings.weight.device
        nn.init.xavier_uniform_(self.missing_embeddings.weight)

        # 新的共现信息 imputation 模块
        self.coocc_imputer = CooccurrenceImputationModule(embedding_dim, condition_dim)
        
        self.to(self.device)
        
        
    def forward(self, batch):
        device = self.missing_embeddings.weight.device
        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)
        
        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']
            
            has_section = attention_mask.sum(dim=1) > 0
            if has_section.any():
                present_indices = torch.where(has_section)[0].to(device)
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids, 
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :].to(device)
                
                for idx_in_present, ehr_idx in enumerate(present_indices):
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1
                    
            missing_indices = torch.where(~has_section)[0].to(device)
            if len(missing_indices) > 0:
                # 获取当前 section 的默认 embedding
                default_emb = self.missing_embeddings(torch.tensor(idx, device=device))  # [embedding_dim]
                # 直接赋值给缺失样本（后续会使用 cooccurrence imputation 进一步修正）
                section_embeddings[missing_indices, idx, :] = default_emb

        # 计算每个样本的存在性向量（共现向量），形状 [batch_size, num_sections]
        # 这里 existing_vectors 即为每个样本的共现信息向量
        # 假设 section_embeddings: [batch_size, num_sections, embedding_dim]
        # existing_vectors: [batch_size, num_sections]，1 表示存在，0 表示缺失
        eps = 1e-8
        mask = existing_vectors.unsqueeze(-1)  # [batch_size, num_sections, 1]
        # print(f"Mask dimension: {mask.shape}")
        # print(f"Mask: {mask}")
        context = (section_embeddings * mask).sum(dim=1) / (mask.sum(dim=1) + eps)  # [batch_size, embedding_dim]

        # 构造默认 embedding
        default_all = self.missing_embeddings(torch.arange(num_sections, device=device))  # [num_sections, embedding_dim]
        default_all_exp = default_all.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_sections, embedding_dim]

        # 扩展上下文
        context_exp = context.unsqueeze(1).expand(-1, num_sections, -1)  # [batch_size, num_sections, embedding_dim]

        # 准备条件信息，这里直接用 existing_vectors, 如有需要可用 MLP 转换
        condition = existing_vectors.float()  # [batch_size, 8]
        condition_exp = condition.unsqueeze(1).expand(-1, num_sections, -1)  # [batch_size, num_sections, 8]
        context_flat = context_exp.reshape(-1, self.missing_embeddings.embedding_dim)
        default_flat = default_all_exp.reshape(-1, self.missing_embeddings.embedding_dim)
        condition_flat = condition_exp.reshape(-1, condition.shape[-1])  # [batch_size*num_sections, 8]

        # 调用新的共现信息 imputation 模块（基于 FiLM 的设计）
        imputed_flat = self.coocc_imputer(context_flat, default_flat, condition_flat)  # 输出 [batch_size*num_sections, embedding_dim]
        # print(f"Imputed flat shape: {imputed_flat.shape}")
        # print(f"Imputed flat: {imputed_flat}")
        imputed_all = imputed_flat.reshape(batch_size, num_sections, self.missing_embeddings.embedding_dim)

        # 更新 section_embeddings：仅用 imputed embedding 替换缺失部分
        existing_mask = existing_vectors.unsqueeze(-1)  # [batch_size, num_sections, 1]
        section_embeddings = existing_mask * section_embeddings + (1 - existing_mask) * imputed_all

        if self.do_dimension_reduction:
            reduced_embeddings = self.reducer(section_embeddings)
            attn_input = reduced_embeddings.transpose(0, 1)
            attn_output, _ = self.attention_layer(attn_input, attn_input, attn_input)
            global_embedding = attn_output.mean(dim=0)
            logits = self.predictor(global_embedding)
        else:
            attn_input = section_embeddings.transpose(0, 1)
            attn_output, _ = self.attention_layer(attn_input, attn_input, attn_input)
            global_embedding = attn_output.mean(dim=0)
            logits = self.predictor(global_embedding)
        
        loss_orth = margin_based_loss(attn_output, delta=self.delta)
        # print(f"Orth Loss: {loss_orth}")
        # print(f"Logits: {logits}")
        # exit(0)
        return ((loss_orth, None), logits)





class DeepPositionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=3, dropout=0.1):
        """
        :param embedding_dim: 输入和输出的维度
        :param hidden_dim: 隐藏层的维度
        :param num_layers: 全连接层数
        :param dropout: Dropout 概率
        """
        super(DeepPositionEncoder, self).__init__()
        layers = []
        input_dim = embedding_dim
        for i in range(num_layers):
            # 每层线性变换后接 ReLU 和 Dropout
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            # 后续层的输入维度设为 hidden_dim
            input_dim = hidden_dim
        # 最后一层映射回 embedding_dim
        self.mlp = nn.Sequential(
            *layers,
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layernorm = nn.LayerNorm(embedding_dim)
    
    def forward(self, default_embed):
        """
        :param default_embed: 原始 default embedding, shape [batch_size, embedding_dim]
        :return: 转换后的位置信息表示, shape [batch_size, embedding_dim]
        """
        # MLP 部分，深层非线性变换
        transformed = self.mlp(default_embed)
        # 残差连接：加上原始输入（如果你希望保留部分原始信息的话）
        output = transformed + default_embed
        # LayerNorm 归一化
        output = self.layernorm(output)
        return output


# 假设 FusionGatedExpertModule 如下定义
class FusionGatedExpertModule(nn.Module):
    def __init__(self, embed_dim, hidden_dim,num_layers=3, dropout=0.1):
        super(FusionGatedExpertModule, self).__init__()
        # self.position_encoder = nn.Linear(embed_dim, embed_dim)
        self.deep_position_encoder = DeepPositionEncoder(embed_dim, hidden_dim, num_layers, dropout)
        
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, default_embed, context_embed):
        """
        :param default_embed: 缺失 section 的原始 default embedding, shape [batch_size, embed_dim]
        :param context_embed: 上下文信息, shape [batch_size, embed_dim]
        :return: imputed embedding, 以及门控值
        """
        # 将 default embedding 通过深层非线性变换转换为位置信息
        pos_encoding = self.deep_position_encoder(default_embed)  # [batch_size, embed_dim]
        
        # 拼接上下文信息和位置信息
        fused_input = torch.cat([context_embed, pos_encoding], dim=-1)  # [batch_size, 2*embed_dim]
        
        # 通过融合网络生成候选 imputed embedding
        candidate = self.fusion_mlp(fused_input)  # [batch_size, embed_dim]
        
        # 计算门控值，决定候选和上下文融合的比例
        gate_value = self.gate_mlp(fused_input)   # [batch_size, 1]
        
        # 最终融合：主要依赖上下文信息，同时候选表示也起补充作用
        output = gate_value * candidate + (1 - gate_value) * context_embed
        output = self.layernorm(output)
        return output, gate_value

# MultiExpertImputer 利用多个 FusionGatedExpertModule，并引入专家竞争（Top-K 路由器）
class MultiExpertImputer(nn.Module):
    def __init__(self, num_experts, embed_dim, hidden_dim, top_k=2, num_layers=3, dropout=0.1):
        super(MultiExpertImputer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            FusionGatedExpertModule(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout) for _ in range(num_experts)
        ])
        # 路由器：输入为拼接的 default 与 context
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, default_embed, context_embed):
        """
        :param default_embed: [batch_size, embed_dim]
        :param context_embed: [batch_size, embed_dim]
        :return: imputed embedding [batch_size, embed_dim] 和专家稀疏权重 [batch_size, num_experts]
        """
        batch_size = default_embed.size(0)
        # print(f"Batch size in expert: {batch_size}")
        expert_outputs = []
        for expert in self.experts:
            out, _ = expert(default_embed, context_embed)
            expert_outputs.append(out.unsqueeze(1))  # shape: [batch_size, 1, embed_dim]
        expert_outputs = torch.cat(expert_outputs, dim=1)  # [batch_size, num_experts, embed_dim]
        # print(f"Shape of expert output: {expert_outputs.shape}")
        # 路由器计算专家得分
        router_input = torch.cat([default_embed, context_embed], dim=-1)  # [batch_size, 2*embed_dim]
        # print(f"Shape of router input in expert: {router_input.shape}")
        scores = self.router(router_input)  # [batch_size, num_experts]
        # print(f"Scores: {scores}")
        weights = F.softmax(scores, dim=-1)  # 归一化
        
        # 稀疏激活：仅保留 top_k 专家
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)  # [batch_size, top_k]
        # print(f"Shape of top weights: {topk_weights}")
        batch_indices = torch.arange(batch_size, device=weights.device).unsqueeze(1)  # [batch_size, 1]
        sparse_weights = torch.zeros_like(weights)  # [batch_size, num_experts]
        sparse_weights[batch_indices, topk_indices] = topk_weights        
        sparse_weights_expanded = sparse_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]

        imputed_embed = (expert_outputs * sparse_weights_expanded).sum(dim=1)  # [batch_size, embed_dim]
        return imputed_embed, sparse_weights


class MOEOrthAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8, init_weight=0.5, reduced_dim=256, 
                delta=0.1, num_experts=2, top_k=2, do_dimension_reduction=True,num_layers=3):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = config
        self.input_size = self.config.hidden_size
        self.do_dimension_reduction = do_dimension_reduction
        self.output_size = config.num_labels
        if do_dimension_reduction:
            self.attention_layer = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
            self.reducer = SectionDimReducer(orig_dim=embedding_dim, reduced_dim=reduced_dim) 
            self.predictor = nn.Linear(reduced_dim, self.output_size)
            
        else:
            self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
            self.predictor = nn.Linear(embedding_dim, self.output_size)

        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32))

        self.sigmoid = nn.Sigmoid()  # 先降维
        self.delta = delta
        self.softmax = nn.Softmax(dim=1)
        self.device = self.missing_embeddings.weight.device

        nn.init.xavier_uniform_(self.missing_embeddings.weight)

        self.expert_imputer = MultiExpertImputer(num_experts=num_experts, embed_dim=embedding_dim, hidden_dim=reduced_dim, top_k=top_k,
                                                 num_layers=num_layers)
        self.to(self.device)
        
        
    def forward(self, batch):
        device = self.missing_embeddings.weight.device
        # print(batch.keys())
        # print(f"Batch keys:  {batch.keys()}")

        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        # print(f"Section embeddings shape: {section_embeddings.shape}")
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)
        # print(f"Existing vectos shape: {existing_vectors.shape}")
        # print(f"Existing_vectors shape: {existing_vectors.shape}")
        # print('-'*150)
        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            #print(section_data.keys())
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']
            # print(f"Input Ids shape: {input_ids.shape}")
            # print(f"Attention mask shape: {attention_mask.shape}")
            # print(f"Token type_ids shape: {token_type_ids.shape}")
            
            has_section = attention_mask.sum(dim=1) > 0
            # print(f"has_section: {has_section}")

            if has_section.any():
                # print(f"Present Indices: {present_indices}")
                present_indices = torch.where(has_section)[0].to(device)
                # print(f"Present Indices: {present_indices}")
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids, 
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                # 多线程处理来提速
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = cls_embeddings.to(device)
                
                for idx_in_present, ehr_idx in enumerate(present_indices):
                    # print(f"idx in present: {idx_in_present}")
                    # print(f"ehr_idx: {ehr_idx}")
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1  # 更新 existing_vectors
            # print(f"Existing vectors: {existing_vectors}")
            missing_indices = torch.where(~has_section)[0].to(device)
            # print(f"Missing Indices: {missing_indices}")
            if len(missing_indices) > 0:
                missing_embeddings = self.missing_embeddings(torch.tensor(idx).to(device))
                # print(f"Default embeddings: {missing_embeddings}")
                section_embeddings[missing_indices, idx, :] = missing_embeddings 

        # print(f"Existing vector: {existing_vectors}")
        mask = existing_vectors.to(device).unsqueeze(-1)  # [batch_size, num_sections, 1]
        # print(f"Mask: {mask}")
        # print(f"Shape of Mask: {mask.shape}")
        # 计算加权和并除以存在的 section 数量（避免除 0，加上 eps）
        eps = 1e-8
        context = (section_embeddings * mask).sum(dim=1) / (mask.sum(dim=1) + eps)  # [batch_size, embed_dim]
        # print(f"Context: {context}")
        # print(f"Shape of Context: {context.shape}")
        # 2. 构造所有 section 的默认嵌入
        # 对于每个 section（按照固定顺序），通过 missing_embeddings 得到默认 embedding
        default_all = self.missing_embeddings(torch.arange(num_sections, device=device))  # [num_sections, embed_dim]
        # print(f"Default embedding: {default_all}")
        # print(f"Default embedding shape: {default_all.shape}")
        # 扩展到 batch： [batch_size, num_sections, embed_dim]
        default_all_exp = default_all.unsqueeze(0).expand(batch_size, -1, -1)
        # print(f"Expanded default embedding: {default_all_exp.shape}")
        
        # 3. 对于每个 sample、每个 section，我们希望对缺失位置进行 imputation
        # 我们只更新缺失的位置（existing_mask == 0）
        # 为了向量化处理，将 default_all_exp 和 context 扩展到 [batch_size, num_sections, embed_dim]
        context_exp = context.unsqueeze(1).expand(-1, num_sections, -1)  # [batch_size, num_sections, embed_dim]
        # print(f"Shape of expaned context: {context_exp.shape}")
        # 将两个 tensor 拉平成 [batch_size * num_sections, embed_dim]
        default_flat = default_all_exp.reshape(-1, self.missing_embeddings.embedding_dim)
        # print(f"Shape of default flat: {default_flat.shape}")
        context_flat = context_exp.reshape(-1, self.missing_embeddings.embedding_dim)
        # print(f"Shape of context flat: {context_flat.shape}")
        
        # 4. 利用多专家 imputer 对所有位置进行 imputation（后续只选取缺失位置的结果）
        imputed_flat, _ = self.expert_imputer(default_flat, context_flat)  # [batch_size * num_sections, embed_dim]
        imputed_all = imputed_flat.reshape(batch_size, num_sections, self.missing_embeddings.embedding_dim)
        # print(f"Shape of imputed_all: {imputed_all.shape}")
        
        # 5. 更新 section_embeddings：对于缺失的位置（existing_mask == 0），替换为 imputed_all 的对应值
        # 注意：existing_mask 是 1 表示存在，因此 (1 - existing_mask) 为缺失位置
        existing_mask_float = existing_vectors.unsqueeze(-1).to(device)  # [batch_size, num_sections, 1]
        section_embeddings = existing_mask_float * section_embeddings + (1 - existing_mask_float) * imputed_all
        # print(f"Shape of section embedding: {section_embeddings.shape}")
        
        # 后续：将更新后的 section_embeddings 输入到注意力层和 predictor 中进行下游任务预测
        if self.do_dimension_reduction:
            reduced_embeddings = self.reducer(section_embeddings)  # [batch_size, num_sections, reduced_dim]
            attn_input = reduced_embeddings.transpose(0, 1)  # [num_sections, batch_size, reduced_dim]
            attn_output, _ = self.attention_layer(attn_input, attn_input, attn_input)
            global_embedding = attn_output.mean(dim=0)  # [batch_size, reduced_dim]
            logits = self.predictor(global_embedding)
        else:
            attn_input = section_embeddings.transpose(0, 1)  # [num_sections, batch_size, embed_dim]
            # print(f"attn input shape: {attn_input.shape}")
            attn_output, _ = self.attention_layer(attn_input, attn_input, attn_input)
            global_embedding = attn_output.mean(dim=0)  # [batch_size, embed_dim]
            logits = self.predictor(global_embedding)
        loss_orth = margin_based_loss(attn_output, delta=self.delta)

        return ((loss_orth, None), logits)


        '''exit(0)
        section_embeddings = self.process_missing_sections(section_embeddings, existing_vectors)
        loss_orth = margin_based_loss(section_embeddings, delta=self.delta)
        H_doc = section_embeddings.mean(dim=1)  # (batch_size, reduced_dim)
        logits = self.predictor(H_doc)
        return ((loss_orth, None), logits)'''

    
    
class OrthAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8, init_weight=0.5, reduced_dim=256, 
                delta=0.1, do_dimension_reduction=True):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = config
        self.input_size = self.config.hidden_size
        self.do_dimension_reduction = do_dimension_reduction
        self.output_size = config.num_labels
        if do_dimension_reduction:
            self.attention_layer = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
            self.reducer = SectionDimReducer(orig_dim=embedding_dim, reduced_dim=reduced_dim) 
            self.predictor = nn.Linear(reduced_dim, self.output_size)
            
        else:
            self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
            self.predictor = nn.Linear(embedding_dim, self.output_size)

        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32))

        self.sigmoid = nn.Sigmoid()  # 先降维
        self.delta = delta
        self.softmax = nn.Softmax(dim=1)
        self.device = self.missing_embeddings.weight.device
        self.to(self.device)

        nn.init.xavier_uniform_(self.missing_embeddings.weight)
    def forward(self, batch):
        device = self.missing_embeddings.weight.device
        # print(batch.keys())
        # print(f"Batch keys:  {batch.keys()}")

        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        # print(f"Section embeddings shape: {section_embeddings.shape}")
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)
        # print(f"Existing vectos shape: {existing_vectors.shape}")
        # print(f"Existing_vectors shape: {existing_vectors.shape}")
        # print('-'*150)
        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            #print(section_data.keys())
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']
            # print(f"Input Ids shape: {input_ids.shape}")
            # print(f"Attention mask shape: {attention_mask.shape}")
            # print(f"Token type_ids shape: {token_type_ids.shape}")
            
            has_section = attention_mask.sum(dim=1) > 0
            # print(f"has_section: {has_section}")

            if has_section.any():
                # print(f"Present Indices: {present_indices}")
                present_indices = torch.where(has_section)[0].to(device)
                # print(f"Present Indices: {present_indices}")
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids, 
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                # 多线程处理来提速
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = cls_embeddings.to(device)
                
                for idx_in_present, ehr_idx in enumerate(present_indices):
                    # print(f"idx in present: {idx_in_present}")
                    # print(f"ehr_idx: {ehr_idx}")
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1  # 更新 existing_vectors
            # print(f"Existing vectors: {existing_vectors}")
            missing_indices = torch.where(~has_section)[0].to(device)
            # print(f"Missing Indices: {missing_indices}")
            if len(missing_indices) > 0:
                missing_embeddings = self.missing_embeddings(torch.tensor(idx).to(device))
                # print(f"Default embeddings: {missing_embeddings}")
                section_embeddings[missing_indices, idx, :] = missing_embeddings 
                # print(f"Processed missing embeddings shape: {missing_embeddings.shape}")
            # print(f"Processed missing embeddings shape: {missing_embeddings.shape}")
            # print(f"Processed section embeddings shape: {section_embeddings.shape}")
            # print('-'*150)
    
            # print("-" * 120)
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        # exit(0)
        # print(f"Existing_vectors: {existing_vectors}")
        # print(f"Initial Section Embeddings: {section_embeddings}'")
        # print(f"Initial Section Embeddings shape: {section_embeddings.shape}")
        if self.do_dimension_reduction:
            section_embeddings = self.reducer(section_embeddings)
        for ehr_idx in range(batch_size):
            ehr_embeddings = section_embeddings[ehr_idx]
            # print(f"EHR_Embeddings: {ehr_embeddings}")
            # print(f"EHR_Embeddings shape: {ehr_embeddings.shape}")
            ehr_existence_vector = existing_vectors[ehr_idx]
            # print(f"EHR_existence_vector: {ehr_existence_vector}")
            # print(f"EHR_existence_vector shape: {ehr_existence_vector.shape}")
            exist_indices = ehr_existence_vector.bool()

            if exist_indices.sum() == 0:
                continue
            # print(f"Exist_indices: {exist_indices}")
            exist_section_embeddings = ehr_embeddings[exist_indices]
            # print(f"Exist_section_embeddings: {exist_section_embeddings}")
            
            # print(f"Exist_section_embeddings shape: {exist_section_embeddings.shape}")
            exist_section_embeddings = exist_section_embeddings.unsqueeze(1)  # [20, 1, 768]
            # print(f"Adjusted exist_section_embeddings: {exist_section_embeddings.shape}")

            missing_indices = torch.where(~exist_indices)[0]
            updated_ehr_embeddings = []

            for idx_in_ehr_embeddings, embedding in enumerate(ehr_embeddings):
                if idx_in_ehr_embeddings in missing_indices:
                    # Get the index within missing_indices
                    idx_in_missing = (missing_indices == idx_in_ehr_embeddings).nonzero(as_tuple=True)[0].item()

                    # Prepare missing_embedding
                    missing_embedding = embedding.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, embedding_dim]

                    # Compute attention output
                    attn_output, _ = self.attention_layer(
                        query=missing_embedding,
                        key=exist_section_embeddings,
                        value=exist_section_embeddings
                    )

                    # Update embedding (no in-place operation)
                    updated_embedding = attn_output.squeeze(0).squeeze(0)
                else:
                    # Keep the original embedding
                    updated_embedding = embedding

                # Append to the list
                updated_ehr_embeddings.append(updated_embedding)

            # Stack the updated embeddings to form a tensor
            ehr_embeddings = torch.stack(updated_ehr_embeddings, dim=0)  # Shape: [num_sections, embedding_dim]

            # Assign back to section_embeddings without in-place modification
            section_embeddings = section_embeddings.clone()
            section_embeddings[ehr_idx] = ehr_embeddings
        
        loss_orth = margin_based_loss(section_embeddings, delta=self.delta)
        H_doc = section_embeddings.mean(dim=1)  # (batch_size, reduced_dim)
        logits = self.predictor(H_doc)
        return ((loss_orth, None), logits)
  

class ContextAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8, la_alpha=0.3, init_weight=0.5):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.la_alpha = la_alpha
        self.config = config
        self.input_size = self.config.hidden_size
        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.output_size = config.num_labels
        self.weight = nn.Parameter(torch.tensor(init_weight))
        self.sigmoid = nn.Sigmoid()
        # self.bert_model = bert_model
        # self.bert_config = bert_config
        # self.topk = topk
        # self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.missing_embeddings.weight.device)

        nn.init.xavier_uniform_(self.missing_embeddings.weight)
        

    def forward(self, batch, do_contrastive_loss=False, memory_bank=None, max_negatives=5, mode='train'):
        device = self.missing_embeddings.weight.device
        # print(batch.keys())
        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)
        # print(f"Existing_vectors shape: {existing_vectors.shape}")

        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            #print(section_data.keys())
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']
            
            has_section = attention_mask.sum(dim=1) > 0
            # print(f"has_section: {has_section}")

            if has_section.any():
                # print(f"Present Indices: {present_indices}")
                present_indices = torch.where(has_section)[0].to(device)
                # print(f"Present Indices: {present_indices}")
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids, 
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                # 多线程处理来提速
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = cls_embeddings.to(device)
                
                for idx_in_present, ehr_idx in enumerate(present_indices):
                    # print(f"idx in present: {idx_in_present}")
                    # print(f"ehr_idx: {ehr_idx}")
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1  # 更新 existing_vectors

            missing_indices = torch.where(~has_section)[0].to(device)
            # print(f"Missing Indices: {missing_indices}")
            if len(missing_indices) > 0:
                missing_embeddings = self.missing_embeddings(torch.tensor(idx).to(device))
                section_embeddings[missing_indices, idx, :] = missing_embeddings 
            # print("-" * 120)
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        # exit(0)
        # print(f"Existing_vectors: {existing_vectors}")
        # print(f"Initial Section Embeddings: {section_embeddings}'")
        # print(f"Initial Section Embeddings shape: {section_embeddings.shape}")

        for ehr_idx in range(batch_size):
            ehr_embeddings = section_embeddings[ehr_idx]
            # print(f"EHR_Embeddings: {ehr_embeddings}")
            # print(f"EHR_Embeddings shape: {ehr_embeddings.shape}")
            ehr_existence_vector = existing_vectors[ehr_idx]
            # print(f"EHR_existence_vector: {ehr_existence_vector}")
            # print(f"EHR_existence_vector shape: {ehr_existence_vector.shape}")
            exist_indices = ehr_existence_vector.bool()

            if exist_indices.sum() == 0:
                continue
            # print(f"Exist_indices: {exist_indices}")
            exist_section_embeddings = ehr_embeddings[exist_indices]
            # print(f"Exist_section_embeddings: {exist_section_embeddings}")
            
            # print(f"Exist_section_embeddings shape: {exist_section_embeddings.shape}")
            exist_section_embeddings = exist_section_embeddings.unsqueeze(1)  # [20, 1, 768]
            # print(f"Adjusted exist_section_embeddings: {exist_section_embeddings.shape}")

            missing_indices = torch.where(~exist_indices)[0]
            updated_ehr_embeddings = []

            for idx_in_ehr_embeddings, embedding in enumerate(ehr_embeddings):
                if idx_in_ehr_embeddings in missing_indices:
                    # Get the index within missing_indices
                    idx_in_missing = (missing_indices == idx_in_ehr_embeddings).nonzero(as_tuple=True)[0].item()

                    # Prepare missing_embedding
                    missing_embedding = embedding.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, embedding_dim]

                    # Compute attention output
                    attn_output, _ = self.attention_layer(
                        query=missing_embedding,
                        key=exist_section_embeddings,
                        value=exist_section_embeddings
                    )

                    # Update embedding (no in-place operation)
                    updated_embedding = attn_output.squeeze(0).squeeze(0)
                else:
                    # Keep the original embedding
                    updated_embedding = embedding

                # Append to the list
                updated_ehr_embeddings.append(updated_embedding)

            # Stack the updated embeddings to form a tensor
            ehr_embeddings = torch.stack(updated_ehr_embeddings, dim=0)  # Shape: [num_sections, embedding_dim]

            # Assign back to section_embeddings without in-place modification
            section_embeddings = section_embeddings.clone()
            section_embeddings[ehr_idx] = ehr_embeddings
        
        section_embeddings = section_embeddings.to(device)
        document_embeddings = torch.mean(section_embeddings, dim=1)
        logits = self.predictor(document_embeddings)
        if do_contrastive_loss == True and memory_bank is not None:
            if mode == 'train':
                memory_bank.update(document_embeddings, batch['labels'])

            if memory_bank.has_sufficient_samples(min_samples_per_class=4):
                # anchors, positive_samples, negative_samples, anchor_labels, pos_labels, neg_labels = construct_pairs(document_embeddings, batch['labels'], memory_bank)
                anchors, positives, negatives, anchor_labels, pos_labels, neg_labels_groups = construct_pairs(document_embeddings, batch['labels'], memory_bank)

                # print(f"Pass the check of the genereate: {pass_check}")
                return ((anchors, positives, negatives), logits)
            # check_pairs_labels(anchors,positive_samples,negative_samples)
            

        # logits = logits.to(device)
        return (None, logits)

    def _build_adjacency_matrix(self, section_embeddings, threshold=0.5, k=None):
        """
        动态生成邻接矩阵，支持稀疏化、对称化和孤立节点处理
        :param section_embeddings: [batch_size, num_sections, embedding_dim]
        :param threshold: 相似度阈值，用于稀疏化
        :param k: top-k 相似邻居（可选）
        :return: 邻接矩阵 A [batch_size, num_sections, num_sections]
        """
        # Step 1: 嵌入归一化
        normalized_embeddings = F.normalize(section_embeddings, dim=-1)

        # Step 2: 计算相似度矩阵
        similarity_matrix = torch.einsum('bnd,bmd->bnm', normalized_embeddings, normalized_embeddings)

        # Step 3: 稀疏化处理
        if k is not None:
            # 使用 top-k 稀疏化
            topk_values, _ = torch.topk(similarity_matrix, k=k, dim=-1)
            threshold = topk_values[:, :, -1].unsqueeze(-1)  # 获取每行的第 k 大值作为阈值
            adjacency_matrix = (similarity_matrix >= threshold).float() * similarity_matrix
        else:
            # 使用固定阈值稀疏化
            adjacency_matrix = (similarity_matrix >= threshold).float() * similarity_matrix

        # Step 4: 对称化邻接矩阵
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose(-1, -2)) / 2.0

        # Step 5: 修正对角线（动态调整自连接权重）
        degree_matrix = adjacency_matrix.sum(dim=-1, keepdim=True)  # 计算节点度
        self_loop_weight = 1.0 / degree_matrix.clamp(min=1e-6)  # 动态调整自连接权重
        self_loops = torch.eye(adjacency_matrix.size(-1), device=adjacency_matrix.device).unsqueeze(0) * self_loop_weight
        adjacency_matrix += self_loops  # 添加自连接

        # Step 6: 孤立节点处理（确保没有节点完全孤立）
        isolation_mask = (adjacency_matrix.sum(dim=-1) == 0)  # 检查是否有孤立节点
        if isolation_mask.any():
            adjacency_matrix[isolation_mask] += torch.eye(adjacency_matrix.size(-1), device=adjacency_matrix.device)

        # Step 7: 对邻接矩阵进行归一化（对称归一化）
        degree_matrix = adjacency_matrix.sum(dim=-1, keepdim=True)  # 再次计算节点度
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0.0  # 防止除零
        adjacency_matrix = degree_matrix_inv_sqrt * adjacency_matrix * degree_matrix_inv_sqrt  # 对称归一化

        return adjacency_matrix


    def _laplacian_smoothing(self, section_embeddings, adjacency_matrix):
        """
        对嵌入应用图拉普拉斯平滑（向量化版本）
        :param section_embeddings: [batch_size, num_sections, embedding_dim]
        :param adjacency_matrix: [batch_size, num_sections, num_sections]
        :return: 平滑后的 section_embeddings
        """
        batch_size, num_sections, embedding_dim = section_embeddings.shape

        # Step 1: 计算度矩阵 D
        degree_matrix = adjacency_matrix.sum(dim=-1)  # [batch_size, num_sections]
        degree_matrix_inv = torch.diag_embed(1.0 / degree_matrix)  # [batch_size, num_sections, num_sections]

        # Step 2: 计算拉普拉斯矩阵 L
        laplacian_matrix = torch.eye(num_sections, device=adjacency_matrix.device).unsqueeze(0) \
                        - torch.matmul(degree_matrix_inv, adjacency_matrix)  # [batch_size, num_sections, num_sections]

        # Step 3: 应用拉普拉斯平滑
        identity_matrix = torch.eye(num_sections, device=adjacency_matrix.device).unsqueeze(0)  # [1, num_sections, num_sections]
        smoothing_matrix = identity_matrix - self.la_alpha * laplacian_matrix  # [batch_size, num_sections, num_sections]

        smoothed_embeddings = torch.matmul(smoothing_matrix, section_embeddings)  # [batch_size, num_sections, embedding_dim]
        return smoothed_embeddings
    def _weighted_document_embedding(self, section_embeddings):
        context_vector = torch.mean(section_embeddings, dim=1, keepdim=True)  # 全局上下文
        attention_scores = torch.bmm(section_embeddings, context_vector.transpose(1, 2)).squeeze(-1)  # [batch_size, num_sections]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_sections]
        weighted_embedding = torch.sum(attention_weights.unsqueeze(-1) * section_embeddings, dim=1)  # 加权平均
        return weighted_embedding
    
    def laplacian_forward(self, batch, do_contrastive_loss, memory_bank=None,max_negatives=4, mode='train'):
        device = self.missing_embeddings.weight.device
        # print(batch.keys())
        batch_size = len(batch['ehr_id'])
        num_sections = len(batch['sections'].keys())
        section_names = list(batch['sections'].keys())
        batch['labels'] = batch['labels'].to(device)
        section_embeddings = torch.zeros((batch_size, num_sections, self.missing_embeddings.embedding_dim), device=device)
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        existing_vectors = torch.zeros((batch_size, num_sections), dtype=torch.float, device=device)
        # print(f"Existing_vectors shape: {existing_vectors.shape}")

        for idx, section_name in enumerate(section_names):
            section_data = batch['sections'][section_name]
            #print(section_data.keys())
            input_ids = section_data['input_ids']
            attention_mask = section_data['attention_mask']
            token_type_ids = section_data['token_type_ids']
            
            has_section = attention_mask.sum(dim=1) > 0
            # print(f"has_section: {has_section}")

            if has_section.any():
                # print(f"Present Indices: {present_indices}")
                present_indices = torch.where(has_section)[0].to(device)
                # print(f"Present Indices: {present_indices}")
                present_input_ids = input_ids[present_indices].to(device)
                present_attention_mask = attention_mask[present_indices].to(device)
                present_token_type_ids = token_type_ids[present_indices].to(device)

                outputs = self.bert.bert(
                    input_ids=present_input_ids, 
                    attention_mask=present_attention_mask,
                    token_type_ids=present_token_type_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
                # 多线程处理来提速
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                cls_embeddings = cls_embeddings.to(device)
                
                for idx_in_present, ehr_idx in enumerate(present_indices):
                    # print(f"idx in present: {idx_in_present}")
                    # print(f"ehr_idx: {ehr_idx}")
                    section_embeddings[ehr_idx, idx, :] = cls_embeddings[idx_in_present]
                    existing_vectors[ehr_idx, idx] = 1  # 更新 existing_vectors

            missing_indices = torch.where(~has_section)[0].to(device)
            # print(f"Missing Indices: {missing_indices}")
            if len(missing_indices) > 0:
                missing_embeddings = self.missing_embeddings(torch.tensor(idx).to(device))
                section_embeddings[missing_indices, idx, :] = missing_embeddings 
            # print("-" * 120)
        # print(f"Section Embedding shape: {section_embeddings.shape}")
        # exit(0)
        # print(f"Existing_vectors: {existing_vectors}")
        # print(f"Initial Section Embeddings: {section_embeddings}'")
        # print(f"Initial Section Embeddings shape: {section_embeddings.shape}")

        for ehr_idx in range(batch_size):
            ehr_embeddings = section_embeddings[ehr_idx]
            # print(f"EHR_Embeddings: {ehr_embeddings}")
            # print(f"EHR_Embeddings shape: {ehr_embeddings.shape}")
            ehr_existence_vector = existing_vectors[ehr_idx]
            # print(f"EHR_existence_vector: {ehr_existence_vector}")
            # print(f"EHR_existence_vector shape: {ehr_existence_vector.shape}")
            exist_indices = ehr_existence_vector.bool()

            if exist_indices.sum() == 0:
                continue
            # print(f"Exist_indices: {exist_indices}")
            exist_section_embeddings = ehr_embeddings[exist_indices]
            # print(f"Exist_section_embeddings: {exist_section_embeddings}")
            
            # print(f"Exist_section_embeddings shape: {exist_section_embeddings.shape}")
            exist_section_embeddings = exist_section_embeddings.unsqueeze(1)  # [20, 1, 768]
            # print(f"Adjusted exist_section_embeddings: {exist_section_embeddings.shape}")

            missing_indices = torch.where(~exist_indices)[0]
            updated_ehr_embeddings = []

            for idx_in_ehr_embeddings, embedding in enumerate(ehr_embeddings):
                if idx_in_ehr_embeddings in missing_indices:
                    # Get the index within missing_indices
                    idx_in_missing = (missing_indices == idx_in_ehr_embeddings).nonzero(as_tuple=True)[0].item()

                    # Prepare missing_embedding
                    missing_embedding = embedding.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, embedding_dim]

                    # Compute attention output
                    attn_output, _ = self.attention_layer(
                        query=missing_embedding,
                        key=exist_section_embeddings,
                        value=exist_section_embeddings
                    )

                    # Update embedding (no in-place operation)
                    updated_embedding = attn_output.squeeze(0).squeeze(0)
                else:
                    # Keep the original embedding
                    updated_embedding = embedding

                # Append to the list
                updated_ehr_embeddings.append(updated_embedding)

            # Stack the updated embeddings to form a tensor
            ehr_embeddings = torch.stack(updated_ehr_embeddings, dim=0)  # Shape: [num_sections, embedding_dim]

            # Assign back to section_embeddings without in-place modification
            section_embeddings = section_embeddings.clone()
            section_embeddings[ehr_idx] = ehr_embeddings
        
        section_embeddings = section_embeddings.to(device)
        # print(f"Raw Section Embedding shape: {section_embeddings.shape}")
        # document_embeddings = torch.mean(section_embeddings, dim=1)

        # Step 2: 动态生成邻接矩阵
        adjacency_matrix = self._build_adjacency_matrix(section_embeddings, k=5)
        '''print(f"Raw Section Embeddings: {section_embeddings}")
        print(f"Raw Section shape: {section_embeddings.shape}")
        print(f"Adjacency matrix: {adjacency_matrix}")
        # Step 3: 图拉普拉斯平滑（向量化版本）'''
        document_embeddings = self._laplacian_smoothing(section_embeddings, adjacency_matrix)
        '''print(f"Smoothed Section Embeddings: {section_embeddings}")
        print(f"Smoothed Section Shape: {section_embeddings.shape}")
        exit(0)'''
        # Step 4: 生成最终文档嵌入并进行分类
        document_embeddings = torch.mean(section_embeddings, dim=1)  # 平均池化
        # document_embeddings = self._weighted_document_embedding(section_embeddings)
        logits = self.predictor(document_embeddings)
        if do_contrastive_loss == True and memory_bank is not None and mode == 'train':
            # print(f"Do add contrastive loss")
            # print(f"Document Embedding shape: {document_embeddings.shape}")
            memory_bank.update(document_embeddings, batch['labels'])

            if memory_bank.has_sufficient_samples(min_samples_per_class=4):
                anchors, positive_samples, negative_samples, __, __, __ = construct_pairs(document_embeddings, batch['labels'], 
                                                                                          memory_bank,n_negatives=max_negatives)
            
            # print(f"Pass the check of the genereate: {pass_check}")
                '''print(f"Anchor shape: {anchors.shape}")
                print(f"Positive samples shape: {positive_samples.shape}")
                print(f"Negative samples shape: {negative_samples.shape}")'''
                return ((anchors, positive_samples, negative_samples), logits)
            # check_pairs_labels(anchors,positive_samples,negative_samples)
            
        # logits = self.predictor(document_embeddings)
        return (None, logits)

class BertLongForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


class LitAugPredictorCrossenc(nn.Module):

    def __init__(self, bert_config, bert_model, topk, strategy='average'):
        super().__init__()
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None
    ):
        note_lit_reps = []
        if 'vote' in self.strategy:
            prob_matrices = []
            for doc_batch in pubmed_docs:
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_logits = self.bert_model(**doc_batch)[0]
                cur_logits_softmax = self.softmax(cur_logits)
                prob_matrices.append(cur_logits_softmax)
            averaged_probs = None
            if self.strategy == 'softvote':
                averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
            if self.strategy == 'weightvote':
                if len(prob_matrices) == 1:
                    averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
                else:
                    weighted_matrices = []
                    total_weight = torch.zeros(prob_matrices[0].size()).cuda()
                    for prob_matrix, weights in zip(prob_matrices, pubmed_doc_weights):
                        weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.num_labels)
                        weighted_matrices.append(weights * prob_matrix)
                        total_weight += weights
                    weighted_matrices = [x/total_weight for x in weighted_matrices]
                    averaged_probs = torch.sum(torch.stack(weighted_matrices), dim=0)
            averaged_log_probs = torch.log(averaged_probs)
            return (None, averaged_log_probs)
        if self.strategy == 'average':
            rep_list = []
            for doc_batch in pubmed_docs:
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_outputs = self.bert_model.bert(**doc_batch)[1]   # 0 - last state, 1 - pooled output
                rep_list.append(cur_outputs)
            final_lit_rep = torch.mean(torch.stack(rep_list), dim=0)
            logits = self.bert_model.classifier(final_lit_rep)
            return (None, logits)
        if self.strategy == 'weightaverage':
            rep_list = []
            total_weight = torch.zeros((input_ids.size()[0], self.bert_config.hidden_size)).cuda()
            for doc_batch, weights in zip(pubmed_docs, pubmed_doc_weights):
                doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
                cur_outputs = self.bert_model.bert(**doc_batch)[1]
                weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.hidden_size)
                rep_list.append(weights * cur_outputs)
                total_weight += weights
            rep_list = [x/total_weight for x in rep_list]
            averaged_reps = torch.sum(torch.stack(rep_list), dim=0)
            logits = self.bert_model.classifier(averaged_reps)
            return (None, logits)


class LitAugPredictorBienc(nn.Module):

    def __init__(self, bert_config, bert_model, topk, strategy='average', context_augment=False, num_head=0):
        super().__init__()
        self.input_size = 2 * bert_config.hidden_size  # embeddings of note + literature
        self.output_size = bert_config.num_labels
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        # self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.context_augment = context_augment
        if self.strategy == 'context':
            self.num_head = num_head
            self.multihead_attention = nn.MultiheadAttention(embed_dim=bert_config.hidden_size, num_heads=self.num_head)
            self.note_norm_layer = nn.LayerNorm(bert_config.hidden_size)
            self.lit_norm_layer = nn.LayerNorm(bert_config.hidden_size)
            self.add_norm1 = nn.LayerNorm(bert_config.hidden_size)
            self.feed_forward = nn.Sequential(
                nn.Linear(bert_config.hidden_size, bert_config.hidden_size * 4),
                nn.ReLU(),
                nn.Linear(bert_config.hidden_size * 4, bert_config.hidden_size)
            )
            self.add_norm2 = nn.LayerNorm(bert_config.hidden_size)
            self.gate = nn.Linear(bert_config.hidden_size * 2, bert_config.hidden_size)
            self.predictor = nn.Linear(bert_config.hidden_size, self.output_size)
        else:
            self.predictor = nn.Linear(self.input_size, self.output_size)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None,
        split='train'
    ):
        '''print(f"Do context augmentation? {context_augment}")
        print(f"Number of head used: {num_head}")'''
        note_outputs = self.bert_model.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        note_reps = note_outputs[1]
        lit_reps = []
        if len(pubmed_docs) >= 50:
            pubmed_docs = pubmed_docs[:50]
        # print(len(pubmed_docs))
        if len(pubmed_docs) == 0:
            lit_reps.append(torch.zeros(note_reps.size()).cuda())
        for doc_batch in pubmed_docs:
            doc_batch = {x:y.cuda() for x,y in doc_batch.items()}
            cur_outputs = self.bert_model.bert(**doc_batch)
            lit_reps.append(cur_outputs[1])
        # for here we finish the process of the literatures and the ehr note
        # the shape of the lit_reps is [batch_size, embedding_dimension_of_bert] * num_top_docs
        # the shape of the note_reps is [batch_size, embedding_dimension_of_bert]
        '''print(f"note_reps: {note_reps}")
        print(f"Type of note_reps: {type(note_reps)}")
        print(f"lit_reps: {lit_reps}")
        print(f"Length of the lit_reps: {len(lit_reps)}") 
        print(f"type of lit_reps: {type(lit_reps[0])}")

        print(f"Shape of the note_reps: {note_reps.shape}")
        print(f"Shape of the lit_reps: {lit_reps[0].shape}")'''
        
        if self.strategy == 'context':
            final_lit_rep = torch.mean(torch.stack(lit_reps), dim=0)
            '''print(f"Shape of the final note rep: {note_reps.shape}")
            # print(f"Type of the final lit rep: {final_lit_rep}")
            print(f"Shape of the final lit rep: {final_lit_rep.shape}")'''
            
            head_dim = final_lit_rep.shape[1] // self.num_head
            norm_lit_rep = self.lit_norm_layer(final_lit_rep)
            norm_note_rep = self.note_norm_layer(note_reps)

            
            queries = norm_note_rep
            keys, values = norm_lit_rep, norm_lit_rep

            queries, keys, values = queries.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0)

        

            '''queries = queries.view(queries.size(0), 1, num_head, head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
            keys = keys.view(keys.size(0), 1, num_head, head_dim).transpose(1, 2)          # [batch_size, num_heads, 1, head_dim]
            values = keys'''

            attn_output, attn_output_weights = self.multihead_attention(queries, keys, values)
        
            attn_output = attn_output.squeeze(0)

            
            combined_input = torch.cat([attn_output, norm_note_rep], dim=1)
            

            
            gate = torch.sigmoid(self.gate(combined_input))
            # print(f"Gate parameter: {gate}")
            gated_output = gate * attn_output + (1-gate) * norm_note_rep
            add_norm_output_1 = self.add_norm1(gated_output + norm_note_rep)

            ff_output = self.feed_forward(add_norm_output_1)

            final_output = self.add_norm2(ff_output + add_norm_output_1)
            # print(f"Final output shape: {final_output.shape}")
            logits = self.predictor(final_output)
            return (None, logits)


        if self.strategy == 'average':
            final_lit_rep = torch.mean(torch.stack(lit_reps), dim=0)
            final_rep = torch.cat([note_reps, final_lit_rep], dim=1)
            # print(f"average final rep shape: {final_rep.shape}")
            
            logits = self.predictor(final_rep)
            return (None, logits)
        if self.strategy == 'weightaverage':
            total_lit_rep = torch.zeros(lit_reps[0].size()).cuda()
            total_weight = torch.zeros((input_ids.size()[0], self.bert_config.hidden_size)).cuda()
            for cur_lit_rep, weights in zip(lit_reps, pubmed_doc_weights):
                weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.bert_config.hidden_size)
                total_weight += weights
                total_lit_rep += (weights * cur_lit_rep)
            if torch.sum(total_weight).item() != 0.0:
                total_lit_rep /= total_weight
            final_rep = torch.cat([note_reps, total_lit_rep], dim=1)
            logits = self.predictor(final_rep)
            return (None, logits)
        if self.strategy == 'softvote' or self.strategy == 'weightvote':
            prob_matrices = []
            for cur_lit_rep in lit_reps:
                cur_final_rep = torch.cat([note_reps, cur_lit_rep], dim=1)
                cur_logits = self.predictor(cur_final_rep)
                cur_logits_softmax = self.softmax(cur_logits)
                prob_matrices.append(cur_logits_softmax)
            averaged_probs = None
            if self.strategy == 'softvote':
                averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
            if self.strategy == 'weightvote':
                if len(prob_matrices) == 1:
                    averaged_probs = torch.mean(torch.stack(prob_matrices), dim=0)
                else:
                    weighted_matrices = []
                    total_weight = torch.zeros(prob_matrices[0].size()).cuda()
                    for prob_matrix, weights in zip(prob_matrices, pubmed_doc_weights):
                        weights = torch.cuda.FloatTensor(weights).unsqueeze(1).repeat(1, self.output_size)
                        weighted_matrices.append(weights * prob_matrix)
                        total_weight += weights
                    weighted_matrices = [x/total_weight for x in weighted_matrices if torch.sum(total_weight).item() != 0.0]
                    averaged_probs = torch.sum(torch.stack(weighted_matrices), dim=0)
            averaged_log_probs = torch.log(averaged_probs)
            return (None, averaged_log_probs)


class L2RLitAugPredictorBienc(nn.Module):

    def __init__(self, bert_config, bert_model, tokenizer, topk, strategy='average', rerank_model=None, query_proj=None):
        super().__init__()
        self.input_size = 2 * bert_config.hidden_size  # embeddings of note + literature
        self.output_size = bert_config.num_labels
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.topk = topk
        self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity(dim=2)
        if rerank_model is not None:
            self.rerank_model = rerank_model
        if query_proj is not None:
            self.query_proj = query_proj
            if query_proj == 'linear':
                self.query_proj_layer = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
            if query_proj == 'transformer':
                encoder_layer = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size, nhead=8)
                self.query_proj_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pubmed_docs=None,
        pubmed_doc_weights=None,
        pubmed_doc_embeds=None,
        ehr_rerank_tokens=None,
        pubmed_doc_ids=None,
        pubmed_doc_labels=None,
        split='train'
    ):
        note_question_outputs, note_question_hidden_states = None, None
        retrieval_loss = 0.0
        if hasattr(self, 'rerank_model'):
            note_question_outputs = self.rerank_model(**ehr_rerank_tokens)
            note_question_outputs = note_question_outputs['last_hidden_state'][:,0,:]
        else:
            note_question_outputs = self.bert_model.bert(**ehr_rerank_tokens)
            note_question_hidden_states = note_question_outputs[0]
            note_question_outputs = note_question_outputs[1]
        if hasattr(self, 'query_proj_layer'):
            if self.query_proj == 'linear':
                note_question_outputs = self.query_proj_layer(note_question_outputs)
            if self.query_proj == 'transformer':
                note_question_hidden_states = note_question_hidden_states.permute(1,0,2)
                note_question_outputs = self.query_proj_layer(note_question_hidden_states)
                note_question_outputs = torch.mean(note_question_outputs.permute(1,0,2), dim=1)
        if hasattr(self, 'query_loss'):
            if self.query_loss == 'pred':
                empty_lit_reps = torch.zeros(note_question_outputs.size()).cuda()
                note_question_lit_reps = torch.cat([note_question_outputs, empty_lit_reps], dim=1)
                note_question_probs = self.predictor(note_question_lit_reps)
                retrieval_loss = nn.CrossEntropyLoss()(note_question_probs, labels)
        note_question_reps = note_question_outputs.unsqueeze(1)
        note_question_rep_repeat = note_question_reps.repeat(1,pubmed_doc_embeds.size()[1],1)
        note_lit_sim = self.cosine(note_question_rep_repeat, pubmed_doc_embeds)
        # note_lit_sim = torch.nan_to_num(note_lit_sim, nan=-1.1)
        # note_lit_sim = torch.inner(note_question_rep_repeat, pubmed_doc_embeds)
        # note_lit_sim = -1 * torch.cdist(note_question_reps, pubmed_doc_embeds)
        # note_lit_sim = note_lit_sim.squeeze(1)
        corrected_note_lit_sim = torch.FloatTensor(np.nan_to_num(note_lit_sim.detach().cpu().numpy(), nan=-1.1)).cuda()
        top_doc_scores, top_doc_inds = torch.topk(corrected_note_lit_sim, self.topk, dim=1)  # Should break graph here
        if pubmed_doc_labels is not None:
            max_sim_array = torch.max(note_lit_sim.detach(), dim=1)[0].unsqueeze(-1)
            max_sim_array = max_sim_array.repeat(1,note_lit_sim.size()[-1])
            note_lit_softmax = self.softmax(note_lit_sim - max_sim_array)
            retrieval_loss -= torch.log(torch.sum(note_lit_softmax * pubmed_doc_labels))
        # Recompute note reps (without question) using outcome prediction LM
        note_outputs = self.bert_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        note_outputs = note_outputs[1]
        if hasattr(self, 'query_loss'):
            if self.query_loss == 'reg':
                retrieval_loss += nn.MSELoss()(note_question_outputs, note_outputs)
        note_reps = note_outputs.unsqueeze(1)
        if split == 'test' and torch.sum(torch.isnan(note_outputs)) > 0:
            note_reps = torch.FloatTensor(np.nan_to_num(note_reps.detach().cpu().numpy(), nan=0)).cuda()
            print('Note rep contains NaNs!!!')
        output_array = []
        for i in range(top_doc_inds.size()[0]):
            cur_doc_inds = top_doc_inds[i,:].detach().cpu().numpy().tolist()
            cur_args = (([pubmed_docs[i][0][0][x] for x in cur_doc_inds], None))
            cur_doc_input = self.tokenizer(*cur_args, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            cur_doc_input = {k:v.cuda() for k,v in cur_doc_input.items()}
            # print(cur_doc_input)
            # print(cur_doc_inds)
            # cur_doc_input = {k:torch.index_select(v.cuda(), 0, cur_doc_inds) for k,v in pubmed_docs[i].items()}
            cur_outputs = self.bert_model.bert(**cur_doc_input)[1]
            if split == 'test' and torch.sum(torch.isnan(cur_outputs)) > 0:
                cur_outputs = torch.FloatTensor(np.nan_to_num(cur_outputs.detach().cpu().numpy(), nan=0)).cuda()
            if self.strategy == 'average':
                final_lit_rep = torch.mean(cur_outputs, dim=0).unsqueeze(0)
                final_rep = torch.cat([note_reps[i,:,:], final_lit_rep], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                output_array.append(logits - max_val)
            if self.strategy == 'weightaverage':
                weights = top_doc_scores[i,:].unsqueeze(1).detach()
                total_weight = torch.sum(weights).item()
                final_lit_rep = []
                if split == 'test' and torch.sum(torch.isnan(cur_outputs)) > 0:
                    print('Lit rep contains NaNs!!!!')
                if math.isnan(total_weight):
                    final_lit_rep = torch.mean(cur_outputs, dim=0).unsqueeze(0)
                else:
                    final_lit_rep = torch.sum((cur_outputs * weights)/total_weight, dim=0).unsqueeze(0)
                final_rep = torch.cat([note_reps[i,:,:], final_lit_rep], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                output_array.append(logits - max_val)
            if 'vote' in self.strategy:
                cur_note_rep = note_reps[i,:,:].repeat(self.topk,1)
                final_rep = torch.cat([cur_note_rep, cur_outputs], dim=1)
                logits = self.predictor(final_rep)
                max_val = max(logits.detach().cpu().numpy().tolist()[0])
                logits_softmax = self.softmax(logits - max_val)
                if self.strategy == 'softvote':
                    output_array.append(torch.mean(logits_softmax, dim=0))
                if self.strategy == 'weightvote':
                    weights = top_doc_scores[i,:].unsqueeze(1).detach()
                    total_weight = torch.sum(weights).item()
                    if math.isnan(total_weight):
                        output_array.append(torch.mean(logits_softmax, dim=0))
                    else:
                        output_array.append(torch.sum((logits_softmax * weights)/total_weight, dim=0))
        final_output = torch.stack(output_array).squeeze(1)
        if 'vote' in self.strategy:
            final_output = torch.log(final_output)
        return (retrieval_loss, final_output)
