import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from transformers import BertForSequenceClassification
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

section_names = [
            'discharge diagnosis', 'major surgical or invasive procedure', 'history of present illness',
            'past medical history', 'brief hospital course', 'chief complaint', 'family history',
            'physical exam', 'admission date', 'discharge date', 'service', 'date of birth',
            'sex', 'allergies', 'social history', 'discharge disposition', 'discharge medications',
            'medications on admission', 'attending', 'discharge condition', 'discharge instructions',
            'followup instructions', 'pertinent results'
        ]

class BertWithContrastiveLearning(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = config
    
    def contrastive_forward(self, anchor_enc, positives_enc, negatives_enc):
        anchor_emb = self.bert.bert(
                    input_ids=anchor_enc['input_ids'], 
                    attention_mask=anchor_enc['attention_mask'],
                    token_type_ids=anchor_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        anchor_emb = anchor_emb.last_hidden_state[:, 0, :]
        anchor_emb = anchor_emb.cuda()

        pos_emb = self.bert.bert(
                    input_ids=positives_enc['input_ids'], 
                    attention_mask=positives_enc['attention_mask'],
                    token_type_ids=positives_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        pos_emb = pos_emb.last_hidden_state[:, 0, :]
        pos_emb = pos_emb.cuda()

        neg_emb = self.bert.bert(
                    input_ids=negatives_enc['input_ids'], 
                    attention_mask=negatives_enc['attention_mask'],
                    token_type_ids=negatives_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        neg_emb = neg_emb.last_hidden_state[:, 0, :]
        neg_emb = neg_emb.cuda()

        return anchor_emb, pos_emb, neg_emb


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

class ContextAwareContrastiveEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = config
        self.input_size = self.config.hidden_size
        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.position_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.section_weights = nn.Embedding(num_sections, embedding_dim)
        self.output_size = config.num_labels
        # self.bert_model = bert_model
        # self.bert_config = bert_config
        # self.topk = topk
        # self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.missing_embeddings.weight.device)

        nn.init.xavier_uniform_(self.missing_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)
        nn.init.xavier_uniform_(self.section_weights.weight)
        
    def contrastive_forward(self, anchor_enc, positives_enc, negatives_enc):
        anchor_emb = self.bert.bert(
                    input_ids=anchor_enc['input_ids'], 
                    attention_mask=anchor_enc['attention_mask'],
                    token_type_ids=anchor_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        anchor_emb = anchor_emb.last_hidden_state[:, 0, :]
        anchor_emb = anchor_emb.cuda()

        pos_emb = self.bert.bert(
                    input_ids=positives_enc['input_ids'], 
                    attention_mask=positives_enc['attention_mask'],
                    token_type_ids=positives_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        pos_emb = pos_emb.last_hidden_state[:, 0, :]
        pos_emb = pos_emb.cuda()

        neg_emb = self.bert.bert(
                    input_ids=negatives_enc['input_ids'], 
                    attention_mask=negatives_enc['attention_mask'],
                    token_type_ids=negatives_enc['token_type_ids'],
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None
                )
        neg_emb = neg_emb.last_hidden_state[:, 0, :]
        neg_emb = neg_emb.cuda()

        return anchor_emb, pos_emb, neg_emb

    def forward(self, batch):
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
        # logits = logits.to(device)
        return (None, logits)




class ContextAwareMissingEmbeddingGenerator(nn.Module):
    def __init__(self, model, config, num_sections=23, embedding_dim=768, num_heads=8):
        super().__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = config
        self.input_size = self.config.hidden_size
        self.missing_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.position_embeddings = nn.Embedding(num_sections, embedding_dim)
        self.section_weights = nn.Embedding(num_sections, embedding_dim)
        self.output_size = config.num_labels
        # self.bert_model = bert_model
        # self.bert_config = bert_config
        # self.topk = topk
        # self.strategy = strategy
        self.predictor = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.missing_embeddings.weight.device)

        nn.init.xavier_uniform_(self.missing_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)
        nn.init.xavier_uniform_(self.section_weights.weight)
        

    def forward(self, batch):
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
        # logits = logits.to(device)
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
