import csv
import os
import pickle
csv.field_size_limit(2147483647)
from collections import Counter
import json
import random
from collections import defaultdict
from tqdm import tqdm

class EHRDataset:

    def __init__(self, train_path, dev_path, test_path, do_train=True, do_test=True, section_segment=False, task_type='mp', selected_sections=None):
        assert do_train or do_test, "if no train and no test, which data should it loads?"
        if section_segment:
            self.train_data = self.read_json(train_path)
            self.dev_data = self.read_json(dev_path)
            self.test_data = self.read_json(test_path)

        else:
            self.train_data = self.read_csv(train_path)
            self.dev_data = self.read_csv(dev_path)
            self.test_data = self.read_csv(test_path)
        self.selected_sections = selected_sections
        print(f"Numebr of selected section in datadataset: {len(self.selected_sections)}")
        '''self.important_sections = ['discharge diagnosis','major surgical or invasive procedure','history of present illness',
                                    'past medical history','brief hospital course','chief complaint','physical exam',
                                    'discharge medications','discharge disposition','medications on admission',
                                    'discharge instructions','followup instructions']
        self.full_sections = [
            'discharge diagnosis', 'major surgical or invasive procedure', 'history of present illness',
            'past medical history', 'brief hospital course', 'chief complaint', 'family history',
            'physical exam', 'admission date', 'discharge date', 'service', 'date of birth',
            'sex', 'allergies', 'social history', 'discharge disposition', 'discharge medications',
            'medications on admission', 'attending', 'discharge condition', 'discharge instructions',
            'followup instructions', 'pertinent results'
        ]'''
        # self.contrastive_data = self.add_contrastive_data(task_type='multi-class')
        self.do_train = do_train
        self.do_test = do_test

    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        data = {}

        # 遍历 JSON 数据中的每个对象
        for entry in json_data:
            entry_id = str(entry['hadm_id'])
            ehr_content = entry['sections']
            outcome_value = int(entry['labels'])

            # 将每个 entry 存入字典，键为 entry_id，值为包含 ehr 和 outcome 的字典
            data[entry_id] = {
                'ehr': ehr_content,
                'outcome': outcome_value
            }

        # 返回处理后的数据字典
        return data

    def read_csv(self, path):
        reader = csv.reader(open(path))
        data = {}
        next(reader, None)
        for row in reader:
            data[row[0]] = {'ehr': row[1], 'outcome': int(row[2])}
        len_data = len(data)
        ten_percent = int(0.1*len_data) 
        #data = {x[0]: x[1] for x in list(data.items())[:ten_percent]} for debug purposes (less data)
        return data

    def compute_class_weights(self):
        class_counts = Counter()
        data = self.train_data
        for example in data:
            class_counts.update([data[example]['outcome']])
        num_samples = sum(list(class_counts.values()))
        num_classes = len(class_counts)
        balance_coeff = float(num_samples)/num_classes
        self.class_weights = {k:balance_coeff/float(v) for k,v in class_counts.items()}

    def add_relevant_literature(self, lit_dir, topk, lit_file):
        all_texts = pickle.load(open(lit_file, 'rb'))

        rankings_files = os.listdir(lit_dir)

        num_rankings_files = len(rankings_files)
        missing_lit_counter = 0

        ehrs_to_process = set()
        if self.do_train:
            ehrs_to_process = ehrs_to_process | set(self.train_data.keys()) | set(self.dev_data.keys())
        if self.do_test:
            ehrs_to_process = ehrs_to_process | set(self.test_data.keys())

        all_ehrs = set(self.train_data.keys()) | set(self.dev_data.keys()) | set(self.test_data.keys())
        

        for i, file in enumerate(rankings_files):
            id = file.split('.pkl')[0]
            if id not in all_ehrs:
                print(f"id {id} not in train/dev/test datasets")
            if id not in ehrs_to_process:
                continue

            docs = pickle.load(open(os.path.join(lit_dir, file), 'rb'))
            if isinstance(docs, dict) and len(docs.keys()) == 1:
                docs = docs[id]
            docs = list(reversed(sorted(docs, key=lambda x:x[1])))
            docs_nums = [x[0] for x in docs]
            not_found_docs = set(docs_nums) - set(all_texts)
            num_not_found_docs = len(not_found_docs)
            if num_not_found_docs > 0:
                print(f"not found: {num_not_found_docs}")

            chosen_docs = docs[:int(topk)] if topk >= 1 else [x for x in docs if x[1] >= topk]
            try:
                chosen_docs = [[x[0], all_texts[x[0]]['text'], x[1]] for x in chosen_docs]  # We may want to include year later?
            except:
                missing_lit_counter += 1
            if id in self.train_data:
                self.train_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.dev_data:
                self.dev_data[id]['pubmed_docs'] = chosen_docs
            elif id in self.test_data:
                self.test_data[id]['pubmed_docs'] = chosen_docs

            print(f"added docs to {i + 1}/{len(rankings_files)} ehr files", end="\r", flush=True)


    def add_literature_matrices(self, lit_embed_file):
        lit_embeds = pickle.load(open(lit_embed_file, 'rb'))

        if self.do_train:
            for id in self.train_data:
                self.train_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.train_data[id]['pubmed_docs']}
            for id in self.dev_data:
                self.dev_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.dev_data[id]['pubmed_docs']}

        if self.do_test:
            for id in self.test_data:
                self.test_data[id]['pubmed_doc_embeds'] = {x[0]:lit_embeds[x[0]] for x in self.test_data[id]['pubmed_docs']}


    
    
    def preprocess_ehr_data_with_sampling(self, data, max_samples_per_outcome=1000):
        """
        预处理 EHR 数据并对每个 section/outcome 采样，限制数据总量。
        Args:
            data: 原始 EHR 数据，格式为 {ehr_id: {ehr, outcome}}。
            important_sections: 重要的 section 列表。
            max_samples_per_outcome: 每个 section 和 outcome 的最大采样数量。
        Returns:
            processed_data: 预处理后的数据，格式为 {section_name: {outcome: list of samples}}。
        """
        processed_data = defaultdict(lambda: defaultdict(list))
        # 遍历原始数据并筛选重要 section
        for ehr_id, ehr in data.items():
            outcome = ehr['outcome']
            filtered_ehr = {k: v for k, v in ehr['ehr'].items() if k in self.selected_sections and v.strip()}
            for section_name, content in filtered_ehr.items():
                processed_data[section_name][outcome].append({
                    'ehr_id': ehr_id,
                    'content': content
                })

        # 针对每个 section/outcome 进行采样，限制样本总量
        for section_name, outcome_dict in processed_data.items():
            for outcome, samples in outcome_dict.items():
                if len(samples) > max_samples_per_outcome:
                    processed_data[section_name][outcome] = random.sample(samples, max_samples_per_outcome)

        return processed_data

    
    '''def sample_contrastive_data(processed_data, section_name, anchor_outcome, k_max=3, max_negatives=5):
        """
        根据预处理数据采样对比学习样本。

        Args:
            processed_data: 预处理后的数据字典。
            section_name: 当前 section 的名称。
            anchor_outcome: Anchor 的分类标签。
            k_max: 最大正样本数量。
            max_negatives: 最大负样本数量。

        Returns:
            positives, negatives: 采样的正负样本列表。
        """
        # 获取正样本
        positive_candidates = processed_data[section_name][anchor_outcome]
        k = min(len(positive_candidates), k_max)
        positives = random.sample(positive_candidates, k) if k > 0 else []

        # 获取负样本
        negative_candidates = []
        for other_outcome, samples in processed_data[section_name].items():
            if other_outcome != anchor_outcome:
                negative_candidates.extend(samples)

        num_negatives = min(len(negative_candidates), max_negatives)
        negatives = random.sample(negative_candidates, num_negatives) if num_negatives > 0 else []

        return positives, negatives'''

    
    def add_contrastive_data_optimized(self, preprocessed_data, max_negatives=10):
        """
        构建对比学习样本，每个anchor只采样一个positive。
        Args:
            preprocessed_data: 预处理后的 EHR 数据，格式为 {section_name: {outcome: list of samples}}。
            max_negatives: 每个 anchor 的最大负样本数量。
        Returns:
            final_data: 包含 anchor, positive, negatives 的对比学习样本。
        """
        final_data = []
        # 预计算负样本集合
        negative_samples_by_label = defaultdict(lambda: defaultdict(list))
        for section_name, outcome_dict in preprocessed_data.items():
            for outcome, samples in outcome_dict.items():
                negative_samples_by_label[outcome][section_name].extend(samples)

        for section_name, outcome_dict in tqdm(preprocessed_data.items(), desc="Sampling Contrastive Pair"):
            for anchor_outcome, samples in outcome_dict.items():
                for anchor in samples:
                    anchor_content = anchor['content']
                    anchor_id = anchor['ehr_id']
                    
                    # 构建正样本候选集
                    positive_candidates = [
                        sample for sample in outcome_dict[anchor_outcome]
                        if sample['ehr_id'] != anchor_id
                    ]
                    
                    # 如果没有正样本候选，跳过当前anchor
                    if not positive_candidates:
                        continue
                    
                    # 只采样一个正样本
                    positive_sample = random.choice(positive_candidates)
                    
                    # 构建负样本集合
                    negative_candidates_different_label = negative_samples_by_label[1 - anchor_outcome][section_name]
                    negative_candidates_same_label = []
                    for other_section, other_outcome_dict in preprocessed_data.items():
                        if other_section != section_name:
                            negative_candidates_same_label.extend(other_outcome_dict[anchor_outcome])
                    
                    # 合并负样本
                    all_negative_candidates = negative_candidates_different_label + negative_candidates_same_label
                    
                    # 检查是否有足够的负样本
                    if not all_negative_candidates:
                        continue
                    if len(all_negative_candidates) < max_negatives:
                        continue
                    
                    selected_negatives = random.sample(all_negative_candidates, max_negatives)
                    
                    # 构建最终数据
                    final_data.append({
                        'anchor': anchor_content,
                        'positive': positive_sample['content'],  # 现在直接存储单个positive的content
                        'negatives': [neg['content'] for neg in selected_negatives]
                    })
        
        return final_data
    
    
    def add_contrastive_data_multiclass_diff_section_negatives_with_min_section(
        self, preprocessed_data, max_negatives=10, min_negatives_per_section=2, 
        treat_same_label_diff_section_as_positive=False
    ):
        """
        在多分类场景下，为每个 anchor 样本创建 (anchor, positive, negatives)，
        negatives 包含「不同 label」的样本，且确保每个 section 至少包含一定数量的负样本。
        
        Args:
            preprocessed_data: 类似 {section_name: {outcome_label: [sample, ...], ...}, ...}
            max_negatives: 每个 anchor 采样多少个负样本。
            min_negatives_per_section: 每个 section 至少包含的负样本数量。
            treat_same_label_diff_section_as_positive: 是否将「相同 label、不同 section」也视为正样本。
        
        Returns:
            final_data: list，每个元素包含 { 'anchor': ..., 'positive': ..., 'negatives': [...] }
        """
        final_data = []
        all_outcomes = set()
        for sec_name, outcome_dict in preprocessed_data.items():
            for label in outcome_dict.keys():
                all_outcomes.add(label)
        all_outcomes = list(all_outcomes)

        samples_by_label_section = defaultdict(lambda: defaultdict(list))
        for sec_name, outcome_dict in preprocessed_data.items():
            for lbl, samples in outcome_dict.items():
                samples_by_label_section[lbl][sec_name].extend(samples)

        for section_name, outcome_dict in tqdm(preprocessed_data.items(), desc="Sampling Contrastive Pair"):
            for anchor_label, anchor_samples in outcome_dict.items():
                for anchor in anchor_samples:
                    anchor_id = anchor['ehr_id']
                    anchor_content = anchor['content']

                    # -----------------------------
                    #    A) 正样本 (positive)
                    # -----------------------------
                    if treat_same_label_diff_section_as_positive:
                        positive_candidates = []
                        for sec_nm_in_all, samples_list in samples_by_label_section[anchor_label].items():
                            for s in samples_list:
                                if s['ehr_id'] != anchor_id:
                                    positive_candidates.append(s)
                    else:
                        same_section_samples = outcome_dict[anchor_label]
                        positive_candidates = [
                            s for s in same_section_samples if s['ehr_id'] != anchor_id
                        ]

                    if not positive_candidates:
                        continue

                    positive_sample = random.choice(positive_candidates)

                    # -----------------------------
                    #    B) 负样本 (negatives)
                    # -----------------------------
                    negative_candidates_by_section = defaultdict(list)
                    for other_label in all_outcomes:
                        if other_label == anchor_label:
                            continue
                        for sec_nm_in_all, samples_list in samples_by_label_section[other_label].items():
                            if sec_nm_in_all != section_name:  # 不同 section
                                negative_candidates_by_section[sec_nm_in_all].extend(samples_list)

                    # 确保每个 section 至少采样 min_negatives_per_section
                    selected_negatives = []
                    for sec_nm, candidates in negative_candidates_by_section.items():
                        if len(candidates) >= min_negatives_per_section:
                            selected_negatives.extend(
                                random.sample(candidates, min_negatives_per_section)
                            )
                        else:
                            selected_negatives.extend(candidates)

                    # 如果仍不足 max_negatives，随机补充
                    remaining_candidates = [
                        sample for sec_nm, candidates in negative_candidates_by_section.items()
                        for sample in candidates if sample not in selected_negatives
                    ]
                    if len(selected_negatives) < max_negatives:
                        additional_negatives = random.sample(
                            remaining_candidates,
                            min(max_negatives - len(selected_negatives), len(remaining_candidates))
                        )
                        selected_negatives.extend(additional_negatives)

                    if len(selected_negatives) < max_negatives:
                        continue  # 若总负样本仍不足，则跳过当前 anchor

                    # -----------------------------
                    #    C) 组装最终数据
                    # -----------------------------
                    final_data.append({
                        'anchor': anchor_content,
                        'positive': positive_sample['content'],
                        'negatives': [neg['content'] for neg in selected_negatives]
                    })

        return final_data
    

    def add_contrastive_data_multiclass_efficient(
        self, preprocessed_data, max_negatives=10, min_negatives_per_section=2, max_positives=2, max_candidates_per_section=100
    ):
        """
        优化版本的对比学习样本生成，确保负样本与 anchor 的 label 不同，完全适配 preprocess 数据格式。
        
        Args:
            preprocessed_data: 类似 {section_name: {outcome_label: [sample, ...], ...}, ...}
            max_negatives: 每个 anchor 最多采样多少个负样本。
            min_negatives_per_section: 每个 section 至少包含的负样本数量。
            max_positives: 每个 anchor 最多采样多少个正样本。
            max_candidates_per_section: 每个 section 的负样本池最大大小。
        
        Returns:
            final_data: list，每个元素包含 { 'anchor': ..., 'positive': ..., 'negatives': [...] }
        """
        final_data = []

        # 1. 限制每个 section 的候选样本池大小
        limited_negative_pool = defaultdict(lambda: defaultdict(list))
        for section_name, outcome_dict in preprocessed_data.items():
            for outcome, samples in outcome_dict.items():
                if len(samples) > max_candidates_per_section:
                    limited_negative_pool[section_name][outcome].extend(random.sample(samples, max_candidates_per_section))
                else:
                    limited_negative_pool[section_name][outcome].extend(samples)

        # 2. 构造对比学习样本
        for section_name, outcome_dict in tqdm(preprocessed_data.items(), desc="Sampling Contrastive Pair"):
            for anchor_label, anchor_samples in outcome_dict.items():
                for anchor in anchor_samples:
                    anchor_id = anchor['ehr_id']
                    anchor_content = anchor['content']

                    # A) 正样本 (positive)
                    positive_candidates = [
                        s for s in outcome_dict[anchor_label] if s['ehr_id'] != anchor_id
                    ]
                    if not positive_candidates:
                        continue

                    # 限制正样本的数量
                    positive_samples = random.sample(positive_candidates, min(len(positive_candidates), max_positives))

                    # B) 负样本 (negatives)
                    for positive_sample in positive_samples:
                        selected_negatives = []
                        remaining_negatives = max_negatives  # 当前需要的负样本数

                        # 每个 section 至少取 min_negatives_per_section 个
                        for sec_nm, outcome_dict_by_section in limited_negative_pool.items():
                            if sec_nm == section_name or remaining_negatives <= 0:
                                continue
                            for neg_label, candidates in outcome_dict_by_section.items():
                                if neg_label == anchor_label:
                                    continue  # 过滤相同 label
                                
                                # 按需采样
                                num_to_sample = min(len(candidates), min_negatives_per_section, remaining_negatives)
                                selected_negatives.extend(random.sample(candidates, num_to_sample))
                                remaining_negatives -= num_to_sample

                        # 如果仍不足 max_negatives，从全局池补充
                        if remaining_negatives > 0:
                            remaining_candidates = [
                                sample for sec_nm, outcome_dict_by_section in limited_negative_pool.items()
                                for neg_label, candidates in outcome_dict_by_section.items()
                                for sample in candidates if neg_label != anchor_label and sample not in selected_negatives
                            ]
                            additional_negatives = random.sample(
                                remaining_candidates,
                                min(remaining_negatives, len(remaining_candidates))
                            )
                            selected_negatives.extend(additional_negatives)

                        # 确保负样本数量不超过 max_negatives
                        selected_negatives = selected_negatives[:max_negatives]

                        if len(selected_negatives) < max_negatives:
                            continue  # 若负样本不足则跳过

                        # C) 添加样本
                        final_data.append({
                            'anchor': anchor_content,
                            'positive': positive_sample['content'],
                            'negatives': [neg['content'] for neg in selected_negatives]
                        })

        return final_data
