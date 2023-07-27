"""
==========================
# -*- coding: utf8 -*-
# @Author   : Miya
# @Time     : 2023/7/27 19:23
# @FileName : cluster.py
# @Email    : Miya.n@foxmail.com
==========================
"""
import os
import json
from tqdm import tqdm

def build_files(data_path, tokenized_data_path,full_tokenizer, min_length, max_length, num_labels): # 用来生成相应的文件，但是固定长度，不切分文件。
    with open(data_path, 'r', encoding='utf8') as f:
        print('prepare data')
        datas = json.load(f)
    labels, lines = list(), list()
    for data in datas:
        label = [0]*num_labels
        label[data["label"]] = 1
        labels.append(label)
        lines.append(data["sentence"])
    lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    lines = [line for line in lines if len(line) > min_length ]
    lines = [line + "0"*(max_length-len(line)-2) if len(line) < max_length-2 else line[:max_length-2] for line in lines]
    lines = [full_tokenizer.tokenize(line) for line in lines]
    lines = [full_tokenizer.convert_tokens_to_ids(line) for line in lines]
    train_data = list()
    line_num = 0
    for line in tqdm(lines):
        line_num += 1
        new_line = list()
        new_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))
        new_line.extend(line)
        new_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))
        train_data.append(new_line)

    with open(tokenized_data_path + 'label.json', 'w') as fi:
        json.dump(labels, fi)
    with open(tokenized_data_path + 'tokenized_data.json', 'w') as f:
        json.dump(train_data, f) # 这里最终的数据是中间加上了空格的，但是我这里是没有的。
    print('finish raw data loading')

