"""
==========================
# -*- coding: utf8 -*-
# @Author   : Miya
# @Time     : 2023/7/25 14:08
# @FileName : cluster.py
# @Email    : Miya.n@foxmail.com
==========================
"""

import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder
import torch.nn as nn

# 文件里面是一个字典，keys = {'label', 'sentence'}
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
    lines = [line.replace('\n', ' [SEP] ') for line in lines]
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
        json.dump(train_data, f)
    print('finish raw data loading')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', default=True, action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--min_length', default=0, type=int, required=False, help='最短句子长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")
    parser.add_argument('--n_embd', default=13317, type=int, help="全连接层输入长度") # 词表长度
    parser.add_argument('--num_labels', default=2, type=int, help="分类标签数量")
    parser.add_argument('--multi_gpu', default=False, type=bool, help="gpu数量")
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    n_embd = args.n_embd
    num_labels = args.num_labels
    multi_gpu = args.multi_gpu
    assert log_step % gradient_accumulation == 0

    MLP_layer = nn.Linear(n_embd, num_labels, device=device, bias=False)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path,
                    full_tokenizer=full_tokenizer, min_length=min_length, max_length=model_config.n_embd, num_labels = num_labels)
        print('files built')

    # 数据加载
    with open(tokenized_data_path + 'tokenized_data.json', 'r') as f:
        samples = json.load(f)
    with open(tokenized_data_path + 'label.json', 'r') as fi:
        labels = json.load(fi)

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    # 计算总的步数
    total_steps = len(samples) * epochs / batch_size / gradient_accumulation
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                  t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')

    # 开始训练
    overall_step = 0
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        samples = [[int(token) for token in line] for line in samples]

        for step in range(len(samples) // batch_size):
            print("开始训练")
            batch = samples[step * batch_size: (step + 1) * batch_size]
            batch_labels = labels[step * batch_size: (step + 1) * batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            #  前向传播，GPT2+MLP，MLP的输入是GPT输出的最后一个token([CLS])的logits,输出是一个二分类。
            gpt_outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = gpt_outputs[:2]
            hidden_states = [data[-1] for data in logits.detach().numpy()]
            hidden_states = torch.tensor(hidden_states, dtype = torch.float).to(device)
            batch_outputs = MLP_layer(hidden_states)
            batch_labels = torch.tensor(batch_labels).float().to(device)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(batch_outputs, batch_labels)

            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (overall_step + 1) % log_step == 0:
                tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1

if __name__ == '__main__':
    main()
