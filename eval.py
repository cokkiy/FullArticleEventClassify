# eval the model on the test set
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from torch.utils.data import DataLoader
from BertBiLSTMCRF import BertBiLSTMCRF
from EventExtratorClassifer import EventExtractorClassifer
from PRF1Calculator import PRF1Calculator
from event_dataset import EventDataset
from arguments_dataset import ArgumentsDataset

import argparse
import os
parser = argparse.ArgumentParser(
    description='Eval the trained model on the test set')
parser.add_argument('model_name', type=str,
                    help='The file name of the trained model')
parser.add_argument('--valid_file', type=str, default='./data/FNDEE_valid.json',
                    help='The file name of the valid dateset file')
parser.add_argument('--batch_size', type=int,
                    help='Batch size, default 4', default=4)
parser.add_argument('--bert_model', type=str, default='../models/xlm-roberta-base',
                    help='Pretrained bert model name (default: "../models/xlm-roberta-base")')

args = parser.parse_args()

num_event_types = 9  # 事件类型数量
num_arguments = 12  # 论元分类数量

model_name = args.model_name
json_file = args.valid_file
batch_size = args.batch_size
bert_name = args.bert_model

# device specific
device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Bert分词器
tokenizer = AutoTokenizer.from_pretrained(bert_name)
config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
bert_model = AutoModelForMaskedLM.from_pretrained(
    bert_name, config=config).to(device)

# 训练数据
eval_args_dataset = ArgumentsDataset(json_file, tokenizer)  # 论元数据集
eval_event_dataset = EventDataset(json_file, tokenizer)  # 事件数据集

eval_event_dataloader = DataLoader(
    eval_event_dataset, batch_size=batch_size, collate_fn=eval_event_dataset.pad_collate_fn, shuffle=False)
eval_args_dataloader = DataLoader(
    eval_args_dataset, batch_size=batch_size, collate_fn=eval_args_dataset.pad_collate_fn, shuffle=False)

# Extract the file name
file_name = os.path.basename(model_name)
# path
path = os.path.dirname(model_name)

# model file name
event_model_file = f'{path}/{file_name}-event.pt'
args_model_file = f'{path}/{file_name}-args.pt'

event_model_data = torch.load(event_model_file, map_location=device)
event_model = EventExtractorClassifer(bert_model, num_event_types).to(device)
event_model.load_state_dict(event_model_data)

# args_model_data = torch.load(args_model_file, map_location=device)
# args_model = BertBiLSTMCRF(bert_model, num_arguments)
# args_model.load_state_dict(args_model_data)

# evaluation loop
event_model.eval()
# args_model.eval()

total_loss = 0
total_correct = 0
total_event_correct = 0
total_args_correct = 0
total_samples = 0

with torch.no_grad():
    loss = nn.CrossEntropyLoss()
    prf1 = PRF1Calculator()
    # 评估事件抽取与分类模型
    for i, batch in enumerate(eval_event_dataloader):
        input_ids, label_ids, attention_mask, text, aligned_labels = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        attention_mask = attention_mask.to(device)
        event_logits, = event_model(input_ids, attention_mask,)
        event_loss = loss(
            event_logits.view(-1, num_event_types), label_ids.view(-1))
        total_loss += event_loss.item()
        event_preds = torch.argmax(event_logits, dim=1)
        total_event_correct += torch.sum(event_preds ==
                                         label_ids.view(-1)).item()
        total_samples += len(label_ids.view(-1))
        prf1.update(event_preds, label_ids.view(-1))
        if (i+1) % 10 == 0:
            print(
                f'{i+1}/{len(eval_event_dataset)} Event Extraction Loss: {total_loss/total_samples}, Accuracy: {total_event_correct/total_samples}')
            print(
                f'     Predict triggers:{prf1.predict} Correct Triggers: {prf1.correct}, Real triggers: {prf1.real}')
            print(
                f'     Trigger Precision: {prf1.P}, Recall: {round(prf1.R, 4)}, F1: {round(prf1.F1, 4)}')
    print(
        f'Event Extraction Loss: {total_loss/total_samples}, Accuracy: {total_event_correct/total_samples}')
    # 评估论元抽取与分类模型
    # for batch in eval_args_dataloader:
    #     input_ids, label_ids, attention_mask,  text = batch
    #     input_ids = input_ids.to(device)
    #     label_ids = label_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    #     args_logits = args_model(input_ids, attention_mask)
    #     args_loss = args_model.loss(args_logits, label_ids)
    #     total_loss += args_loss.item()
    #     args_preds = torch.argmax(args_logits, dim=1)
    #     total_args_correct += torch.sum(args_preds == label_ids).item()
    #     total_samples += len(label_ids)
