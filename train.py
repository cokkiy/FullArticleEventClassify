import time
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

# get current timestamp and convert to string
timestamp = str(int(time.time()))

parser = argparse.ArgumentParser(
    description='Train model on the FNDEE dataset')
parser.add_argument('--path', type=str, default='../result/models',
                    help='The file path will save the trained model (default: ../result/models)')
parser.add_argument('--model_name', type=str, default=f'{timestamp}',
                    help='The file name of the trained model will be saved (default: timestamp)')
parser.add_argument('--train_file', type=str, default='./data/FNDEE_train1.json',
                    help='The file name of the train dateset file')
parser.add_argument('--batch_size', type=int,
                    help='Train Batch size, default 4', default=4)
parser.add_argument('--bert_model', type=str, default='../models/xlm-roberta-base',
                    help='Pretrained bert model name (default: "../models/xlm-roberta-base")')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs to train (default: 20)')

args = parser.parse_args()

path = args.path
model_name = args.model_name
train_file = args.train_file
batch_size = args.batch_size
bert_name = args.bert_model
num_epochs = args.num_epochs

if not os.path.exists(path):
    os.makedirs(path)

# 设置超参数
num_event_types = 9  # 事件类型数量
num_arguments = 12  # 论元分类数量
learning_rate = 1e-5


# device specific
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Bert模型和分词器
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained(bert_name)
config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
model = AutoModelForMaskedLM.from_pretrained(
    bert_name, config=config).to(device)

# 训练数据
print("Loading training data...")
train_args_dataset = ArgumentsDataset(train_file, tokenizer)  # 论元数据集
train_event_dataset = EventDataset(train_file, tokenizer)  # 事件数据集
# Dataloader
train_event_dataloader = DataLoader(
    train_event_dataset, batch_size=batch_size, collate_fn=train_event_dataset.pad_collate_fn, shuffle=False)
train_args_dataloader = DataLoader(
    train_args_dataset, batch_size=batch_size, collate_fn=train_args_dataset.pad_collate_fn, shuffle=False)

# event extract and classifier model
event_model = EventExtractorClassifer(
    bert_model=model, num_labels=num_event_types).to(device)
# arguments extract and classify model
args_model = BertBiLSTMCRF(
    model, num_classes=num_arguments, freeze_bert=True).to(device)

# Optimizer
event_optimizer = torch.optim.AdamW(
    event_model.parameters(), lr=learning_rate, eps=1e-8)

# Training loop for event extraction and classification training
print("Training event extraction and classification model...")
weight = torch.tensor([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
event_type_criterion = nn.CrossEntropyLoss()
prf1 = PRF1Calculator()
for epoch in range(num_epochs):
    event_model.train()
    total_loss = 0
    for batch in train_event_dataloader:
        event_optimizer.zero_grad()
        input_ids, label_ids, attention_mask,  _, _ = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device)
        event_type_logits = event_model(input_ids, attention_mask)
        loss = event_type_criterion(
            event_type_logits.view(-1, num_event_types), label_ids.view(-1))
        loss.backward()
        event_optimizer.step()
        total_loss += loss.item()
        event_preds = torch.argmax(event_type_logits[1], dim=1)
        prf1.update(event_preds, label_ids.view(-1))
    avg_train_loss = total_loss / len(train_event_dataloader)
    print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}')
    print(
        f'     Trigger Precision: {prf1.P}, Recall: {round(prf1.R, 4)}, F1: {round(prf1.F1, 4)}')


print("Training arguments extraction and classification model...")
# Optimizer
args_optimizer = torch.optim.AdamW(
    args_model.parameters(), lr=learning_rate, eps=1e-8)

# Training loop for args extraction and classification training
# for epoch in range(num_epochs):
#     args_model.train()
#     total_loss = 0
#     for batch in train_args_dataloader:
#         input_ids, label_ids, attention_mask,  _ = batch
#         input_ids = input_ids.to(device)
#         label_ids = label_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         loss = args_model(
#             input_ids, attention_mask=attention_mask, labels=label_ids)
#         loss.backward()
#         args_optimizer.step()
#         args_optimizer.zero_grad()
#         total_loss += loss.item()
#     avg_train_loss = total_loss / len(train_args_dataloader)
#     print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}')

print("Saving model...")
event_filename = f"{path}/{model_name}-event.pt"
args_filename = f"{path}/{model_name}-args.pt"
torch.save(event_model.state_dict(), event_filename)
# torch.save(args_model.state_dict(), args_filename)
print(f"model file {event_filename} and {args_filename} saved")
