import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from torch.utils.data import DataLoader
from BertBiLSTMCRF import BertBiLSTMCRF
from EventExtratorClassifer import EventExtractorClassifer
from event_dataset import EventDataset
from arguments_dataset import ArgumentsDataset


# 设置超参数
num_event_types = 9  # 事件类型数量
num_arguments = 12  # 论元分类数量
learning_rate = 1e-5
num_epochs = 30
batch_size = 4  # 16

# device specific
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Bert模型和分词器
tokenizer = AutoTokenizer.from_pretrained('../models/xlm-roberta-base')
config = AutoConfig.from_pretrained(
    "../models/xlm-roberta-base", output_hidden_states=True)
model = AutoModelForMaskedLM.from_pretrained(
    "../models/xlm-roberta-base", config=config).to(device)

# 训练数据
train_args_dataset = ArgumentsDataset(
    './data//FNDEE_train1.json', tokenizer)  # 论元数据集
train_event_dataset = EventDataset(
    './data//FNDEE_train1.json', tokenizer)  # 事件数据集
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
event_type_criterion = nn.CrossEntropyLoss()
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
    avg_train_loss = total_loss / len(train_event_dataloader)
    print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}')


# Optimizer
args_optimizer = torch.optim.AdamW(
    args_model.parameters(), lr=learning_rate, eps=1e-8)

# Training loop for args extraction and classification training
for epoch in range(num_epochs):
    args_model.train()
    total_loss = 0
    for batch in train_args_dataloader:
        input_ids, label_ids, attention_mask,  _ = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss = args_model(
            input_ids, attention_mask=attention_mask, labels=label_ids)
        loss.backward()
        args_optimizer.step()
        args_optimizer.zero_grad()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_args_dataloader)
    print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}')

# get current timestamp and convert to string
timestamp = str(int(time.time()))
# create file name using timestamp
event_filename = f"../result/models/event_model_{timestamp}"
args_filename = f"../result/models/args_model_{timestamp}"
torch.save(event_model.state_dict, event_filename)
torch.save(args_model.state_dict, args_filename)
print(f"model file {event_filename} and {args_filename} saved")

# Evaluation loop
# model.eval()
# with torch.no_grad():
#     eval_loss = 0
#     for batch in valid_dataloader:
#         b_input_ids, b_input_mask, b_labels = tuple(
#             t.to(device) for t in batch)
#         loss = model(b_input_ids, token_type_ids=None,
#                      attention_mask=b_input_mask, labels=b_labels)
#         eval_loss += loss.item()
#     avg_eval_loss = eval_loss / len(valid_dataloader)
#     print(f'Validation Loss: {avg_eval_loss}')


# # 使用训练好的模型进行预测（假设有测试数据）
# test_data = ...
# predictions = []

# for data in test_data:
#     text = data['text']
#     encoded = tokenizer.encode(text, add_special_tokens=True)
#     input_ids = torch.tensor(encoded).unsqueeze(0)

#     event_type_logits, argument_logits = extractor(input_ids)

#     predicted_event_type = torch.argmax(event_type_logits, dim=1).item()
#     predicted_argument = torch.argmax(argument_logits, dim=1).item()

#     predictions.append({'event_type': predicted_event_type,
#                        'argument': predicted_argument})

# print(predictions)
