import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 加载预训练的GPT模型和分词器
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# 定义事件抽取和分类模型
class EventExtractor(nn.Module):
    def __init__(self, gpt_model, num_event_types, num_arguments):
        super(EventExtractor, self).__init__()
        self.gpt_model = gpt_model
        self.event_type_classifier = nn.Linear(gpt_model.config.hidden_size, num_event_types)
        self.argument_classifier = nn.Linear(gpt_model.config.hidden_size, num_arguments)
    
    def forward(self, input_ids):
        outputs = self.gpt_model(input_ids)
        pooled_output = outputs[1]
        event_type_logits = self.event_type_classifier(pooled_output)
        argument_logits = self.argument_classifier(pooled_output)
        return event_type_logits, argument_logits

# 设置超参数
import itertools
num_event_types = 5  # 事件类型数量
num_arguments = 10  # 论元分类数量
learning_rate = 1e-5
num_epochs = 10
batch_size = 16

# 准备训练数据（假设已经有了）
train_data = ...

# 创建模型实例
extractor = EventExtractor(model, num_event_types, num_arguments)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(extractor.parameters(), lr=learning_rate)
event_type_criterion = nn.CrossEntropyLoss()
argument_criterion = nn.CrossEntropyLoss()

# 微调模型
for _, i in itertools.product(range(num_epochs), range(0, len(train_data), batch_size)):
    batch_data = train_data[i:i+batch_size]
    input_ids = []
    event_type_labels = []
    argument_labels = []
    for data in batch_data:
        text = data['text']
        event_type = data['event_type']
        argument = data['argument']

        encoded = tokenizer.encode(text, add_special_tokens=True)
        input_ids.append(encoded)
        event_type_labels.append(event_type)
        argument_labels.append(argument)

    input_ids = torch.tensor(input_ids)
    event_type_labels = torch.tensor(event_type_labels)
    argument_labels = torch.tensor(argument_labels)

    optimizer.zero_grad()
    event_type_logits, argument_logits = extractor(input_ids)

    event_type_loss = event_type_criterion(event_type_logits, event_type_labels)
    argument_loss = argument_criterion(argument_logits, argument_labels)

    loss = event_type_loss + argument_loss
    loss.backward()
    optimizer.step()

# 使用训练好的模型进行预测（假设有测试数据）
test_data = ...
predictions = []

for data in test_data:
    text = data['text']
    encoded = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    event_type_logits, argument_logits = extractor(input_ids)

    predicted_event_type = torch.argmax(event_type_logits, dim=1).item()
    predicted_argument = torch.argmax(argument_logits, dim=1).item()

    predictions.append({'event_type': predicted_event_type, 'argument': predicted_argument})

print(predictions)