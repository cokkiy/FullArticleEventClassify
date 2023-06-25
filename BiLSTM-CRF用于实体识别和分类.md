# 以下是使用PyTorch微调Bert预训练模型，结合BiLSTM-CRF用于实体识别和分类的代码示例：

首先，你需要安装 pytorch-transformers 库，它提供了许多预训练模型，包括Bert。

```python
!pip install pytorch-transformers
```

接下来，我们可以加载 Bert 预训练模型并添加一个双向LSTM层和CRF层。 

```python
import torch
import torch.nn as nn
from pytorch_transformers import BertModel

class BertBiLSTMCRF(nn.Module):
    def __init__(self, num_classes, freeze_bert=False):
        super(BertBiLSTMCRF, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=self.bert.config.to_dict()['hidden_size'],
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(256, self.num_classes)
        self.crf = CRF(num_tags=self.num_classes, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            return (-1) * log_likelihood
        else:
            return self.crf.decode(logits, mask=attention_mask.byte())

class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=False):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        self.transitions.data[:, 0] = -10000.0
        self.transitions.data[0, :] = -10000.0

    def forward(self, logits, labels, mask=None):
        if mask is None:
            mask = torch.ones(logits.shape[:2], dtype=torch.long)

        if self.batch_first:
            logits = logits.transpose(0, 1)
            labels = labels.transpose(0, 1)
            mask = mask.transpose(0, 1)

        batch_size, seq_len, _ = logits.shape
        mask = mask.float()

        alpha = torch.Tensor(batch_size, 1, self.num_tags).fill_(-10000.0)
        alpha[:, 0, 0] = 0.0
        alpha = alpha.cuda() if torch.cuda.is_available() else alpha

        for i in range(seq_len):
            emit_score = logits[:, i, :].unsqueeze(1).repeat(1, self.num_tags, 1)
            trans_score = self.transitions.unsqueeze(0).repeat(batch_size, 1, 1)
            score = emit_score + trans_score + alpha
            score = torch.logsumexp(score, dim=2)
            alpha = score * mask[:, i].unsqueeze(1) + alpha * (1 - mask[:, i]).unsqueeze(1)

        Z = torch.logsumexp(alpha, dim=2)
        all_score = logits.new_zeros((batch_size,))
        for b in range(batch_size):
            if seq_len[b] == 0:
                all_score[b] = 0.0
            else:
                all_score[b] = Z[b]

        emit_score = logits.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1) * mask
        trans_score = self.transitions[labels[:, :-1], labels[:, 1:]]
        trans_score *= mask[:, :-1]
        log_likelihood = (emit_score + trans_score).sum(dim=1) - all_score

        return log_likelihood.mean()

```

这里，我们定义了一个名为 BertBiLSTMCRF 的 PyTorch 模型，该模型具有以下特征：

* 通过 BertModel.from_pretrained() 方法加载预训练 Bert 模型。
* 可以选择冻结 Bert 模型的参数，以便仅更新 BiLSTM 和 CRF 层的参数。
* 添加一个双向 LSTM 层，并将其输入嵌入 Bert 的 hidden states 中。
* 添加一个线性层来执行分类任务。
* 添加一个CRF层，以便可以在训练过程中执行会话解码。

在训练模型时，我们使用 CRF 层计算负对数对数似然，以便在后续会话解码方面进行训练。如果没有标签，则使用 forward() 方法中的解码操作。 

```python
from torch.utils.data import TensorDataset, DataLoader

# Create dataset
train_dataset = TensorDataset(train_inputs, train_masks, train_tags)
valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_tags)

# Dataloader
batch_size = 32
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = BertBiLSTMCRF(num_classes=len(label2id), freeze_bert=False)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(dataloader)
    print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}')

# Evaluation loop
model.eval()
with torch.no_grad():
    eval_loss = 0
    for batch in valid_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        eval_loss += loss.item()
    avg_eval_loss = eval_loss / len(valid_dataloader)
    print(f'Validation Loss: {avg_eval_loss}')
```

这里，我们使用 PyTorch 的 DataLoader 从我们的训练和验证数据集中加载数据，设置优化器和训练循环（epoch）。 我们在训练模型时，使用带有 backward() 和 zero_grad() 方法的自动微分机制来计算梯度，并使用 AdamW 优化器进行参数优化。 在评估模型时，我们通过设置 model.eval() 来关闭 dropout 和 batch normalization。

希望这可以帮助您了解如何使用Bert预训练模型，并结合CRF和BiLSTM进行实体识别和分类任务的微调。