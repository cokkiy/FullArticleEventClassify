import torch
import torch.nn as nn
from transformers import BertModel


class EventExtractorClassifer(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(EventExtractorClassifer, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
