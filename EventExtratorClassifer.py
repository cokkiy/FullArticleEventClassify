import torch
import torch.nn as nn
from transformers import BertModel


class EventExtractorClassifer(nn.Module):
    def __init__(self, bert_model, num_labels, freeze_bert=False):
        super(EventExtractorClassifer, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = outputs[1][-1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
