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
        output = self.bert(input_ids, attention_mask)
        pooled_output = output[1][-1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def save_pretrained(self, path, save_function=None):
        if save_function is None:
            save_function = torch.save
        save_function(self.state_dict(), path)
