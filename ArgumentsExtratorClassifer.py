import torch
import torch.nn as nn
from transformers import BertModel


class ArgumentsExtratorClassifer(nn.Module):
    def __init__(self, bert_ner_model, bert_model, num_labels, freeze_bert=False):
        super(ArgumentsExtratorClassifer, self).__init__()
        self.bert_ner = bert_ner_model
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.bert_ner.config.hidden_size + self.bert.config.hidden_size, num_labels
        )

        # not modify the NER model parameters
        for param in self.bert_ner.parameters():
            param.requires_grad = False

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output1 = output["hidden_states"][-1]
        ner_output = self.bert_ner(input_ids, attention_mask=attention_mask)
        pooled_output2 = ner_output["hidden_states"][-1]
        pooled_output = torch.cat((pooled_output1, pooled_output2), dim=2)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def save_pretrained(self, path, save_function=None):
        if save_function is None:
            save_function = torch.save
        save_function(self.state_dict(), path)
