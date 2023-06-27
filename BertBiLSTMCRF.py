import torch
import torch.nn as nn
from torchcrf import CRF


class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model, num_classes, freeze_bert=False):
        super(BertBiLSTMCRF, self).__init__()
        self.num_classes = num_classes
        self.bert = bert_model
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
        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[1][-1]
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        if labels is None:
            return self.crf.decode(logits, mask=attention_mask.byte())
        log_likelihood = self.crf(logits, labels)
        return (-1) * log_likelihood
