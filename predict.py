# predict the input text with the trained model
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoConfig,
)

from EventExtratorClassifer import EventExtractorClassifer
from arguments_dataset2 import ArgumentsDataset
from event_dataset2 import EventDataset
from datasets import load_dataset

num_event_types = 17
num_args_types = 23
bert_name = "../models/xlm-roberta-base-ner-hrl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
bert_model = AutoModelForTokenClassification.from_pretrained(
    bert_name, config=config
).to(device)
tokenizer = AutoTokenizer.from_pretrained(bert_name)

# events prediction model
event_model = EventExtractorClassifer(bert_model, num_labels=num_event_types).to(device)
event_model.load_state_dict(
    torch.load(
        "../result/models/1688560366/model_35-5617.pt",
        map_location=torch.device(device),
    )
)

# arguments prediction model
bert_model_args = AutoModelForTokenClassification.from_pretrained(
    bert_name, config=config
).to(device)
args_model = EventExtractorClassifer(bert_model_args, num_labels=num_args_types).to(
    device
)
args_model.load_state_dict(
    torch.load(
        "../result/models-args/1688708719/model_48.pt",
        map_location=torch.device(device),
    )
)
dataset = load_dataset("json", data_files="./data/FNDEE_valid.json")
event_model.eval()
# args_model.eval()

for sample in dataset["train"]:
    text = sample["text"]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = event_model(**inputs)
    predictions = outputs.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze(),
        skip_special_tokens=True,
    )
    predicted_labels = [
        EventDataset.LabelNames[prediction.item()] for prediction in predictions[0]
    ]
    # print(" ".join([f"{token}/{label}" for token, label in zip(tokens, predicted_labels)]))
    labeld_tokens = [
        (idx, predicted_labels[idx], tokens[idx])
        for idx in range(len(tokens))
        if predicted_labels[idx] != "O"
    ]

    merged_tokens = []
    for labeld_token in labeld_tokens:
        idx, t, trigger = labeld_token
        if t.startswith("B"):
            merged_tokens.append((idx, t, trigger))
        elif t.startswith("I"):
            merged_tokens[-1] = (
                merged_tokens[-1][0],
                merged_tokens[-1][1],
                merged_tokens[-1][2] + trigger,
            )

    trigger_text = ""
    trigger_position = 0
    for idx, label, token in merged_tokens:
        trigger_text = token
        trigger_position = idx
        prompt_text = f"{trigger_text},位置{trigger_position}"
        prompted_text = f"{tokenizer.cls_token}{prompt_text}{tokenizer.sep_token}{text}"
        inputs = tokenizer(prompted_text, return_tensors="pt").to(device)
        args_outputs = args_model(**inputs)
        args_predictions = args_outputs.argmax(dim=-1)
        args_tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze(),
            # skip_special_tokens=True,
        )
        args_predicted_labels = [
            ArgumentsDataset.LabelNames[prediction.item()]
            for prediction in args_predictions[0]
        ]
