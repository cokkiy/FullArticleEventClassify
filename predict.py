# predict the input text with the trained model
import argparse
import json
import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoConfig,
    BertModel,
)
from ArgumentsExtratorClassifer import ArgumentsExtratorClassifer

from EventExtratorClassifer import EventExtractorClassifer
from arguments_dataset2 import ArgumentsDataset
from event_dataset2 import EventDataset
from datasets import load_dataset
from commonfn import convert_tokenoffset_to_charoffset
from GlobalExtractor import EffiGlobalExtractor as GlobalExtractor

parser = argparse.ArgumentParser(description="Predict test data with trained model")
parser.add_argument(
    "--event_model",
    type=str,
    default="../result/lu/ent_model.pth",
    help="The trained event model path",
)

parser.add_argument(
    "--args_model",
    type=str,
    default="../result/model_19/model_19.pt",
    help="The trained arguments model path",
)

parser.add_argument(
    "--test_file",
    type=str,
    default="./data/FNDEE_valid.json",
    help="The file name of the test json file",
)

args = parser.parse_args()
event_model_name = args.event_model
arg_model_name = args.args_model
test_file = args.test_file

num_event_types = 17
num_args_types = 23
event_bert_name = "../models/roberta_wwm"
bert_ner_name = "../models/xlm-roberta-base-ner-hrl"
bert_name = "../models/xlm-roberta-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(bert_ner_name, output_hidden_states=True)
bert_model = BertModel.from_pretrained(event_bert_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(event_bert_name)

# 事件类型数量
ENT_CLS_NUM = 9
# 事件普通论元数量
ARGUMENT_CLS_NUM = 11

# events prediction model
event_model = GlobalExtractor(bert_model, ENT_CLS_NUM, ARGUMENT_CLS_NUM, 64).to(device)
event_model.load_state_dict(
    torch.load(
        event_model_name,
        map_location=torch.device(device),
    )
)

# arguments prediction model
bert_ner_model_args = AutoModelForTokenClassification.from_pretrained(
    bert_ner_name, config=config
).to(device)
config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
bert_model_args = AutoModelForMaskedLM.from_pretrained(bert_name, config=config).to(
    device
)
args_model = ArgumentsExtratorClassifer(
    bert_ner_model_args, bert_model=bert_model_args, num_labels=num_args_types
).to(device)
args_model.load_state_dict(
    torch.load(
        arg_model_name,
        map_location=torch.device(device),
    )
)
dataset = load_dataset("json", data_files=test_file)
event_model.eval()
args_model.eval()
results = []  # store the results
current_arg_end_idx = 0
for sample in dataset["train"]:
    text = sample["text"]
    event_inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt").to(
        device
    )
    outputs = event_model(event_inputs["input_ids"], event_inputs["attention_mask"])
    predictions = outputs.argmax(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(
        event_inputs["input_ids"].squeeze(),
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
        if t.startswith("B-"):
            merged_tokens.append((idx, t, trigger))
        elif t.startswith("I-") and len(merged_tokens) > 0:
            merged_tokens[-1] = (
                merged_tokens[-1][0],
                merged_tokens[-1][1],
                merged_tokens[-1][2] + trigger,
            )

    event_offsets = convert_tokenoffset_to_charoffset(event_inputs["offset_mapping"][0])
    # event list
    event_list = []
    event = {}

    trigger_text = ""
    trigger_position = 0
    for idx, label, token in merged_tokens:
        trigger_text = token
        trigger_position = idx
        # create the event node
        event = {
            "event_type": label[2:],
            "trigger": {
                "text": trigger_text,
                "offset": [
                    event_offsets[trigger_position][0].data.cpu().item()
                    + 1,  # start offset
                    event_offsets[trigger_position][0].data.cpu().item()
                    + 1
                    + len(trigger_text),  # end offfset
                ],
            },
            "arguments": [],
        }

        # extract the arguments
        prompt_text = f"{trigger_text},位置{trigger_position}"
        prompted_text = f"{tokenizer.cls_token}{prompt_text}{tokenizer.sep_token}{text}"
        inputs = tokenizer(
            prompted_text, return_offsets_mapping=True, return_tensors="pt"
        ).to(device)
        args_outputs = args_model(inputs["input_ids"], inputs["attention_mask"])
        args_predictions = args_outputs.argmax(dim=-1)
        args_tokens = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze(),
            # skip_special_tokens=True,
        )
        args_predicted_labels = [
            ArgumentsDataset.LabelNames[prediction.item()]
            for prediction in args_predictions[0]
        ]

        # if a label in middle is not I-*, but the previous and next labels are the same, then change the label to the previous label
        for i in range(len(args_predicted_labels)):
            if (
                args_predicted_labels[i] == "O"
                and i > 0
                and i + 1 < len(args_predicted_labels)
                and args_predicted_labels[i - 1].startswith("I-")
                and args_predicted_labels[i + 1] == args_predicted_labels[i - 1]
            ):
                args_predicted_labels[i] = args_predicted_labels[i - 1]
        args_offsets = convert_tokenoffset_to_charoffset(inputs["offset_mapping"][0])
        pre_label = "O"
        current_arg_text = ""
        current_arg_start_idx = 0
        current_arg_type = ""
        arguments = []
        for i in range(len(args_predicted_labels)):
            if args_predicted_labels[i].startswith("B-"):
                current_arg_type = args_predicted_labels[i].replace("B-", "")
                current_arg_text = args_tokens[i]
                current_arg_start_idx = args_offsets[i][0]
                # current_arg_end_idx = args_offsets[i][1]
                pre_label = args_predicted_labels[i]
            elif args_predicted_labels[i].startswith("I-"):
                if pre_label != "O":
                    current_arg_text += args_tokens[i]
                    # current_arg_end_idx = inputs["offset_mapping"][0][i][1]
                    pre_label = args_predicted_labels[i]
                else:
                    print("Warning: I- label is not after B- label")
            elif args_predicted_labels[i] == "O":
                if pre_label != "O":
                    arguments.append(
                        {
                            "role": current_arg_type,
                            "text": current_arg_text,
                            "offset": [
                                current_arg_start_idx.data.cpu().item()
                                - len(
                                    prompt_text
                                )  # should minus len of prompt text because we need the offset in the original text
                                - 2  # should minus 2 because of [CLS] and [SEP]
                                - 5,  # don't know why, but it works
                                current_arg_start_idx.data.cpu().item()
                                - len(prompt_text)
                                - 2
                                - 5
                                + len(current_arg_text),
                            ],
                        }
                    )
                pre_label = "O"
        event["arguments"] = arguments
        event_list.append(event)
    result = {"id": sample["id"], "event_list": event_list}
    results.append(result)

# save to file
if os.path.exists("../result/predict") is False:
    os.makedirs("../result/predict")
timestamp = str(int(time.time()))
with open(f"../result/predict/results_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
