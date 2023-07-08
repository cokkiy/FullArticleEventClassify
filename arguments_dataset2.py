from datasets import load_dataset, Dataset
from traitlets import Any
from transformers import AutoTokenizer
from commonfn import align_labels, get_token_index

label_to_index = {
    # convert label to index
    "O": 0,
    "B-Subject": 1,
    "I-Subject": 2,
    "B-Equipment": 3,
    "I-Equipment": 4,
    "B-Date": 5,
    "I-Date": 6,
    "B-Location": 7,
    "I-Location": 8,
    "B-Area": 9,
    "I-Area": 10,
    "B-Content": 11,
    "I-Content": 12,
    "B-Militaryforce": 13,
    "I-Militaryforce": 14,
    "B-Object": 15,
    "I-Object": 16,
    "B-Materials": 17,
    "I-Materials": 18,
    "B-Result": 19,
    "I-Result": 20,
    "B-Quantity": 21,
    "I-Quantity": 22,
}

label_names = {
    0: "O",
    1: "B-Subject",
    2: "I-Subject",
    3: "B-Equipment",
    4: "I-Equipment",
    5: "B-Date",
    6: "I-Date",
    7: "B-Location",
    8: "I-Location",
    9: "B-Area",
    10: "I-Area",
    11: "B-Content",
    12: "I-Content",
    13: "B-Militaryforce",
    14: "I-Militaryforce",
    15: "B-Object",
    16: "I-Object",
    17: "B-Materials",
    18: "I-Materials",
    19: "B-Result",
    20: "I-Result",
    21: "B-Quantity",
    22: "I-Quantity",
}


class ArgumentsDataset:
    def __init__(self, train_file, valid_file, tokenizer, max_length=512) -> None:
        dataset = load_dataset(
            "json", data_files={"train": train_file, "test": valid_file}
        )
        self.tokenizer = tokenizer
        processed_dataset1 = dataset.map(
            self.__expand_and_align_labels__,
            batched=False,
            remove_columns=["text", "event_list", "coref_arguments", "id"],
        )
        processed_dataset2 = {
            "train": {"input_ids": [], "attention_mask": [], "labels": []},
            "test": {"input_ids": [], "attention_mask": [], "labels": []},
        }
        for k, v in processed_dataset1.items():
            for item in v:
                self.to_flatten(k, item, processed_dataset2)

        self.processed_dataset = {
            "train": Dataset.from_dict(processed_dataset2["train"]),
            "test": Dataset.from_dict(processed_dataset2["test"]),
        }

    def __expand_and_align_labels__(self, examples):
        text = examples["text"]
        labels = []
        data = {"input_ids": [], "attention_mask": [], "labels": []}
        for event in examples["event_list"]:
            event_type = event["event_type"]
            trigger_text = event["trigger"]["text"]
            trigger_offset = event["trigger"]["offset"]
            arguments = [
                (argument["role"], argument["text"], argument["offset"])
                for argument in event["arguments"]
            ]
            labels.append((event_type, trigger_text, trigger_offset, arguments))

        # 将文本转换为标记的ID
        input_tokens = self.tokenizer.tokenize(
            text, max_length=512, truncation=True, padding=False
        )

        # 对齐标签到分词后的文本
        for event_type, trigger_text, trigger_offset, arguments in labels:
            # 对齐标签到分词后的文本
            prompt_position = get_token_index(text, input_tokens, trigger_offset[0])
            prompt_text = f"{trigger_text},位置{prompt_position}"
            prompt_tokens = self.tokenizer.tokenize(prompt_text)

            # 将Prompt和输入文本的分词结果合并
            tokens = (
                [self.tokenizer.cls_token]
                + prompt_tokens
                + [self.tokenizer.sep_token]
                + input_tokens
            )
            prompted_text = f"{self.tokenizer.cls_token}{prompt_text}{self.tokenizer.sep_token}{text}"
            aligned_labels = align_labels(text, arguments, input_tokens)
            aligned_labels = ["O"] * (len(tokens) - len(input_tokens)) + aligned_labels
            label_ids = [label_to_index[label] for label in aligned_labels]
            inputs = self.tokenizer(
                prompted_text,
                add_special_tokens=False,
                max_length=512,
                truncation=True,
                padding=False,
            )
            data["input_ids"].append(inputs["input_ids"])
            data["attention_mask"].append(inputs["attention_mask"])
            data["labels"].append(label_ids)
        return data

    def to_flatten(self, key, examples, processed_dataset):
        for i in range(len(examples["input_ids"])):
            processed_dataset[key]["input_ids"].append(examples["input_ids"][i])
            processed_dataset[key]["attention_mask"].append(
                examples["attention_mask"][i]
            )
            processed_dataset[key]["labels"].append(examples["labels"][i])

    def __getitem__(self, key: Any):
        return self.processed_dataset[key]

    def __len__(self):
        return len(self.processed_dataset)

    def __len__(self, key):
        return len(self.processed_dataset[key])

    @property
    def num_args_types(self) -> int:
        """get number of event types"""
        return len(label_to_index)

    @property
    def label_names(self):
        return label_names

    @classmethod
    @property
    def LabelNames(cls):
        return label_names

    @property
    def class_weights(self):
        """get class weights"""
        return [
            0.05,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
