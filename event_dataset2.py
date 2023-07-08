from datasets import load_dataset
from traitlets import Any
from transformers import AutoTokenizer
from commonfn import align_labels

label_to_index = {
    # convert label to index
    "O": 0,
    "B-Experiment": 1,
    "I-Experiment": 2,
    "B-Manoeuvre": 3,
    "I-Manoeuvre": 4,
    "B-Deploy": 5,
    "I-Deploy": 6,
    "B-Support": 7,
    "I-Support": 8,
    "B-Accident": 9,
    "I-Accident": 10,
    "B-Exhibit": 11,
    "I-Exhibit": 12,
    "B-Conflict": 13,
    "I-Conflict": 14,
    "B-Injure": 15,
    "I-Injure": 16,
}

label_names = {
    0: "O",
    1: "B-Experiment",
    2: "I-Experiment",
    3: "B-Manoeuvre",
    4: "I-Manoeuvre",
    5: "B-Deploy",
    6: "I-Deploy",
    7: "B-Support",
    8: "I-Support",
    9: "B-Accident",
    10: "I-Accident",
    11: "B-Exhibit",
    12: "I-Exhibit",
    13: "B-Conflict",
    14: "I-Conflict",
    15: "B-Injure",
    16: "I-Injure",
}


class EventDataset:
    def __init__(self, train_file, valid_file, tokenizer) -> None:
        self.dataset = load_dataset(
            "json", data_files={"train": train_file, "test": valid_file}
        )
        self.tokenizer = tokenizer
        self.processed_dataset = self.dataset.map(
            self.__tokenize_and_align_labels__, batched=False
        ).remove_columns(["text", "event_list", "coref_arguments", "id"])

    def __tokenize_and_align_labels__(self, examples):
        tokenized_inputs = self.tokenizer(examples["text"], is_split_into_words=False)

        labels = []
        for event in examples["event_list"]:
            event_type = event["event_type"]
            trigger_text = event["trigger"]["text"]
            trigger_offset = event["trigger"]["offset"]
            labels.append((event_type, trigger_text, trigger_offset))
        # 将文本转换为标记的ID
        tokens = self.tokenizer.tokenize(examples["text"])
        # 对齐标签到分词后的文本
        aligned_labels = align_labels(examples["text"], labels, tokens)

        # 将标签转换为索引
        label_ids = [label_to_index[label] for label in aligned_labels]
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    def __getitem__(self, key: Any):
        return self.processed_dataset[key]

    @property
    def num_event_types(self) -> int:
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
        ]
