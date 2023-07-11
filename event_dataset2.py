from datasets import load_dataset
import numpy as np
import torch
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
    def __init__(
        self,
        train_file,
        valid_file,
        tokenizer,
        mask=True,
        mask_ratio=0.3,
        max_mask_num=10,
        replace=True,
        replace_ratio=0.3,
        max_replace_num=5,
    ) -> None:
        """
        Initializes an EventDataset object.

        Args:
            train_file (str): The path to the training data file.
            valid_file (str): The path to the validation data file.
            tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the text.
            mask (bool, optional): Whether to mask some tokens with [MASK]. Defaults to True.
            mask_radio (float, optional): The ratio of tokens to mask. Defaults to 0.3.
            replace (bool, optional): Whether to replace argument tokens with the same type of entity. Defaults to True.
            replace_radio (float, optional): The ratio of argument tokens to replace. Defaults to 0.3.
        """
        # Load the dataset from the given files
        self.dataset = load_dataset(
            "json", data_files={"train": train_file, "test": valid_file}
        )
        self.tokenizer = tokenizer
        self.mask_id = 103  # [MASK] token ID
        # Tokenize and align the labels for each example in the dataset
        self.processed_dataset = self.dataset.map(
            self.__tokenize_and_align_labels__, batched=False
        )  # .remove_columns(["text", "event_list", "coref_arguments", "id"])

        # from the processed_dataset, collects all event trigger and it's corresponding arguments role and text
        self.event_arguments_dict = self.__create_event_args_dict__()

        self.mask = mask
        self.replace = replace
        if not mask and not replace:
            self.processed_dataset = self.processed_dataset.remove_columns(
                ["text", "event_list", "coref_arguments", "id"]
            )
        # Apply the mask and replace transformations to the training data if specified
        self.mask_radio = mask_ratio
        self.replace_radio = replace_ratio
        self.max_mask_num = max_mask_num
        self.max_replace_num = max_replace_num
        if mask and replace:
            self.processed_dataset["train"].set_transform(self.__mask_and_replace__)
        elif mask:
            self.processed_dataset["train"].set_transform(self.__mask__)
        elif replace:
            self.processed_dataset["train"].set_transform(self.__replace__)

    def __create_event_args_dict__(self):
        event_arguments_dict = {}
        for events in self.processed_dataset["train"]["event_list"]:
            for event in events:
                event_type = event["event_type"]
                if event_type not in event_arguments_dict:
                    event_arguments_dict[event_type] = {}
                arguments = [(arg["role"], arg["text"]) for arg in event["arguments"]]
                arguments_dict = event_arguments_dict[event_type]
                for role, text in arguments:
                    if role in arguments_dict:
                        arguments_dict[role].append(text)
                    else:
                        arguments_dict[role] = [text]
        return event_arguments_dict

    def __tokenize_and_align_labels__(self, examples):
        """
        Tokenizes the text in the given examples and aligns the labels to the tokenized text.

        Args:
            examples (dict): A dictionary containing the text and event list for an example.

        Returns:
            dict: A dictionary containing the tokenized text and the aligned labels.
        """
        tokenized_inputs = self.tokenizer(examples["text"], is_split_into_words=False)

        labels = []
        for event in examples["event_list"]:
            event_type = event["event_type"]
            trigger_text = event["trigger"]["text"]
            trigger_offset = event["trigger"]["offset"]
            labels.append((event_type, trigger_text, trigger_offset))
        # Convert the text to token IDs
        tokens = self.tokenizer.tokenize(examples["text"])
        # Align the labels to the tokenized text
        aligned_labels = align_labels(examples["text"], labels, tokens)

        # Convert the labels to indices
        label_ids = [label_to_index[label] for label in aligned_labels]
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    def __mask__(self, examples, remove_columns=True):
        """
        Masks some tokens with [MASK] in the given examples.

        Args:
            examples (dict): A dictionary containing the tokenized text and the aligned labels.

        Returns:
            dict: A dictionary containing the masked tokenized text and the aligned labels.
        """
        # Mask some tokens with [MASK], those masked token should not be trigger and arguments
        random_list = torch.rand(len(examples["input_ids"]))
        idxs = list(range(len(examples["input_ids"])))
        selected_idx = [
            idx for value, idx in zip(random_list, idxs) if value < self.mask_radio
        ]
        for idx in selected_idx:
            # select a random token which not a trigger or argument, then mask it
            labels = examples["labels"][idx]
            random_token_choice = torch.rand(len(labels))
            labels_idx = list(range(len(labels)))
            masked_num = 0
            for label_idx, random_token, label in zip(
                labels_idx, random_token_choice, labels
            ):
                if (
                    random_token < self.mask_radio
                    and masked_num <= self.max_mask_num
                    and label == 0
                ):
                    examples["input_ids"][idx][label_idx] = self.mask_id
                    examples["attention_mask"][idx][label_idx] = 0
                    masked_num += 1
        if remove_columns:
            self.__remove_columns__(examples)
        return examples

    def __remove_columns__(self, examples):
        for key in ["text", "event_list", "coref_arguments", "id"]:
            examples.pop(key)

    def __replace__(self, examples, remove_columns=True):
        """
        Replaces argument tokens with the same type of entity in the given examples.

        Args:
            examples (dict): A dictionary containing the tokenized text and the aligned labels.

        Returns:
            dict: A dictionary containing the replaced tokenized text and the aligned labels.
        """
        # select which example's trigger arguments will be replaced
        random_list = torch.rand(len(examples["input_ids"]))
        idxs = list(range(len(examples["input_ids"])))
        selected_idx = [
            idx for value, idx in zip(random_list, idxs) if value < self.replace_radio
        ]

        # replace the selected example's trigger arguments
        for idx in selected_idx:
            # collects all event trigger and it's corresponding arguments role and text
            event_list = examples["event_list"][idx]
            arguments = []
            event_labels = []

            for event in event_list:
                arguments.extend(
                    (event["event_type"], arg["role"], arg["text"], arg["offset"])
                    for arg in event["arguments"]
                )
                event_type = event["event_type"]
                trigger_text = event["trigger"]["text"]
                trigger_offset = event["trigger"]["offset"]
                event_labels.append((event_type, trigger_text, trigger_offset))

            random_list = torch.rand(len(arguments))
            # slelect which argument will be replaced
            selected_args = [
                arg
                for value, arg in zip(random_list, arguments)
                if value < self.replace_radio
            ]

            replace_num = 0
            pos_modify = 0
            text = examples["text"][idx]
            for arg in selected_args:
                if replace_num <= self.max_replace_num:
                    event_type = arg[0]
                    start_idx = arg[3][0] + pos_modify
                    old_text = arg[2]
                    new_text = np.random.choice(
                        self.event_arguments_dict[event_type][arg[1]]
                    )
                    old_len = len(text)
                    text = text[:start_idx] + text[start_idx:].replace(
                        old_text, new_text, 1
                    )
                    pos_modify += len(text) - old_len
                    replace_num += 1
                    # update the labels offset
                    self.__update_event_labels_pos__(
                        event_labels, start_idx, len(text) - old_len
                    )
            inputs = self.tokenizer(text)
            examples["input_ids"][idx] = inputs["input_ids"]
            examples["attention_mask"][idx] = inputs["attention_mask"]
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
            # Align the labels to the tokenized text
            aligned_labels = align_labels(text, event_labels, tokens)
            label_ids = [label_to_index[label] for label in aligned_labels]
            examples["labels"][idx] = label_ids

        if remove_columns:
            self.__remove_columns__(examples)
        return examples

    def __mask_and_replace__(self, examples):
        examples = self.__mask__(examples, remove_columns=False)
        examples = self.__replace__(examples)
        return examples

    def __update_event_labels_pos__(self, event_labels, start_idx, modify):
        if modify == 0:
            return
        for event_label in event_labels:
            if event_label[2][0] >= start_idx:
                event_label[2][0] += modify
                event_label[2][1] += modify

    def __getitem__(self, key: Any):
        """
        Returns the item at the given key in the processed dataset.

        Args:
            key (Any): The key of the item to return.

        Returns:
            dict: The item at the given key in the processed dataset.
        """
        if key == "test" and (self.mask or self.replace):
            return self.processed_dataset.remove_columns(
                ["text", "event_list", "coref_arguments", "id"]
            )[key]
        return self.processed_dataset[key]

    @property
    def num_event_types(self) -> int:
        """
        Returns the number of event types.

        Returns:
            int: The number of event types.
        """
        return len(label_to_index)

    @property
    def label_names(self):
        """
        Returns a dictionary mapping label indices to label names.

        Returns:
            dict: A dictionary mapping label indices to label names.
        """
        return label_names

    @classmethod
    @property
    def LabelNames(cls):
        """
        Returns a dictionary mapping label indices to label names.

        Returns:
            dict: A dictionary mapping label indices to label names.
        """
        return label_names

    @property
    def class_weights(self):
        """
        Returns the class weights.

        Returns:
            list: The class weights.
        """
        return [
            0.05,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
        ]
