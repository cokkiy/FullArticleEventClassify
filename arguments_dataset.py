
# 如果你的自定义JSON数据集中的文本和标签需要对齐，你可以按照以下步骤修改上述示例代码：
# 1. 加载和预处理数据：加载JSON数据集时，同时记录每个样本中文本和标签的位置信息。
# 2. 创建Dataset类：在Dataset类中，除了返回文本的标记ID和标签索引之外，还返回文本和标签的对齐信息。
# 3. 修改DataLoader：根据对齐信息，在DataLoader中对文本和标签进行对齐。
# 下面是一个简单的示例代码，展示了如何为具有文本和标签对齐需求的自定义JSON数据集创建一个适用于Transformer模型的DataLoader：

import json
import torch
from torch.utils.data import Dataset, DataLoader
from commonfn import align_labels, create_attention_mask, get_token_index
label_to_index = {
    # convert label to index
    'O': 0,
    'Subject': 1,
    'Equipment': 2,
    'Date': 3,
    'Location': 4,
    'Area': 5,
    'Content': 6,
    'Militaryforce': 7,
    'Object': 8,
    'Materials': 9,
    'Result': 10,
    'Quantity': 11
}


class ArgumentsDataset(Dataset):
    def __init__(self, data_path, tokenizer=None):
        self.data = []
        self.tokenizer = tokenizer  # 根据实际情况选择合适的分词器
        data = self.load_data(data_path)
        self.__prepare__(data)

    def load_data(self, data_path):
        # 从JSON文件中加载数据
        with open(data_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __prepare__(self, data):
        for idx in range(len(data)):
            sample = data[idx]
            text = sample['text']
            labels = []
            for event in sample['event_list']:
                event_type = event['event_type']
                trigger_text = event['trigger']['text']
                trigger_offset = event['trigger']['offset']
                arguments = [
                    (argument['role'], argument['text'], argument['offset'])
                    for argument in event['arguments']
                ]
                labels.append((event_type, trigger_text,
                              trigger_offset, arguments))

            # 将文本转换为标记的ID
            input_tokens = self.tokenizer.tokenize(text)

            # 对齐标签到分词后的文本
            for event_type, trigger_text, trigger_offset, arguments in labels:
                # 对齐标签到分词后的文本
                prompt_position = get_token_index(
                    text, input_tokens, trigger_offset[0])
                prompt_text = f'{trigger_text}，位置{prompt_position}'
                prompt_tokens = self.tokenizer.tokenize(prompt_text)
                # 将Prompt和输入文本的分词结果合并
                tokens = [self.tokenizer.cls_token] + prompt_tokens + \
                    [self.tokenizer.sep_token] + input_tokens
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                aligned_labels = align_labels(text, arguments, input_tokens)
                aligned_labels = ['O'] * \
                    (len(tokens)-len(input_tokens))+aligned_labels
                label_ids = [label_to_index[label] for label in aligned_labels]
                # get attention mask
                attention_mask = create_attention_mask(self.tokenizer, tokens)

                self.data.append((input_ids, label_ids, attention_mask, text))

    def __getitem__(self, idx):
        input_ids, label_ids, attention_mask, text = self.data[idx]
        input_ids = torch.tensor(input_ids).squeeze()
        label_ids = torch.tensor(label_ids).squeeze()
        attention_mask = torch.tensor(attention_mask).squeeze()
        return input_ids, label_ids, attention_mask, text

    def pad_collate_fn(self, batch):
        """Collate function for DataLoader.

        Args:
            batch (list): List of batch data.

        Returns:
            _type_: tuple
            _describe_: Batch data.
        """
        from torch.nn.utils.rnn import pad_sequence
        input_ids, label_ids, attention_mask, text = zip(
            *batch)
        input_ids = pad_sequence(input_ids, batch_first=True)
        label_ids = pad_sequence(label_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        return input_ids, label_ids, attention_mask,  text
