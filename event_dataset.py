# 如果你的自定义JSON数据集中的文本和标签需要对齐，你可以按照以下步骤修改上述示例代码：
# 1. 加载和预处理数据：加载JSON数据集时，同时记录每个样本中文本和标签的位置信息。
# 2. 创建Dataset类：在Dataset类中，除了返回文本的标记ID和标签索引之外，还返回文本和标签的对齐信息。
# 3. 修改DataLoader：根据对齐信息，在DataLoader中对文本和标签进行对齐。
# 下面是一个简单的示例代码，展示了如何为具有文本和标签对齐需求的自定义JSON数据集创建一个适用于Transformer模型的DataLoader：

import json
import torch
from torch.utils.data import Dataset, DataLoader
from commonfn import align_labels, create_attention_mask

label_to_index = {
    # convert label to index
    "O": 0,
    "Experiment": 1,
    "Manoeuvre": 2,
    "Deploy": 3,
    "Support": 4,
    "Accident": 5,
    "Exhibit": 6,
    "Conflict": 7,
    "Injure": 8,
}


class EventDataset(Dataset):
    def __init__(self, data_path, tokenizer=None):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer  # 根据实际情况选择合适的分词器

    def load_data(self, data_path):
        # 从JSON文件中加载数据
        with open(data_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        labels = []
        for event in sample["event_list"]:
            event_type = event["event_type"]
            trigger_text = event["trigger"]["text"]
            trigger_offset = event["trigger"]["offset"]
            labels.append((event_type, trigger_text, trigger_offset))
        # 将文本转换为标记的ID
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 对齐标签到分词后的文本
        aligned_labels = align_labels(text, labels, tokens)

        # 将标签转换为索引
        label_ids = [label_to_index[label] for label in aligned_labels]
        # get attention mask
        attention_mask = create_attention_mask(self.tokenizer, tokens)
        input_ids = torch.tensor([input_ids]).squeeze()
        label_ids = torch.tensor([label_ids]).squeeze()
        attention_mask = torch.tensor([attention_mask]).squeeze()
        return input_ids, label_ids, attention_mask,  text, aligned_labels
