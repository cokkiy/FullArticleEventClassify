# 说明

全文事件抽取和论元抽取及分类。使用了分步模型和事件提示词。首先抽取事件触发词，然后利用事件触发词和位置作为提示符，再进行论元抽取和分类。

## 必需条件

首先安装`transformers`和`pytorch-crf`。

```shell
pip install transformers
pip install pytorch-crf
```

## 训练

运行下面命令开始训练：

```shell
python -u train.py 
```

可选参数有：

```shell
    --path: 训练后模型的保存路径，默认为../result/models，如果目录不存在，会自动创建
    --model_name: 要保持的模型名称，默认为当前时间戳
    --train_file: 训练集文件名，默认为./data/FNDEE_train1.json
    --batch_size: 批大小，默认为4
    --bert_model: Bert预训练模型名称，默认为../models/xlm-roberta-base
    --num_epochs: 训练轮次，默认为20

```

> 说明：
>
>+ 模型名称说明：默认为1687960343格式的时间戳，最后文件名为1687960343-args.pt和1687960343-event.pt，如果输入mymodal,则会训练完毕后自动保存mymodal-event.pt和mymodal-args.pt两个模型文件。其中event表示事件触发词模型，args表示论元抽取和分类模型
>+ Bert预训练模型名称说明：默认从文件夹../models/中加载xlm-roberta-base模型，如果只输入模型名称，则会从huggingface下载模型并加载，如果要加载下载好的模型，则指定路径和名称

示例:

```shell
python -u train.py  --path ../result/models --model_name test -- batch_size 8 --num_epochs 30
```

## 评价

运行下面命令用测试集评价训练后的模型:

```shell
python -u eval.py ../result/models/mymodal
```

> 使用训练集评价训练的模型`../result/models/mymodal`

TODO: 尚未做完

## 预测

TODO: 尚未做完

## 结构

项目由数据集、模型和训练、评级及预测代码组成

### 数据集

数据集由`event_dataset.py`和`arguments_dataset.py`组成。`event_dataset.py`定义了事件抽取和分类数据集，`arguments_dataset.py`定义了论元抽取和分类数据集，在生成论元数据时，前面加入了由事件和位置组成的提示词，以便模型识别对应事件的论元而不是全部论元。

### 模型

1. 事件抽取和分类模型
   `EventExtractorClassifer`定义了一个全连接层模型，用来事件抽取和分类。

2. 论元抽取和分类模型
   `BertBiLSTMCRF`定义了一个全连接层模型，然后用CRF用来论元最有解计算。

### 训练、评级及预测代码

`train.py`定义了训练代码。

`eval.py`定义了评价代码(尚未全部完成)。  

`predict.py`定义了预测代码(尚未完成)。  
