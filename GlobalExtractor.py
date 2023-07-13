# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn


# 评估指标计算类，该类包含了三个方法：get_sample_f1、get_sample_precision
# 和 get_evaluate_fpr，这三个方法分别计算了 F1 score、精确率和召回率，并且
# 都依赖于 y_pred 和 y_true 两个参数。其中，y_pred是模型的预测结果，y_true
# 是实际标签
class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        # 将 y_pred 中大于 0 的数转换为 1
        y_pred = torch.gt(y_pred, 0).float()

        # 计算 F1 score
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        # 将 y_pred 中大于 0 的数转换为 1
        y_pred = torch.gt(y_pred, 0).float()

        # 计算精确率
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        # 初始化变量
        X, Y, Z = 1e-10, 1e-10, 1e-10
        # 将 y_pred 和 y_true 转换为 numpy 数组
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []

        # 遍历 y_pred 中大于 0 的元素，并保存下标信息
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))

        # 遍历 y_true 中大于 0 的元素，并保存下标信息
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        # 计算 TP 与 FP
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        # 如果 Y 或 Z 为 0，返回 0
        if Y == 0 or Z == 0:
            return 0, 0, 0

        # 计算 F1 score、精确率和召回率
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


class RawGlobalExtractor(nn.Module):
    def __init__(
        self, encoder, ent_type_size, argument_type_size, inner_dim, RoPE=True
    ):
        # encoder: RoBERTa-Large 模型作为编码器，也可以是Bart、NEZHA、ChatGLM、Alpaca、Vicuna等开源模型；
        # inner_dim: 64，模型中间层的维度大小
        # ent_type_size: ent_cls_num，事件实体类别数量
        # argument_type_size: arg_cls_num，普通论元类别数量
        # RoPE: 布尔量，代表是否使用位置编码（Relative Positional Encoding）
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.argument_type_size = argument_type_size

        # 中间层维度
        self.inner_dim = inner_dim

        # 隐藏层维度
        self.hidden_size = encoder.config.hidden_size

        # 定义事件实体线性层self.dense，输入维度是RoBERTa-Large模型隐藏层的维度大小，输出维度为
        # self.ent_type_size * self.inner_dim * 2，即实体类别数量乘以中间层的维度大小乘以2，
        # 因为每个实体都有开始和结束两个位置
        self.dense = nn.Linear(
            self.hidden_size, self.ent_type_size * self.inner_dim * 2
        )

        # 定义事件论元线性层self.arg_dense，输入维度是RoBERTa-Large模型隐藏层的维度，输出维度为
        # self.argument_type_size * self.inner_dim * 2，即论元类别数量乘以中间层的维度乘以2，
        # 因为每个论元都有开始和结束两个位置
        self.arg_dense = nn.Linear(
            self.hidden_size, self.argument_type_size * self.inner_dim * 2
        )

        # 将RoPE的值存储到类的成员变量self.RoPE中，RoPE值为True，则使用位置编码，否则不使用
        self.RoPE = RoPE

    # 定义生成正弦曲线位置嵌入的函数。在自然语言处理中，位置嵌入是为了模型能够区分不同位置的词汇，从而更好地理
    # 解句子的含义。函数接收3个参数：batch_size（批量大小），seq_len（序列长度）和output_dim（输出维度）
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        # 首先生成长度为seq_len的一维张量 position_ids，其元素值从0到seq_len-1。然后在最后一维增加一个维度，变成二维张量
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        # 定义一个一维张量 indices，其元素值从0到output_dim/2-1
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)

        # 对indices中的每个元素执行指数运算，每个元素值都变成了 10000^(-2*i/d)，其中 i 是该元素在indices中的下标
        indices = torch.pow(10000, -2 * indices / output_dim)

        # 将 position_ids 和 indices 相乘，得到一个二维张量 embeddings，其每个位置的值都为 sin/cos 函数的输入值
        embeddings = position_ids * indices

        # 将 embeddings 中的每个元素分别计算 sin 函数和 cos 函数的值，并将这两个值按最后一维组成一个新的张量
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # 将 embeddings 张量在第一维重复 batch_size 次，使得每个样本都有一份相同的位置编码
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))

        # 将 embeddings 张量变形成 batch_size × seq_len × output_dim 的三维张量，其中seq_len是序列长度
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))

        # 将 embeddings 张量移动到指定计算设备上，本设备为RTX 4090 GPU 24G
        embeddings = embeddings.to(self.device)
        return embeddings

    # PyTorch模型的forward函数，用于根据输入的input_ids和attention_mask以及token_type_ids生成模型的输出
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将输入tensor的设备信息赋给self.device
        self.device = input_ids.device

        # 通过调用encoder将输入tensor编码为上下文向量，其中input_ids表示输入的词语id序列，attention_mask
        # 表示哪些词语需要被注意力机制考虑，token_type_ids表示不同句子之间的分隔符
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        # 从编码器的输出中取出最后一个隐藏状态，也就是整个句子的表示。last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # 分别获取批次大小和序列长度
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # 通过全连接层将上下文向量投影到一个更高维度的空间中
        outputs = self.dense(last_hidden_state)
        arg_outputs = self.arg_dense(last_hidden_state)

        # 将上步操作的结果按照指定维度拆分成两份，分别作为query和key的输入，并在最后一个维度上进行拼接操作
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., : self.inner_dim], outputs[..., self.inner_dim :]
        arg_outputs = torch.split(arg_outputs, self.inner_dim * 2, dim=-1)
        arg_outputs = torch.stack(arg_outputs, dim=-2)
        qwa, kwa = (
            arg_outputs[..., : self.inner_dim],
            arg_outputs[..., self.inner_dim :],
        )

        # 实现事件相对位置编码（relative position encoding（RPE））。如果self.RoPE为True，
        # 则生成一个正弦和余弦函数的相对位置嵌入，然后将query和key分别乘上余弦和正弦位置向量
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(
                batch_size, seq_len, self.inner_dim
            )
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)

            # 事件相对位置编码
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

            # 论元相对位置编码
            qwa2 = torch.stack([-qwa[..., 1::2], qwa[..., ::2]], -1)
            qwa2 = qwa2.reshape(qwa.shape)
            qwa = qwa * cos_pos + qwa2 * sin_pos
            kwa2 = torch.stack([-kwa[..., 1::2], kwa[..., ::2]], -1)
            kwa2 = kwa2.reshape(kwa.shape)
            kwa = kwa * cos_pos + kwa2 * sin_pos

        # 计算query和key之间的点积得分，其中使用了PyTorch中的einsum函数实现张量乘法和求和
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)
        arg_logits = torch.einsum("bmhd,bnhd->bhmn", qwa, kwa)

        # 生成两个mask矩阵，用于避免对填充位置的单词进行注意力计算。将mask矩阵与得分相乘，并将不需要注意力的位置设为一个极小的值（-1e12）
        pad_mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, self.ent_type_size, seq_len, seq_len)
        )
        arg_pad_mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, self.argument_type_size, seq_len, seq_len)
        )
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        arg_logits = arg_logits * arg_pad_mask - (1 - arg_pad_mask) * 1e12

        # 排除下三角部分的得分，确保不会出现预测实体之前的实体作为后继实体的情况
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        arg_mask = torch.tril(torch.ones_like(arg_logits), -1)
        arg_logits = arg_logits - arg_mask * 1e12

        # 将得分除以内部维度的平方根，缩放得分以避免梯度消失。最终输出得分作为模型的预测结果
        return logits / self.inner_dim**0.5, arg_logits / self.inner_dim**0.5


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding"""

    def __init__(self, output_dim, merge_mode="add", custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum("bn,d->bnd", position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == "add":
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == "mul":
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == "zero":
            return embeddings.to(inputs.device)


class EffiGlobalExtractor(nn.Module):
    def __init__(
        self, encoder, ent_type_size, argument_type_size, inner_dim, RoPE=True
    ):
        # encodr: RoBERTa-Large 模型作为编码器；
        # inner_dim: 64，模型中间层的维度大小
        # ent_type_size: ent_cls_num，事件实体类别数量
        # argument_type_size: arg_cls_num，普通论元类别数量
        # RoPE: 布尔量，代表是否使用位置编码（Relative Positional Encoding）
        super(EffiGlobalExtractor, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.argument_type_size = argument_type_size

        # 中间层维度
        self.inner_dim = inner_dim

        # 隐藏层维度
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        # 以下两行代码是关于PyTorch事件模型的定义，其中包括两个线性层（linear layer）：dense_1和dense_2
        # 每个线性层都由一个全连接层（fully connected layer）组成，可以将输入张量与权重矩阵相乘并加上
        # 偏置向量，从而生成一个输出张量

        # dense_1 是一个线性层，它有 self.hidden_size 个输入特征和 self.inner_dim2个输出特征
        # 这意味着 dense_1 的权重矩阵将有 self.hidden_size 行和 self.inner_dim2 列
        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)

        # dense_2 是一个线性层，它有 self.hidden_size 个输入特征和 self.ent_type_size2 个输出特征
        # 这意味着 dense_2 的权重矩阵将有 self.hidden_size 行和 self.ent_type_size2 列
        self.dense_2 = nn.Linear(
            self.hidden_size, self.ent_type_size * 2
        )  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

        # 以下两行代码是关于PyTorch事件论元模型的定义，其中包括两个线性层（linear layer）：arg_dense_1和arg_dense_2
        self.arg_dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.arg_dense_2 = nn.Linear(
            self.hidden_size, self.argument_type_size * 2
        )  # 原版的dense2是(inner_dim * 2, argument_type_size * 2)

    def sequence_masking(self, x, mask, value="-inf", axis=None):
        if mask is None:
            return x
        else:
            if value == "-inf":
                value = -1e12
            elif value == "inf":
                value = 1e12
            assert axis > 0, "axis must be greater than 0"
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, "-inf", logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, "-inf", logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 将输入tensor的设备信息赋给self.device
        self.device = input_ids.device

        # 通过调用encoder将输入tensor编码为上下文向量，其中input_ids表示输入的词语id序列，attention_mask
        # 表示哪些词语需要被注意力机制考虑，token_type_ids表示不同句子之间的分隔符
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        # 从编码器的输出中取出最后一个隐藏状态，也就是整个句子的表示。last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs.last_hidden_state

        # 分别获取批次大小和序列长度
        outputs = self.dense_1(last_hidden_state)
        arg_outputs = self.arg_dense_1(last_hidden_state)

        # 将上步操作的结果按照指定维度拆分成两份，分别作为query和key的输入，并在最后一个维度上进行拼接操作
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2
        qwa, kwa = arg_outputs[..., ::2], arg_outputs[..., 1::2]  # 从0,1开始间隔为2

        # 实现事件相对位置编码（relative position encoding（RPE））。如果self.RoPE为True，
        # 则生成一个正弦和余弦函数的相对位置嵌入，然后将query和key分别乘上余弦和正弦位置向量
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, "zero")(outputs)
            arg_pos = SinusoidalPositionEmbedding(self.inner_dim, "zero")(arg_outputs)

            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            arg_cos_pos = arg_pos[..., 1::2].repeat_interleave(2, dim=-1)
            arg_sin_pos = arg_pos[..., ::2].repeat_interleave(2, dim=-1)

            # 事件相对位置编码
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

            # 论元相对位置编码
            qwa2 = torch.stack([-qwa[..., 1::2], qwa[..., ::2]], 3)
            qwa2 = torch.reshape(qwa2, qwa.shape)
            qwa = qwa * arg_cos_pos + qwa2 * arg_sin_pos
            kwa2 = torch.stack([-kwa[..., 1::2], kwa[..., ::2]], 3)
            kwa2 = torch.reshape(kwa2, kwa.shape)
            kwa = kwa * arg_cos_pos + kwa2 * arg_sin_pos

        # logits是通过矩阵乘法计算出的注意力分数，为了提取事件而计算的
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.inner_dim**0.5

        # bias是一个可学习的偏置项，用来调整注意力分数
        bias = torch.einsum("bnh->bhn", self.dense_2(last_hidden_state)) / 2

        # 将bias加到logits上，增加一个维度以匹配输入形状
        logits = (
            logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        )  # logits[:, None] 增加一个维度

        # 对logits应用一个下三角掩码，确保只有当前和之前的单词能相互作用
        logits = self.add_mask_tril(logits, mask=attention_mask)

        # argument_logits是与logits类似的注意力分数，为了提取事件论元而计算的
        argument_logits = torch.einsum("bmd,bnd->bmn", qwa, kwa) / self.inner_dim**0.5

        # argument_logits也需要一个可学习的偏置项
        bias = torch.einsum("bnh->bhn", self.arg_dense_2(last_hidden_state)) / 2

        # 将bias加到argument_logits上，增加一个维度以匹配输入形状
        argument_logits = (
            argument_logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        )  # logits[:, None] 增加一个维度

        # 对argument_logits应用一个下三角掩码，确保只有当前和之前的单词能相互作用
        argument_logits = self.add_mask_tril(argument_logits, mask=attention_mask)

        return logits, argument_logits
