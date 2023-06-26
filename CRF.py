import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags

        # 定义转移矩阵，矩阵大小为 num_tags x num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # 定义开始标记和结束标记的转移矩阵，大小为1 x num_tags
        self.start_transitions = nn.Parameter(torch.randn(1, num_tags))
        self.end_transitions = nn.Parameter(torch.randn(1, num_tags))

    def forward(self, feats):
        # feats: (seq_len, batch_size, num_tags)
        seq_length = feats.shape[0]
        batch_size = feats.shape[1]

        # 计算分数矩阵
        scores = feats.transpose(0, 1)  # (batch_size, seq_len, num_tags)
        # (batch_size, seq_len, num_tags, num_tags)
        scores = scores.unsqueeze(2).repeat(1, 1, self.num_tags, 1)
        scores = scores + self.transitions.unsqueeze(0).unsqueeze(0)
        scores[:, 0, :, :] = scores[:, 0, :, :] + \
            self.start_transitions.unsqueeze(0).unsqueeze(0)
        scores[:, -1, :, :] = scores[:, -1, :, :] + \
            self.end_transitions.unsqueeze(0).unsqueeze(0)

        # 定义标记序列
        tags = torch.arange(self.num_tags).unsqueeze(0).unsqueeze(
            0).repeat(batch_size, seq_length, 1).to(feats.device)

        # 动态规划计算最优路径
        for i in range(1, seq_length):
            # [batch_size, num_tags, num_tags]
            transition_scores = scores[:, i, :, :] + \
                tags.unsqueeze(-1) + tags.unsqueeze(-2)
            # [batch_size, num_tags]
            max_scores, max_score_tags = torch.max(transition_scores, dim=-1)
            # [batch_size, num_tags]
            scores[:, i, :, :] = max_scores + feats[i, :, :].unsqueeze(-1)
            # [batch_size, num_tags]
            tags[:, i, :] = max_score_tags

        # 回溯最优路径
        path_scores = scores[:, -1, :, :]
        path_scores += self.end_transitions.unsqueeze(0).unsqueeze(0)
        best_tags = torch.argmax(path_scores, dim=-1)

        best_paths = [best_tags[:, -1]]
        for i in range(seq_length - 2, -1, -1):
            best_tags = tags[:, i, :]
            best_tag_ids = best_tags.gather(-1,
                                            best_paths[-1].unsqueeze(-1)).squeeze()
            best_paths.append(best_tag_ids)
        best_paths.reverse()

        return best_paths
