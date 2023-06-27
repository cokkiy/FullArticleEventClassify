from typing import List


def align_labels(text: str, labels: list[(str, str, (int, int))], tokens: list[str]):
    """Aligning labels to the tokenized text.

    Args:
        text (str): Original text.
        labels (List[(str,str,(int,int))]): List of label which has type, text and offset in original text.
        tokens (list[str]): Tokenized original text.

    Returns:
        _type_: List[str]
        _describe_: Aligned labels to tokenized text.
    """
    # 对齐标签到分词后的文本
    aligned_labels = ['O'] * len(tokens)
    start = 0
    label_idx = 0
    finshed_label_len = 0

    if not labels:
        return aligned_labels

    type, label_text, offset = labels[label_idx]
    for i, token in enumerate(tokens):
        if label_idx >= len(labels):
            break
        if text.find(token, start) != -1:
            if start >= offset[0] and start <= offset[1]:
                finshed_label_len += len(token)
                if start+len(token) <= offset[1]:
                    aligned_labels[i] = type
                if finshed_label_len >= len(label_text):
                    label_idx += 1
                    finshed_label_len = 0
                    if label_idx < len(labels):
                        type, label_text, offset = labels[label_idx]
            start = text.find(token, start) + len(token)
    return aligned_labels


def get_token_index(text: str, tokens: list[str], start_in_text: int):
    """Get index in tokens.

    Args:
        text (str): Original text.
        tokens (list[str]): Tokenized original text.
        start_in_text (int): Start position in text.

    Returns:
        _type_: int
        _describe_: Index in tokens.
    """
    start = 0
    for i, token in enumerate(tokens):
        if text.find(token, start) != -1:
            if start == start_in_text:
                return i
            start = text.find(token, start) + len(token)
    return -1


def create_attention_mask(tokenizer, tokens: list):
    attention_mask = [1 for _ in range(len(tokens))]
    for i, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens:
            attention_mask[i] = 0
    return attention_mask
