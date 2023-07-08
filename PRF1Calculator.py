from typing import Any

import torch


class PRF1Calculator:
    '''PRF1计算器
    '''

    def __init__(self, predict=0, correct=0, real=0) -> None:
        """PRF1计算器
        Args:
            predict (int, optional): 预测的事件数. Defaults to 0.
            correct (int, optional): 正确的事件数. Defaults to 0.
            real (int, optional): 实际的事件数. Defaults to 0.
        """
        self.predict = predict
        self.correct = correct
        self.real = real

    def __repr__(self) -> str:
        return f'PRF1Calc(predict={self.predict}, correct={self.correct}, total={self.total})'

    def update(self, preds, label_ids):
        """从预测结果和标签中更新PRF1

        Args:
            preds (_type_): 预测的结果
            label_ids (_type_): 实际的标签
        """
        self.predict += torch.sum(preds != 0).item()
        self.correct += torch.sum((preds != 0) &
                                  (preds == label_ids.view(-1))).item()
        self.real += torch.sum(label_ids.view(-1) != 0).item()

    def __getitem__(self, key: Any) -> Any:
        if key == 'P':
            return 'NAN' if self.predict == 0 else self.correct/self.predict
        elif key == 'R':
            return self.correct/self.real
        elif key == 'F1':
            return 2*self.correct/(self.predict+self.real)
        else:
            raise KeyError(f'Key {key} is not supported')

    @property
    def P(self):
        return 'NAN' if self.predict == 0 else round(self.correct/self.predict, 4)

    @property
    def R(self):
        return self.correct/self.real

    @property
    def F1(self):
        return 2*self.correct/(self.predict+self.real)
