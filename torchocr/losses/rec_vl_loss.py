import torch
from torch import nn


class VLLoss(nn.Module):
    def __init__(self, mode='LF_1', weight_res=0.5, weight_mas=0.5, **kwargs):
        super(VLLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        assert mode in ['LF_1', 'LF_2', 'LA']
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    def flatten_label(self, target):
        label_flatten = []
        label_length = []
        for i in range(0, target.shape[0]):
            cur_label = target[i].tolist()
            label_flatten += cur_label[:cur_label.index(0) + 1]
            label_length.append(cur_label.index(0) + 1)
        label_flatten = torch.tensor(label_flatten, dtype=torch.int64)
        label_length = torch.tensor(label_length, dtype=torch.int32)
        return (label_flatten, label_length)

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def forward(self, predicts, batch):
        text_pre = predicts[0]
        target = batch[1].astype('int64')
        label_flatten, length = self.flatten_label(target)
        text_pre = self._flatten(text_pre, length)
        if self.mode == 'LF_1':
            loss = self.loss_func(text_pre, label_flatten)
        else:
            text_rem = predicts[1]
            text_mas = predicts[2]
            target_res = batch[2].astype('int64')
            target_sub = batch[3].astype('int64')
            label_flatten_res, length_res = self.flatten_label(target_res)
            label_flatten_sub, length_sub = self.flatten_label(target_sub)
            text_rem = self._flatten(text_rem, length_res)
            text_mas = self._flatten(text_mas, length_sub)
            loss_ori = self.loss_func(text_pre, label_flatten)
            loss_res = self.loss_func(text_rem, label_flatten_res)
            loss_mas = self.loss_func(text_mas, label_flatten_sub)
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        return {'loss': loss}
