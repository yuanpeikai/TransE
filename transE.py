import torch
import torch.nn as nn
import lib


class transE(nn.Module):

    def __init__(self, entity_len, rel_len):
        super(transE, self).__init__()
        self.e_embeding = nn.Embedding(entity_len, lib.embedding_dim)
        self.rel_embeding = nn.Embedding(rel_len, lib.embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.bn = torch.nn.BatchNorm1d(50)
        nn.init.xavier_uniform_(self.e_embeding.weight.data)
        nn.init.xavier_uniform_(self.rel_embeding.weight.data)

    def get_score(self, h, t, r):
        score = torch.norm(h + r - t, p=1, dim=-1)
        return score

    def forward(self, h, t, r, h_2, t_2, r_2):
        # 完好三元组
        h = self.dropout(self.bn(self.e_embeding(h)))
        t = self.dropout(self.bn(self.e_embeding(t)))
        r = self.dropout(self.bn(self.rel_embeding(r)))
        # 破损三元组
        h_2 = self.dropout(self.bn(self.e_embeding(h_2)))
        t_2 = self.dropout(self.bn(self.e_embeding(t_2)))
        r_2 = self.dropout(self.bn(self.rel_embeding(r_2)))
        # 此处的分数表示距离，越低越好
        score_good = self.get_score(h, t, r)
        score_bad = self.get_score(h_2, t_2, r_2)
        return score_good, score_bad

    def forecast(self, h, r):
        t = self.e_embeding.weight.data  # eneity,50
        t = t.unsqueeze(dim=0)  # 1,eneity,50
        t = t.repeat(h.shape[0], 1, 1)  # batch_size,eneity,50

        h = self.e_embeding(h)  # batch_size,50
        h = h.unsqueeze(dim=1)  # batch_size,1,50
        h = h.repeat(1, t.shape[1], 1)  # batch_size,eneity,50
        r = self.rel_embeding(r)
        r = r.unsqueeze(dim=1)  # batch_size,1,50
        r = r.repeat(1, t.shape[1], 1)  # batch_size,eneity,50
        score = self.get_score(h, self.e_embeding.weight.data, r)  # batch_size,eneity
        return score
