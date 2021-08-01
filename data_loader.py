import torch
import random
import codecs


def data_loader(eneity_dict, rel_dict):
    # entity_id,rel_id初始化
    with open("./data/FB15k-237/entity2id.txt", "r", encoding='utf8') as loader:
        for line in loader.readlines():
            e, idx = line.strip().split()
            if e not in eneity_dict.keys():
                eneity_dict[e] = int(idx)

    with open("./data/FB15k-237/relation2id.txt", "r", encoding='utf8') as loader:
        for line in loader.readlines():
            rel, idx = line.strip().split()
            if rel not in rel_dict.keys():
                rel_dict[rel] = int(idx)


def train_data_process(train_data, eneity_dict, rel_dict):
    with open("./data/FB15k-237/train.txt", "r", encoding='utf8') as loader:
        for line in loader.readlines():
            h, r, t = line.strip().split()
            h, t, r = eneity_dict[h], eneity_dict[t], rel_dict[r]
            # 生成破损三元组
            if (random.random() > 0.5):
                h_2 = random.randint(0, len(eneity_dict) - 1)
                while (h_2 == h):
                    h_2 = random.randint(0, len(eneity_dict) - 1)
                train_data.append((h, t, r, h_2, t, r))
            else:
                t_2 = random.randint(0, len(eneity_dict) - 1)
                while (t_2 == t):
                    t_2 = random.randint(0, len(eneity_dict) - 1)
                train_data.append((h, t, r, h, t_2, r))
    # 转为LongTensor类型
    train_data_tensor = torch.LongTensor(train_data).cuda()
    return train_data_tensor


def test_data_process(eneity_dict, rel_dict):
    test_data = []
    label = []
    with open("./data/FB15k-237/test.txt", 'r') as loader:
        for line in loader.readlines():
            h, r, t = line.strip().split()
            h, t, r = eneity_dict[h], eneity_dict[t], rel_dict[r]
            test_data.append((h, r))
            label.append(t)
    # 转为LongTensor类型
    test_data_tensor = torch.LongTensor(test_data).cuda()
    return test_data_tensor, label
