import torch
import numpy as np


def hits_and_ranks(model, test_data_batch):
    ranks = []
    hits_one = []
    hits_three = []
    hits_ten = []
    for (test_data,label) in test_data_batch:
        h,r=test_data[:,0],test_data[:,1]
        score = model.forecast(h, r)  # batch_size,eneity

        max_values, argsort = torch.sort(score, 1, descending=False)
        argsort = argsort.cpu().numpy()  # batch_size,eneity

        for i in range(label.shape[0]):
            rank = np.where(argsort[i] == label[i].item())[0][0]
            ranks.append(rank + 1)

            # @1命中率
            if rank + 1 <= 1:
                hits_one.append(1.0)
            else:
                hits_one.append(0.0)

            # @3命中率
            if rank + 1 <= 3:
                hits_three.append(1.0)
            else:
                hits_three.append(0.0)

            # @10命中率
            if rank + 1 <= 10:
                hits_ten.append(1.0)
            else:
                hits_ten.append(0.0)

    print("@1 命中率：{}".format(np.mean(hits_one)))

    print("@3 命中率：{}".format(np.mean(hits_three)))

    print("@10 命中率：{}".format(np.mean(hits_ten)))

    print("MR:{}".format(np.mean(ranks)))

    print("MRR:{}".format(np.mean(1.0 / np.array(ranks))))
