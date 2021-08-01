from data_loader import *
from transE import transE
import torch
import lib
from MydataSet import My_Train_DataSet, My_Test_DataSet
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
from evaluation import hits_and_ranks

# 处理数据集
eneity_dict = {}
rel_dict = {}

# entity_id,rel_id初始化
data_loader(eneity_dict, rel_dict)
# 处理训练集
print("加载训练集")
train_data = []
# 加载处理训练集数据
train_data = train_data_process(train_data, eneity_dict, rel_dict)
# 训练集批处理
train_data_batch = Data.DataLoader(dataset=My_Train_DataSet(train_data), batch_size=128, shuffle=True)
print("训练集加载完成")

# 处理测试集
print("加载测试集")
test_data = []
label = []
test_data, label = test_data_process(eneity_dict, rel_dict)
test_data_batch = Data.DataLoader(dataset=My_Test_DataSet(test_data, label), batch_size=128, shuffle=True)

print("测试集加载完成")
print("开始训练")
# 训练参数
model = transE(len(eneity_dict), len(rel_dict)).cuda()
loss_fn = torch.nn.MarginRankingLoss(margin=1, reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)

# if(lib.load==True):
#     model.load_state_dict(torch.load('./model/model.pkl'))
#     model.eval()
#     with torch.no_grad():
#         hits_and_ranks(model, test_data, len(eneity_dict))

for epoch in range(lib.epochs):
    model.train()
    loss_all = []
    for index, data in enumerate(train_data_batch):
        h, t, r = data[:, 0], data[:, 1], data[:, 2]
        h_2, t_2, r_2 = data[:, 3], data[:, 4], data[:, 5]
        score_good, score_bad = model(h, t, r, h_2, t_2, r_2)
        loss = loss_fn(score_good, score_bad, torch.Tensor([-1]).cuda())
        loss_all.append(loss.cpu().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print("第{}次训练的loss：{}".format(epoch + 1, np.mean(loss_all)))
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), './model/model.pkl')
        model.eval()
        with torch.no_grad():
            hits_and_ranks(model, test_data_batch)

