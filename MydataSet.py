import torch.utils.data as Data

class My_Train_DataSet(Data.Dataset):
    def __init__(self, train_data):
        super(My_Train_DataSet, self).__init__()
        self.train_data = train_data

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        return self.train_data[idx]

class My_Test_DataSet(Data.Dataset):
    def __init__(self, test_data, label):
        super(My_Test_DataSet, self).__init__()
        self.test_data = test_data
        self.label = label

    def __len__(self):
        return self.test_data.shape[0]

    def __getitem__(self, idx):
        return self.test_data[idx], self.label[idx]
