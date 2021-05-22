import torch
import pandas as pd
from torch import optim, nn
import torch.utils.data as Data

data_path = "data/house-prices-advanced-regression-techniques"
train_path = data_path + "/train.csv"
test_path = data_path + "/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# (1460, 81)
# print(train_data.shape)
# (1459, 80)
# print(test_data.shape)

# data pre process
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
# (2919, 331)
# print(all_features.shape)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data["SalePrice"], dtype=torch.float).view(-1, 1)

# Train
loss = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_net(feature_num, device="cpu"):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net.to(device)

# loss function
def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    
    return rmse.item()

 
def train(net, train_features, train_labels, test_features, test_labels, batch_size, learning_rate, weight_decay, num_epochs):

    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, test_features)
    train_iter = Data.DataLoader(train_features, test_features, batch_size, shuffle=True)

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()

    for epoch in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls




    


