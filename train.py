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

# dataloader
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data["SalePrice"], dtype=torch.float).view(-1, 1)

# Train
loss = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

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
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()

    for _ in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, x, y):
    '''
    return x_train, y_train, x_valid, y_valid
    '''
    assert k > 1
    fold_size = len(x) // k
    x_train, y_train = None, None
    for j in range(k):
        index = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[index, :], y[index]
        if i == j:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return x_train, y_train, x_valid, y_valid

def k_fold(k, i, x_train, y_train, batch_size, learning_rate, weight_decay, num_epochs):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net(train_data.shape[1])
        train_ls, valid_ls = train(net=net, *data, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, num_epochs=num_epochs)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        
    return train_loss, valid_loss







    






    


