import pandas as pd
import os
import numpy as np
import sklearn.model_selection as sm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



tr_path = '../machineLearning/base/pytorch/hw1/covid.train.csv'  # path to training data
tt_path = '../machineLearning/base/pytorch/hw1/covid.test.csv'   # path to testing data

class COVIDDataset(Dataset):
    def __init__(self ,path, model='train', target_only=False, rate=0.8):
        super().__init__()
        self.model = model
        data = pd.read_csv(path)
        data = data.drop(['id'], axis=1)
        if target_only:
            feats = list(range(93))
        else:
            cols = list(data.columns)  #拿到特征列名称
            data_corr = data.iloc[:, 40:].corr()  #计算各列与目标值的相关性
            target_col = data_corr['tested_positive.2']
            features = target_col[target_col > 0.8]
            features_clos =  features.index.tolist()
            feats = [cols.index(col) for col in features_clos]  #获取该特征对应列索引编号，后续就可以用feats + feats_selected作为特征值
        indices ,target= [], []
        if model == 'test':
                data = data.iloc[:,feats]
                self.data = torch.FloatTensor(np.array(data))
        else:
            target = pd.DataFrame(data['tested_positive.2'])
            data = data.iloc[:,feats]
            if model == 'train':
                indices = [i for i in range(len(data)) if i % 10 !=0]
            # elif model == 'dev':
            else:
                indices = [i for i in range(len(data)) if i % 10 ==0]
            self.data = torch.FloatTensor(np.array(data.loc[indices]))
            target = pd.DataFrame(target)
            self.target = torch.FloatTensor(np.array(target.loc[indices]))
        self.dim = self.data.shape[1]
        print('完成读取 COVID19 的 {} 模型数据读取, 样本数: ({})'.format(model, len(self.data)))

    def __getitem__(self, index):
        if(self.model in ['train','dev']):
            return self.data[index], self.target[index]
        else:
            return self.data[index]
    def __len__(self):
        return len(self.data)

def my_Dataload(path,model, batch_size, target_only=False):
    data_set = COVIDDataset(path, model, target_only)
    dataload = DataLoader(data_set,batch_size)
    return dataload

def validation(de_set,model,device):
    ''' 验证模型 '''
    model.eval()  # model 设为评估模式,此模式不更新参数
    total_loss = 0
    for x,y in de_set:
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():
            pred = model(x)
            mes_loss = model.cal_loss(pred, y)
            total_loss += mes_loss.detach().cpu().item()  # detach()复制此张量、cpu()拷贝到cup、item()将张量值转成python值,只有张量一个值时才可以使用、不止一个数值时用numpy(),
    total_loss = total_loss / len(de_set.dataset)              # compute averaged loss
    return total_loss



class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self,pred, target):
        return self.criterion(pred, target)

def train(train_set,valida_set,config,model,device):
    n_epochs = config['n_epochs']  
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    # optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x,y in train_set:
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            pred_y = model(x)
            print('预测值:{}'.format(pred_y))
            mse_loss = model.cal_loss(pred_y,y)
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())
        valida_mse = validation(valida_set,model,device)
        if valida_mse < min_mse:
            min_mse = valida_mse
            print('保存训练模型 (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt = 1

        epoch +=1
        loss_record['dev'].append(valida_mse)
        print('训练中、apoch={}, loss={}'.format(epoch,min_mse))
        if early_stop_cnt > config['early_stop']:
            break
    
    print('完成训练, 训练次数:{},min_mse:{}'.format(epoch,min_mse))
    return min_mse, loss_record

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

def getDevice():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = getDevice()
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
#target_only = False  ## TODO: Using 40 states & 2 tested_positive features
target_only = True   # 使用自己的特征，如果设置成False，用的是全量特征
# TODO: How to tune these hyper-parameters to improve your model's performance? 这里超参数没怎么调，已经最优的了
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.005,                 # learning rate of SGD
        'momentum': 0.5              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}
def plot_learn_curve(loss_record):
    steps = len(loss_record['train'])
    x_1 = range(steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of loss')
    plt.legend()
    plt.show()



def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

train_data = my_Dataload(tr_path,'train',config['batch_size'])
dev_data = my_Dataload(tr_path,'dev',config['batch_size'])
test_data = my_Dataload(tt_path,'test',config['batch_size'],True)

model = NeuralNet(train_data.dataset.dim).to(device)
min_mse, loss_record = train(train_data, dev_data,config,model,device)

plot_learn_curve(loss_record)

validation(dev_data,model,device)
del model
model = NeuralNet(train_data.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dev_data,model,device)