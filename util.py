import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch.nn.functional as F
import torch.nn.init as init

import os
import glob
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, data_path, labels_path):
        # データを読み込む
        self.data = torch.load(data_path)
        # ラベルを読み込む
        self.labels = torch.load(labels_path)
        
    def __len__(self):
        # データセットの長さを返す
        return len(self.data)
    
    def __getitem__(self, idx):
        # 指定されたインデックスのデータとラベルを返す
        data_item = self.data[idx]
        label_item = self.labels[idx]
        return data_item, label_item
    
# パラメータの初期化を行う関数 : He initialization
def init_weights(m):
    if type(m) == nn.Conv1d:
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        init.constant_(m.bias, 0)
        
# 損失関数 : クロスエントロピー 
def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h

# 乱数固定
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
# 学習用関数
def fit(model, optimizer, num_epochs, train_loader, test_loader, device, history, save_dir):

    base_epochs = len(history)
    
    # 訓練
  
    for epoch in range(base_epochs, num_epochs+base_epochs):
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0
        n_train_acc, n_val_acc = 0, 0

        #訓練フェーズ
        model.train()

        for inputs, labels in tqdm(train_loader, desc=f"Epoch Train {epoch+1}/{num_epochs}"):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size
    
            # GPUヘ転送　 float()でデータ型をfloat32に変換　to(device)でデータをモデルと同じデバイス(GPU or CPU)に移す
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = model(inputs)
            # print(outputs)

            # 損失計算
            loss = loss_fn(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()
            
            predicted = torch.max(outputs,1)[1]
            # print(predicted)

            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size 
            n_train_acc += (predicted == torch.max(labels,1)[1]).sum().item() 
            # print(predicted[0], labels[0])
        
        # モデルの保存
        torch.save(model, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # 検証

        #　予測フェーズ
        model.eval()

        for inputs_test, labels_test in tqdm(test_loader, desc=f"Epoch Valid {epoch+1}/{num_epochs}"):
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.float().to(device)
            labels_test = labels_test.float().to(device)

            # 予測計算
            outputs_test = model(inputs_test)

            # 損失計算
            loss_test = loss_fn(outputs_test, labels_test)
            
            predicted_test = torch.max(outputs_test,1)[1]

            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss +=  loss_test.item() * test_batch_size
            n_val_acc +=  (predicted_test == torch.max(labels_test,1)[1]).sum().item()

        # 精度計算
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        
        # 結果表示
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f} val_acc: {val_acc:.5f}')
        # 記録
        item = np.array([epoch+1, avg_train_loss, avg_val_loss])
        history = np.vstack((history, item))
        
        if epoch+1 == 1:
            f = open(save_dir + '/history.list', 'w')
        else:
            f = open(save_dir + '/history.list', 'a')
        f.write('{:} {:} {:} {:} {:}\n'.format(epoch+1, avg_train_loss, avg_val_loss, train_acc, val_acc))
        f.close()
    
    return history