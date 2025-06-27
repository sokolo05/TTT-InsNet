import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm
from torch.nn import functional as F
from loss_function import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomDataset(Dataset):
    def __init__(self, file_list, timesteps):
        self.file_list = file_list
        self.timesteps = timesteps
        self.file_data = self.load_all_data()  # 预加载所有文件数据

    def load_all_data(self):
        all_features = []
        all_labels = []
        for file_index in self.file_list:
            data = np.load(file_index)
            features, labels = data[:, :-1], data[:, -1]
            features = features.reshape(-1, 200, 5, 1)
            labels = labels.reshape(-1, 1)
            all_features.append(features)
            all_labels.append(labels)
        return list(zip(all_features, all_labels))

    def __len__(self):
        total_samples = sum(len(features) for features, _ in self.file_data)
        return total_samples

    def __getitem__(self, index):
        cumulative_samples = 0
        for file_index, (features, labels) in enumerate(self.file_data):
            if cumulative_samples + len(features) > index:
                break
            cumulative_samples += len(features)
        
        file_index_within_file = index - cumulative_samples
        return features[file_index_within_file], labels[file_index_within_file]

def collate_fn(batch, timesteps, batch_size):

    features, labels = zip(*batch)
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    
    return torch.from_numpy(features).float(), torch.from_numpy(labels).float()

# 使用 DataLoader 加载数据
def get_data_loader(file_list, timesteps, batch_size, shuffle=False):
    dataset = CustomDataset(file_list, timesteps)
    data_loader = DataLoader(dataset, batch_size=batch_size * timesteps, shuffle=shuffle, collate_fn=lambda batch: collate_fn(batch, timesteps, batch_size), num_workers=4)
    return data_loader

class EarlyStopping:
    
    def __init__(self, patience=30, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_f1 = -np.inf
        self.counter_f1 = 0
        self.early_stop = False

    def __call__(self, f1, model):
        # 监控 F1 值
        if f1 > self.best_f1 + self.delta:
            self.best_f1 = f1
            self.counter_f1 = 0
        else:
            self.counter_f1 += 1

        # 检查是否触发早停
        if self.counter_f1 >= self.patience:
            self.early_stop = True
            print(f"EarlyStopping triggered after {self.patience} epochs without improvement in F1.")

    def save_checkpoint(self, f1, model):
        # 保存模型
        if f1 > self.best_f1:
            print(f"F1 value increased ({self.best_f1:.6f} --> {f1:.6f}). Saving model ...")
            torch.save(model.state_dict(), self.path)
            self.best_f1 = f1

def calculate_metrics(y_true, y_pred):
    """
    计算 TP, FP, TN, FN, 准确率, 召回率, 精确率, F1 分数
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score
    }

def train_fn(model, device, timesteps, train_loader, optimizer, loss_fn, train_losses, train_accuracy):
    
    model.to(device)
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        features, labels = features.to(device), labels.to(device)
        # print(f'features = {features.shape}, labels = {labels.shape}')
        remain = features.size(0) % (timesteps * 2)
        if remain != 0:
            # padding_size = timesteps * 2 - remain
            # patch_features = np.zeros((padding_size, 200, 5, 1))
            # padding_labels = np.zeros((padding_size, 1))
            # patch_features, padding_labels = torch.as_tensor(patch_features, dtype=torch.float32).to(device), torch.as_tensor(padding_labels, dtype=torch.float32).to(device) 
            # features_padding = torch.cat((features, patch_features), dim=0)
            # labels_padding = torch.cat((labels, padding_labels), dim=0)
            features_padding = features[:features.size(0) - remain]
            labels_padding = labels[:labels.size(0) - remain]
        else:
            features_padding, labels_padding = features, labels
        if len(features_padding) == 0:
                continue  
        optimizer.zero_grad()
        # print(f'features_padding = {features_padding.shape}, labels_padding = {labels_padding.shape}')
        outputs = model(features_padding)  # 输出形状：[batch_size, seq_len, 1]
        # print(f'outputs = {outputs.shape}')
        loss = loss_fn(outputs.view(-1), labels_padding.view(-1)) # 计算损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 计算预测值
        predicted = (torch.sigmoid(outputs)>0.5).float().view(-1, 1)  # 找到每个样本的预测类别
        y_true.extend(labels_padding.cpu().numpy().flatten().astype(int))
        y_pred.extend(predicted.cpu().numpy().flatten().astype(int))

    train_loss = total_loss / len(train_loader)
    train_metrics = calculate_metrics(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_pred)

    return train_loss, train_metrics, AUC

def valid_fn(model, device, timesteps, valid_loader, loss_fn, val_losses, val_accuracy):
    
    model.to(device)
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(tqdm(valid_loader, desc="Validation", unit="batch")):
            features, labels = features.to(device), labels.to(device)
            remain = features.size(0) % (timesteps * 2)
            if remain != 0:
                # padding_size = timesteps * 2 - remain
                # patch_features = np.zeros((padding_size, 200, 5, 1))
                # padding_labels = np.zeros((padding_size, 1))
                # patch_features, padding_labels = torch.as_tensor(patch_features, dtype=torch.float32).to(device), torch.as_tensor(padding_labels, dtype=torch.float32).to(device)
                # features_padding = torch.cat((features, patch_features), dim=0)
                # labels_padding = torch.cat((labels, padding_labels), dim=0)
                features_padding = features[:features.size(0) - remain]
                labels_padding = labels[:labels.size(0) - remain]
            else:
                features_padding, labels_padding = features, labels
            if len(features_padding) == 0:
                continue   
            outputs = model(features_padding)
            loss = loss_fn(outputs.view(-1), labels_padding.view(-1))
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs)>0.5).float().view(-1, 1) 
            y_true.extend(labels_padding.cpu().numpy().flatten().astype(int))
            y_pred.extend(predicted.cpu().numpy().flatten().astype(int))

    valid_loss = total_loss / len(valid_loader)
    valid_metrics = calculate_metrics(y_true, y_pred)
    
    return valid_loss, valid_metrics