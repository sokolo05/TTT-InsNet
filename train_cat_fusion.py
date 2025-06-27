import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import warnings
from loss_function import FocalLoss
from model_insnet import Insnet_model
from model_transformer import TwinsSVT
# from model_SVT import TwinsSVT
from until import get_data_loader, train_fn, valid_fn
from until import EarlyStopping
import torch.optim as optim

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model)) 

    def forward(self, x):
        x = self.input_linear(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        # (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)
        return x

class TimeDistributed_image(nn.Module):
    
    def __init__(self, module, batch_first=True):
        super(TimeDistributed_image, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        batch_size, timesteps, C, H, W = x.size()
        # print(f'batch_size = {batch_size}, timesteps = {timesteps}, H = {H}, W = {W}')
        x_reshaped = x.contiguous().view(batch_size * timesteps, C, H, W)
        y = self.module(x_reshaped) 
        output = y.view(batch_size, timesteps, y.size(1))
        # print(f'output ={output.shape}')
        
        return output
    
class TimeDistributed_sequence(nn.Module):
    
    def __init__(self, module, batch_first=True):
        super(TimeDistributed_sequence, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        
        x = x.squeeze(2) 
        batch_size, timesteps, H, W = x.size()
        x_reshaped = x.contiguous().view(batch_size * timesteps, H, W)
        y = self.module(x_reshaped)  
        output = y.view(batch_size, timesteps, y.size(1))
        
        return output

class TTB_Insnet(nn.Module):
    def __init__(self, timesteps):
        super(TTB_Insnet, self).__init__()
        self.timesteps = timesteps
        self.twins_svt = TwinsSVT(input_dim=5, embed_dim=128, num_heads=8, num_layers=4, output_dim=320, window_size=(5, 1), sr_ratio=1)
        self.insnet_model = Insnet_model()
        # TimeDistributed
        self.time_distributed_image = TimeDistributed_image(self.insnet_model)
        self.time_distributed_sequence = TimeDistributed_sequence(self.twins_svt)
        self.weight_insnet = nn.Parameter(torch.tensor(0.5))
        self.weight_svt = nn.Parameter(torch.tensor(0.5))
        self.transformer = TransformerModel(input_dim=320, d_model=256, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)
        self.fc1 = nn.Linear(256, 128)  # 128 = 2 * hidden_size
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)  # 128 = 2 * hidden_size
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.4)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        batches, H, W, C = x.size()
        batch_size = int(batches / self.timesteps)
        x = x.view(batch_size, self.timesteps, C, H, W)  # (batch, timesteps, C, H, W)
        x_local = self.time_distributed_sequence(x)  # (batch_size, timesteps, embed_dim)
        x_global = self.time_distributed_image(x)  # (batch_size, timesteps, embed_dim)
        # x = torch.cat((x_local, x_global), dim=2)
        
        weights = torch.cat([self.weight_insnet.unsqueeze(0), self.weight_svt.unsqueeze(0)], dim=0)
        normalized_weights = F.softmax(weights, dim=0)
        x = normalized_weights[0] * x_local + normalized_weights[1] * x_global

        # Transformer
        x = self.transformer(x)
        # print(f'x.shape after Transformer: {x.shape}')

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.output(x)
 
        return x

def train_Insnet(train_file_list, valid_file_list, best_model_path, num_epochs, timesteps, batch_size):

    train_loader = get_data_loader(train_file_list, timesteps, batch_size=batch_size)
    valid_loader  = get_data_loader(valid_file_list, timesteps, batch_size=batch_size)

    train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []
    model = TTB_Insnet(timesteps)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model = nn.DataParallel(model).to(DEVICE)
    
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean") 
    print(f'loss_fn')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.01) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=best_model_path)
    train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        print(f"------------------------Epoch {epoch + 1}/{num_epochs} Learning Rate: {optimizer.param_groups[0]['lr']}-------------------------")
        train_loss, train_metrics, AUC = train_fn(model, DEVICE, timesteps, train_loader, optimizer, loss_fn, train_losses, train_accuracy)
        print(f"Train Loss: {train_loss:.4f}, AUC = {AUC:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_score']:.4f}, TP: {train_metrics['TP']:.4f}, FP: {train_metrics['FP']:.4f}, TN: {train_metrics['TN']:.4f}, FN: {train_metrics['FN']:.4f}")
        valid_loss, valid_metrics, AUC = valid_fn(model, DEVICE, timesteps, valid_loader, loss_fn, val_losses, val_accuracy)
        print(f"Valid Loss: {valid_loss:.4f}, AUC = {AUC:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}, Precision: {valid_metrics['precision']:.4f}, Recall: {valid_metrics['recall']:.4f}, F1: {valid_metrics['f1_score']:.4f}, TP: {valid_metrics['TP']:.4f}, FP: {valid_metrics['FP']:.4f}, TN: {valid_metrics['TN']:.4f}, FN: {valid_metrics['FN']:.4f}")
        early_stopping.save_checkpoint(valid_metrics['f1_score'], model)
        scheduler.step(valid_metrics['f1_score'])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
