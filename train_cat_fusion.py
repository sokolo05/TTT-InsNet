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

        # 输入线性层，将输入维度映射到 Transformer 的维度
        self.input_linear = nn.Linear(input_dim, d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))  # 假设最大序列长度为 100

    def forward(self, x):
        # 调整输入维度
        x = self.input_linear(x)

        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer 编码器的输入需要是 (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # 调整输出维度为 (batch_size, seq_len, d_model)
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
        
        # 转换 x 的形状以应用模块
        batch_size, timesteps, C, H, W = x.size()
        # print(f'batch_size = {batch_size}, timesteps = {timesteps}, H = {H}, W = {W}')
        x_reshaped = x.contiguous().view(batch_size * timesteps, C, H, W)
        y = self.module(x_reshaped)  # 通过 CNN 层
        # 恢复原始形状
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
        
        x = x.squeeze(2)  # 去掉 C
        batch_size, timesteps, H, W = x.size()
        x_reshaped = x.contiguous().view(batch_size * timesteps, H, W)
        y = self.module(x_reshaped)  # 通过 CNN 层
        output = y.view(batch_size, timesteps, y.size(1))
        
        return output

class TTB_Insnet(nn.Module):
    def __init__(self, timesteps):
        super(TTB_Insnet, self).__init__()
        self.timesteps = timesteps
        # 定义特征提取模块
        self.twins_svt = TwinsSVT(input_dim=5, embed_dim=128, num_heads=8, num_layers=4, output_dim=320, window_size=(5, 1), sr_ratio=1)
        self.insnet_model = Insnet_model()
        # TimeDistributed 模块, 用于将 VisionTransformer 和 Insnet_model 的合并结果应用于每个时间步
        self.time_distributed_image = TimeDistributed_image(self.insnet_model)
        self.time_distributed_sequence = TimeDistributed_sequence(self.twins_svt)

        # 可学习权重参数
        self.weight_insnet = nn.Parameter(torch.tensor(0.5))
        self.weight_svt = nn.Parameter(torch.tensor(0.5))

        # 定义 Transformer 模块
        self.transformer = TransformerModel(input_dim=320, d_model=256, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)

        # 定义双向 Bi-GRU
        self.bigru1 = nn.GRU(input_size=320, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        
        # 全连接层，用于最终的输出
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
        x = x.view(batch_size, self.timesteps, C, H, W)  # 调整为 (batch, timesteps, C, H, W)
        x_local = self.time_distributed_sequence(x)  # 输出形状为 (batch_size, timesteps, embed_dim)
        x_global = self.time_distributed_image(x)  # 输出形状为 (batch_size, timesteps, embed_dim)
        # x = torch.cat((x_local, x_global), dim=2)  # 合并局部和全局特征
        
        weights = torch.cat([self.weight_insnet.unsqueeze(0), self.weight_svt.unsqueeze(0)], dim=0)
        normalized_weights = F.softmax(weights, dim=0)
        x = normalized_weights[0] * x_local + normalized_weights[1] * x_global

        # 通过 Transformer 模块
        x = self.transformer(x)
        # print(f'x.shape after Transformer: {x.shape}')

        # # 通过双向 Bi-GRU
        # x, _ = self.bigru1(x)
        # x, _ = self.bigru2(x)
        
        # 全连接层
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

    # 初始化模型
    model = TTB_Insnet(timesteps)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model = nn.DataParallel(model).to(DEVICE)
    
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")  # 示例损失函数
    print(f'loss_fn')
    # loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.01) 
    # 定义 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=best_model_path)
    # 训练和验证循环的参数
    train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        print(f"------------------------Epoch {epoch + 1}/{num_epochs} Learning Rate: {optimizer.param_groups[0]['lr']}-------------------------")
        train_loss, train_metrics, AUC = train_fn(model, DEVICE, timesteps, train_loader, optimizer, loss_fn, train_losses, train_accuracy)
        print(f"Train Loss: {train_loss:.4f}, AUC = {AUC:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1_score']:.4f}, TP: {train_metrics['TP']:.4f}, FP: {train_metrics['FP']:.4f}, TN: {train_metrics['TN']:.4f}, FN: {train_metrics['FN']:.4f}")
        valid_loss, valid_metrics, AUC = valid_fn(model, DEVICE, timesteps, valid_loader, loss_fn, val_losses, val_accuracy)
        print(f"Valid Loss: {valid_loss:.4f}, AUC = {AUC:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}, Precision: {valid_metrics['precision']:.4f}, Recall: {valid_metrics['recall']:.4f}, F1: {valid_metrics['f1_score']:.4f}, TP: {valid_metrics['TP']:.4f}, FP: {valid_metrics['FP']:.4f}, TN: {valid_metrics['TN']:.4f}, FN: {valid_metrics['FN']:.4f}")
        early_stopping.save_checkpoint(valid_metrics['f1_score'], model)
        # 更新学习率
        scheduler.step(valid_metrics['f1_score'])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# train_data_path = '/home/laicx/03.study/04.Insnet/01.insnet_gao/02.filter_data/01.HG002_PB_70x_RG_HP10XtrioRTG/01.train_data/'
# valid_data_path = '/home/laicx/03.study/04.Insnet/01.insnet_gao/02.filter_data/01.HG002_PB_70x_RG_HP10XtrioRTG/03.valid_data/'

# train_filelist = [train_data_path + f for f in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, f)) and 'npy' in f]
# train_file_list = [train_file for train_file in train_filelist if "index" not in train_file]

# valid_filelist = [valid_data_path + f for f in os.listdir(valid_data_path) if os.path.isfile(os.path.join(valid_data_path, f)) and 'npy' in f]
# valid_file_list = [valid_file for valid_file in valid_filelist if "index" not in valid_file]

# print(f'length of train_filelist = {len(train_filelist)}, length of train_file_list = {len(train_file_list)}')
# print(f'length of valid_filelist = {len(valid_filelist)}, length of valid_file_list = {len(valid_file_list)}')

# train_data_path_1 = '/home/laicx/03.study/04.Insnet/01.insnet_gao/02.filter_data/02.HG002_GRCh37_ONT-UL_UCSC_20200508.phased/01.train_data/'
# valid_data_path_1 = '/home/laicx/03.study/04.Insnet/01.insnet_gao/02.filter_data/02.HG002_GRCh37_ONT-UL_UCSC_20200508.phased/03.valid_data/'

# train_filelist_1 = [train_data_path_1 + f for f in os.listdir(train_data_path_1) if os.path.isfile(os.path.join(train_data_path_1, f)) and 'npy' in f]
# train_file_list_1 = [train_file for train_file in train_filelist_1 if "index" not in train_file]

# valid_filelist_1 = [valid_data_path_1 + f for f in os.listdir(valid_data_path_1) if os.path.isfile(os.path.join(valid_data_path_1, f)) and 'npy' in f]
# valid_file_list_1 = [valid_file for valid_file in valid_filelist_1 if "index" not in valid_file]

# print(f'length of train_filelist = {len(train_filelist)}, length of train_file_list = {len(train_file_list)}')
# print(f'length of valid_filelist = {len(valid_filelist)}, length of valid_file_list = {len(valid_file_list)}')

# print(f'length of train_filelist = {len(train_filelist_1)}, length of train_file_list = {len(train_file_list_1)}')
# print(f'length of valid_filelist = {len(valid_filelist_1)}, length of valid_file_list = {len(valid_file_list_1)}')

# # train_file = train_file_list + train_file_list_1
# # valid_file = valid_file_list + valid_file_list_1

# train_file = train_file_list
# valid_file = valid_file_list

# print(f'length of train_file = {len(train_file)}, length of valid_file = {len(valid_file)}')

# best_model_path = "/home/laicx/03.study/04.Insnet/03.TTT-Insnet/01.modle_save/train_cat_fusion_5.pth"
# num_epochs, timesteps, batch_size = 50, 5, 64

# '''
# Total number of label 1: 20558
# Total number of label 0: 8681009

# Total number of label 1: 1622
# Total number of label 0: 653863

# Total number of label 1: 13117
# Total number of label 0: 4919724
# '''

# # 设置随机种子
# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# train_Insnet(train_file_list, valid_file_list, best_model_path, num_epochs, timesteps, batch_size)