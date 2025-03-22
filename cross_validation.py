
import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import pickle
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
from torch.utils.data import Dataset
from models.LGSyn_datasets import FastTensorDataLoader
from models.LGSyn_utils import save_args, arg_min, conf_inv, calc_stat, save_best_model, find_best_model, random_split_indices
time_str = str(datetime.now().strftime('%y%m%d%H%M'))
from sklearn.preprocessing import StandardScaler
from codes.models.kg import KG
from models.pre import getKgIndexsFromKgTriples, construct_adj, readKGData, readRecData
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

torch.cuda.empty_cache()

# set model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--gpu', type=int, default=0, help='the number of epochs')
parser.add_argument('--hidden', type=int, default=4096, help='the number of epochs')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=50,
                        help='how long to wait after last time validation loss improved')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--train_test_mode', type=int, default=1, help='Judeg train or test.')
parser.add_argument('--suffix', type=str, default='results_kgedc', help="model dir suffix")
args = parser.parse_args()

#files
OUTPUT_DIR = '/output/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
args = parser.parse_args()
out_dir = os.path.join(OUTPUT_DIR, '{}'.format(args.suffix))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
log_file = os.path.join(out_dir, 'cv.log')
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)
save_args(args, os.path.join(out_dir, 'args.json'))
test_loss_file = os.path.join(out_dir, 'test_loss.pkl')
if torch.cuda.is_available() and (args.gpu is not None):
    gpu_id = args.gpu
else:
    gpu_id = None


class FastSynergyDataset(Dataset):
    def __init__(self, drug_feat_file, drug_emb, cell_feat_file, cell_emb, synergy_score_file, use_folds, train=True):
        # 加载嵌入文件
        self.drug_feat2 = np.load(drug_emb)
        self.cell_feat2 = np.load(cell_emb)
        # 加载特征文件
        self.drug_feat1 = self._load_features(drug_feat_file)
        self.cell_feat = self._load_features(cell_feat_file)
        # 初始化样本列表
        self.samples = []
        # 初始化原始样本列表
        self.raw_samples = []
        # 是否是训练集标志
        self.train = train
        with open(synergy_score_file, 'r') as f:
            for line in f:
                # 解析每一行，提取药物1ID、药物2ID、细胞ID、分数和折叠数
                drug1_id, drug2_id, cell_id, score, fold = line.rstrip().split('\t')
                # 检查折叠数是否在使用的折叠集合中
                if int(fold) in use_folds:
                    # 创建样本（药物1、药物2、细胞及其对应特征和分数）
                    sample = [
                        torch.from_numpy(self.drug_feat1[int(drug1_id)]).float(),
                        torch.from_numpy(self.drug_feat2[int(drug1_id)]).float(),  # 修改这里
                        torch.from_numpy(self.drug_feat1[int(drug2_id)]).float(),
                        torch.from_numpy(self.drug_feat2[int(drug2_id)]).float(),  # 修改这里
                        torch.from_numpy(self.cell_feat[int(cell_id)]).float(),
                        torch.from_numpy(self.cell_feat2[int(cell_id)]).float(),  # 修改这里
                        torch.FloatTensor([float(score)]),
                    ]
                    # 将样本添加到样本列表
                    self.samples.append(sample)
                    # 创建原始样本记录（仅包含ID和分数）
                    raw_sample = [int(drug1_id), int(drug2_id), int(cell_id), score]
                    self.raw_samples.append(raw_sample)
                    # 如果是训练集，还要创建药物1和药物2对调的样本
                    if train:
                        sample = [
                            torch.from_numpy(self.drug_feat1[int(drug2_id)]).float(),
                            torch.from_numpy(self.drug_feat2[int(drug2_id)]).float(),
                            torch.from_numpy(self.drug_feat1[int(drug1_id)]).float(),
                            torch.from_numpy(self.drug_feat2[int(drug1_id)]).float(),
                            torch.from_numpy(self.cell_feat[int(cell_id)]).float(),
                            torch.from_numpy(self.cell_feat2[int(cell_id)]).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [int(drug2_id), int(drug1_id), int(cell_id), score]
                        self.raw_samples.append(raw_sample)
    def _load_features(self, feat_file):
        features = {}
        with open(feat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                feat_id = int(parts[0])
                feat_values = np.array([float(x) for x in parts[1:]])
                features[feat_id] = feat_values
        return features
    def __len__(self):
        # 返回样本数量
        return len(self.samples)
    def __getitem__(self, item):
        # 返回指定索引的样本
        return self.samples[item]
    def drug_feat1_len(self):
        # 返回药物特征1的长度
        return len(next(iter(self.drug_feat1.values())))
    def drug_feat2_len(self):
        # 返回药物特征2的长度
        return self.drug_feat2.shape[-1]
    def cell_feat_len(self):
        # 返回细胞特征的长度
        return len(next(iter(self.cell_feat.values())))
    def cell_feat2_len(self):
        # 返回细胞特征2的长度
        return self.cell_feat2.shape[-1]
    def tensor_samples(self, indices=None):
        # 将样本转换为张量格式
        if indices is None:
            indices = list(range(len(self)))
        # 按索引组合各个特征
        d1_f1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d1_f2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d2_f1 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2_f2 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        c2 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        # 返回组合后的张量
        return d1_f1, d1_f2, d2_f1, d2_f2, c, c2, y

##create model
class DNN_con(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, cell_feat_len:int, cell_feat2_len:int, hidden_size: int):
        super(DNN_con, self).__init__()
        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len) # 416
        )
        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len)
        )
        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len),
            nn.Linear(cell_feat_len, cell_feat_len ) #256
        )
        self.cell_network2 = nn.Sequential(
            nn.Linear(cell_feat2_len, cell_feat2_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat2_len ),
            nn.Linear(cell_feat2_len, cell_feat2_len)
        )
        self.fc_network = nn.Sequential(
            nn.BatchNorm1d(2*(drug_feat1_len + drug_feat2_len)+cell_feat_len+cell_feat2_len),
            nn.Linear(2*(drug_feat1_len + drug_feat2_len)+cell_feat_len+cell_feat2_len, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, cell_feat1: torch.Tensor, cell_feat2: torch.Tensor):
        drug1_feat1_vector = self.drug_network1( drug1_feat1 )
        drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        drug2_feat1_vector = self.drug_network1( drug2_feat1 )
        drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        cell_feat_vector = self.cell_network(cell_feat1)
        cell_feat_vector2 = self.cell_network2(cell_feat2)
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector, drug2_feat1_vector, drug2_feat2_vector, cell_feat_vector, cell_feat_vector2], 1)
        out = self.fc_network(feat)
        return out


class DNN_ave(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, cell_feat_len:int, cell_feat2_len:int, hidden_size: int):
        super(DNN_ave, self).__init__()
        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len) # 416
        )
        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, 416) # 原drug_feat2_len，调为416以符合药物局部特征维度
        )
        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len),
            nn.Linear(cell_feat_len, cell_feat_len ) #256
        )
        self.cell_network2 = nn.Sequential(
            nn.Linear(cell_feat2_len, cell_feat2_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat2_len ),
            nn.Linear(cell_feat2_len, 256)#384，调成细胞系局部特征维度
        )
        self.fc_network = nn.Sequential(
            #nn.BatchNorm1d(2*(drug_feat1_len + drug_feat2_len)+cell_feat_len+cell_feat2_len),
            #nn.Linear(2*(drug_feat1_len + drug_feat2_len)+cell_feat_len+cell_feat2_len, hidden_size),
            nn.BatchNorm1d(2 * drug_feat1_len + cell_feat_len),
            nn.Linear( 2 * drug_feat1_len + cell_feat_len, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor,
                drug2_feat2: torch.Tensor, cell_feat1: torch.Tensor, cell_feat2: torch.Tensor):
        # 对药物1特征进行平均
        drug1_feat_avg = (self.drug_network1(drug1_feat1) + self.drug_network2(drug1_feat2)) / 2
        # 对药物2特征进行平均
        drug2_feat_avg = (self.drug_network1(drug2_feat1) + self.drug_network2(drug2_feat2)) / 2
        # 对细胞特征进行平均
        cell_feat_avg = (self.cell_network(cell_feat1) + self.cell_network2(cell_feat2)) / 2
        # 将平均后的药物1、药物2和细胞特征拼接
        feat = torch.cat([drug1_feat_avg, drug2_feat_avg, cell_feat_avg], 1)
        # 通过全连接层得到输出
        out = self.fc_network(feat)
        return out

class DNN_wei(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, cell_feat_len:int, cell_feat2_len:int, hidden_size: int):
        super(DNN_wei, self).__init__()
        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len) # 416
        )
        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, 416) # 原drug_feat2_len，调为416以符合药物局部特征维度
        )
        self.cell_network1 = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len),
            nn.Linear(cell_feat_len, cell_feat_len ) #256
        )
        self.cell_network2 = nn.Sequential(
            nn.Linear(cell_feat2_len, cell_feat2_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat2_len ),
            nn.Linear(cell_feat2_len, 256)#384，调成细胞系局部特征维度
        )
        self.fc_network = nn.Sequential(
            nn.BatchNorm1d(2 * drug_feat1_len + cell_feat_len),
            nn.Linear(2 * drug_feat1_len + cell_feat_len, hidden_size),

            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
        # 初始化加权参数，给局部和全局特征分别赋予权重
        self.local_weight = nn.Parameter(torch.tensor(0.5))  # 初始化局部特征权重
        self.global_weight = nn.Parameter(torch.tensor(0.5))  # 初始化全局特征权重

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor,
                drug2_feat2: torch.Tensor, cell_feat1: torch.Tensor, cell_feat2: torch.Tensor):
        drug1_local = self.drug_network1(drug1_feat1)
        drug1_global = self.drug_network2(drug1_feat2)
        drug2_local = self.drug_network1(drug2_feat1)
        drug2_global = self.drug_network2(drug2_feat2)
        # 使用加权融合
        drug1_fusion = self.local_weight * drug1_local + self.global_weight * drug1_global
        drug2_fusion = self.local_weight * drug2_local + self.global_weight * drug2_global
        # 对细胞特征进行类似的加权融合
        cell_local = self.cell_network1(cell_feat1)
        cell_global = self.cell_network2(cell_feat2)
        cell_fusion = self.local_weight * cell_local + self.global_weight * cell_global
        # 将药物和细胞的融合特征拼接
        combined_features = torch.cat([drug1_fusion, drug2_fusion, cell_fusion], dim=1)
        # 使用全连接层预测（假设你已经定义了 self.fc 来预测最终输出）
        out = self.fc_network(combined_features)
        return out



#useful functions
def create_model(data, hidden_size, gpu_id=None):
    model = DNN_wei(data.drug_feat1_len(), data.drug_feat2_len(), data.cell_feat_len(), data.cell_feat2_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    if gpu_id is not None:
        batch = [x.cuda(gpu_id) for x in batch]
    drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2, y_true = batch
    if train:
        y_pred = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
        yp2 = model(drug2_feats1, drug2_feats2, drug1_feats1, drug1_feats2, cell_feats, cell_feats2)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true)
    return loss

def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def eval_epoch(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, gpu_id, train=False)
            epoch_loss += loss.item()
    return epoch_loss

def train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=False, mdl_dir=None):
    min_loss = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            if sl:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break
    if sl:
        model.load_state_dict(torch.load(find_best_model(mdl_dir)))
    return min_loss

def eval_model(model, optimizer, loss_func, train_data, test_data,
               batch_size, n_epoch, patience, gpu_id, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)
    train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=True, mdl_dir=mdl_dir)
    test_loss = eval_epoch(model, test_loader, loss_func, gpu_id)
    test_loss /= len(test_data)
    return test_loss

SYNERGY_FILE = '/data1/synergy.txt'
data = pd.read_csv(SYNERGY_FILE, sep='\t', header=None)
data.columns = ['drugname1','drugname2','cell_line','synergy','fold']
drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2']))))
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line'])))
cellscount = len(cellslist)
threshold = 30
n_folds = 5
n_delimiter = 60
test_losses = []
test_pccs = []
class_stats = np.zeros((n_folds, 7))
for test_fold in range(5):
    valid_fold = list(range(5))[test_fold - 1]
    train_fold = [x for x in list(range(5)) if x != test_fold and x != valid_fold]
    test_data = data[data['fold'] == test_fold]
    valid_data = data[data['fold'] == valid_fold]
    train_data = data[(data['fold'] != test_fold) & (data['fold'] != valid_fold)]
    print('processing test fold {0} train folds {1} valid folds{2}.'.format(test_fold, train_fold, valid_fold))
    print('test shape{0} train shape{1} valid shape {2}'.format(test_data.shape, train_data.shape, valid_data.shape))
    print("Training ... ")

    ####predictor module
    print('begining predictor......')
    mdl_dir = os.path.join(out_dir, str(test_fold))
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    logging.info("Outer: train folds {}, valid folds {} ,test folds {}".format(train_fold, valid_fold, test_fold))
    logging.info("-" * n_delimiter)

    best_hs, best_lr = args.hidden, args.lr
    logging.info("Best hidden size: {} | Best learning rate: {}".format(best_hs, best_lr))

    DRUG_FEAT_FILE = '/data1/drug_feat.txt'
    embeddings_drug = '/data1/drug_embeddings.npy'
    CELL_FEAT_FILE = '/data1/cell_feat.txt'
    #new_reduced_cell_features.txt
    embeddings_cell = '/data1/cell_embeddings.npy'
    SYNERGY_FILE = '/data1/synergy.txt'


    ##preprocess data
    train_data = FastSynergyDataset(DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell, SYNERGY_FILE,
                                    use_folds=train_fold)
    valid_data = FastSynergyDataset(DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell, SYNERGY_FILE,
                                    use_folds=[valid_fold], train=False)
    test_data = FastSynergyDataset(DRUG_FEAT_FILE, embeddings_drug, CELL_FEAT_FILE, embeddings_cell, SYNERGY_FILE,
                                   use_folds=[test_fold], train=False)


    train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=args.batch, shuffle=True)
    valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data))
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))

    model = create_model(train_data, best_hs, gpu_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    loss_func = nn.MSELoss(reduction='sum')


    ##train
    min_loss = float('inf')
    for epoch in range(1, args.epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        # if epoch % 100 == 0:
        print("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
        logging.info("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= args.patience:
                break


    model.load_state_dict(torch.load(find_best_model(mdl_dir)))



    ##test predict
    with torch.no_grad():
        for test_each in test_loader:
            test_each = [x.cuda(gpu_id) for x in test_each]
            drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2, y_true = test_each
            yp1 = model(drug1_feats1, drug1_feats2, drug2_feats1, drug2_feats2, cell_feats, cell_feats2)
            yp2 = model(drug2_feats1, drug2_feats2, drug1_feats1, drug1_feats2, cell_feats, cell_feats2)
            y_pred = (yp1 + yp2) / 2
            test_loss = loss_func(y_pred, y_true).item()
            y_pred = y_pred.cpu().numpy().flatten()
            y_true = y_true.cpu().numpy().flatten()
            test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
            test_loss /= len(y_true)
            y_pred_binary = [1 if x >= threshold else 0 for x in y_pred]
            y_true_binary = [1 if x >= threshold else 0 for x in y_true]
            roc_score = roc_auc_score(y_true_binary, y_pred)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
            auprc_score = auc(recall, precision)
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary)
            kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

    class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
    class_stats[test_fold] = class_stat
    test_losses.append(test_loss)
    test_pccs.append(test_pcc)
    logging.info("Test loss: {:.4f}".format(test_loss))
    logging.info("Test pcc: {:.4f}".format(test_pcc))
    logging.info("*" * n_delimiter + '\n')

    ##cal the stats in each cell line mse

    # 读取数据并指定列名
    data = pd.read_csv( SYNERGY_FILE, sep='\t', header=None )
    data.columns = ['drugname1', 'drugname2', 'cell_line', 'synergy', 'fold']
    drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2']))))  # 38
    drugscount = len(drugslist)
    cellslist = sorted(list(set(data['cell_line'])))
    cellscount = len(cellslist)
    test_data = data[data['fold'] == test_fold]

    all_data = pd.read_csv(SYNERGY_FILE, sep='\t', header=None)
    all_data.columns = ['drugname1', 'drugname2', 'cell_line', 'synergy', 'fold']
    test_data_orig = all_data[all_data['fold'] == test_fold]
    test_data_orig.loc[:, 'pred'] = y_pred


    test_data_orig.to_csv(out_dir + '/test_data_' + str(test_fold) + '.txt', sep='\t', header=True, index=False)
    cells_stats = np.zeros((cellscount, 9))
    for cellidx in range(cellscount):
        # cellidx = 0
        cellname = cellslist[cellidx]
        each_data = test_data_orig[test_data_orig['cell_line'] == cellname]

        each_true = each_data['synergy'].tolist()
        each_pred = each_data['pred'].tolist()
        each_loss = mean_squared_error(each_true, each_pred)
        each_pcc = np.corrcoef(each_pred, each_true)[0, 1]

        # class
        each_pred_binary = [1 if x >= threshold else 0 for x in each_pred]
        each_true_binary = [1 if x >= threshold else 0 for x in each_true]
        roc_score_each = roc_auc_score(each_true_binary, each_pred)
        precision, recall, _ = precision_recall_curve(each_true_binary, each_pred_binary)
        auprc_score_each = auc(recall, precision)
        accuracy_each = accuracy_score(each_true_binary, each_pred_binary)
        f1_each = f1_score(each_true_binary, each_pred_binary)
        precision_each = precision_score(each_true_binary, each_pred_binary, zero_division=0)
        recall_each = recall_score(each_true_binary, each_pred_binary)
        kappa_each = cohen_kappa_score(each_true_binary, each_pred_binary)
        t = [each_loss, each_pcc, roc_score_each, auprc_score_each, accuracy_each, f1_each, precision_each, recall_each,
             kappa_each]
        cells_stats[cellidx] = t

    pd.DataFrame(cells_stats).to_csv(out_dir + '/test_data_cells_stats_' + str(test_fold) + '.txt', sep='\t', header=None, index=None)



logging.info("CV completed")
with open(test_loss_file, 'wb') as f:
    pickle.dump(test_losses, f)
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

reg_stats = pd.DataFrame()
reg_stats['mse'] = test_losses
reg_stats['pcc'] = test_pccs
# reg_stats.loc['stats'] = [str(round(mu,2)) +'±' str(round(sigma,2)), ]
reg_stats.to_csv(out_dir + '/reg_stats.txt', sep='\t', header=None, index=None)

class_stats = np.concatenate([class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv(out_dir + '/class_stats.txt', sep='\t', header=None, index=None)


