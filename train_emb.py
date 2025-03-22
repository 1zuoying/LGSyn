from sklearn.metrics import mean_squared_error
import collections
from codes.models.kg import KG
import scipy.sparse as sp
from codes.models import osUtils as ou
import argparse
import random
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    device = torch.device('cuda')  # 将设备设置为 GPU
    print('GPU is available!')
else:
    device = torch.device('cpu')
    print('GPU is not available, using CPU instead.')
def getKgIndexsFromKgTriples(kg_triples):
    kg_indexs = collections.defaultdict(list)
    for h, r, t in kg_triples:
        kg_indexs[str(h)].append([int(t), int(r)]) # 头实体对应的尾实体和关系
    return kg_indexs


# 根据kg邻接列表，得到实体邻接列表和关系邻接列表
def construct_adj(neighbor_sample_size, kg_indexes, entity_num):
    adj_entity = np.zeros([16102, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([16102, neighbor_sample_size], dtype=np.int64)
    for entity in range(16102):
        neighbors = kg_indexes[str(entity)]
        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            continue
        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)),
                                               size=neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)),
                                               size=neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation

def normalize_adj(mx):
    """Row Normalized Sparse Matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def readKGData(path='/data1/kg.txt'):
    print('Read knowledge graph data...')
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in ou.readTriple(path, sep='\t'):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples

def readRecData(path='/data1/synergy.txt', test_ratio=0.2):
    print('Read Drug Combination Synergy Data...')
    drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, i, r, flod in ou.readTriple(path, sep='\t'):
        drug_set1.add(int(d1))
        drug_set2.add(int(d2))
        cell_set.add(int(i))
        triples.append((int(d1), int(d2), int(i), float(r), int(flod)))
    return list(drug_set1), list(drug_set2), list(cell_set), triples


def train():
    now = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))
    hours = time.strftime("%Y-%m-%d-%H_%M", time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--CV', type=int, default=1, help='the number of CV')
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--n_heads', type=int, default=2, help='the number of multi-head')
    parser.add_argument('--n_neighbors', type=int, default=5, help='the number of neighbors to be sampled')

    parser.add_argument('--e_dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--r_dim', type=int, default=16, help='dimension of user and relation embeddings')
    parser.add_argument('--n_iter', type=int, default=4,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')  # OS
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--patience', type=int, default=10,
                        help='how long to wait after last time validation loss improved')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # OS
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--train_test_mode', type=int, default=1, help='Judeg train or test.')

    args = parser.parse_args(['--l2_weight', '1e-4'])

    for cv in range(1):
        drug1, drug2, cells, triples = readRecData()  # 读取药物组合协同数据
        entitys, relations, kgTriples = readKGData()  # 读取知识图谱数据
        kg_indexes = getKgIndexsFromKgTriples(kgTriples)  # 获得三元组
        adj_entity, adj_relation = construct_adj(args.n_neighbors,
                                                                 kg_indexes, len(entitys))


        loss_fcn = nn.MSELoss()
        np.random.seed(23)
        random.shuffle(triples)
        # split = math.ceil(len(triples) / 5)
        triples_DF = pd.DataFrame(triples)


        test_fold = 0
        # for i in range(10):
        idx_test = np.where(triples_DF[4] == test_fold)
        idx_train = np.where(triples_DF[4] != test_fold)
        test_set = [triples[xx] for xx in idx_test[0]]
        train_set = [triples[xx] for xx in idx_train[0]]
        print(len(train_set) // args.batch_size)

        net = KG(args, max(entitys) + 1, max(relations) + 1,
                    args.e_dim, args.r_dim, adj_entity, adj_relation)
        net = net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

        if args.train_test_mode == 1:
            t_total = time.time()
            for e in range(args.n_epochs):
                t = time.time()
                net.train()
                all_loss = 0.0
                batch_count = 0
                for u1, u2, c, r, fold in DataLoader(train_set, batch_size=args.batch_size,
                                                     shuffle=True):
                    u1, u2, c, r = u1.to(device), u2.to(device), c.to(device), r.to(device)
                    logits, combine_drug, cell_embs = net(u1, u2, c)
                    optimizer.zero_grad()
                    loss = loss_fcn(logits, r.float())
                    loss.backward()
                    optimizer.step()
                    # .item():得到张量里的元素值
                    all_loss += loss.item()
                    torch.cuda.empty_cache()
                loss_train = all_loss / (len(train_set) // args.batch_size)
                print('[test_fold {},epoch {}],avg_loss={:.4f}'.format(test_fold, e,
                                                                       loss_train))
                if (e == 0):
                    best_train_loss = loss_train
                    torch.save(net.state_dict(),
                               '/net/{}_decoder{}.pkl'.format(now, test_fold))  # 保存网络中的参数
                    print("save model")
                    earlystop_count = 0

                else:
                    if best_train_loss > loss_train:
                        best_train_loss = loss_train
                        torch.save(net.state_dict(),
                                   '/net/{}_decoder{}.pkl'.format(now, test_fold))  # 保存网络中的参数
                        print("save model")
                        earlystop_count = 0

                    if earlystop_count != args.patience:
                        earlystop_count += 1
                    else:
                        print("early stop!!!!")
                        break

            print("\nOptimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        net = KG(args, max(entitys) + 1, max(relations) + 1,
                    args.e_dim, args.r_dim, adj_entity, adj_relation)
        net.load_state_dict(torch.load('/net/{}_decoder{}.pkl'.format(now, test_fold)))
        test_set = torch.LongTensor(test_set)
        test_loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            net.eval()
            drug1_ids = test_set[:, 0]
            drug2_ids = test_set[:, 1]
            cell_ids = test_set[:, 2]
            labels = test_set[:, 3]
            logitst, combine_drugt, cell_embst = net(drug1_ids, drug2_ids, cell_ids)
            logitst = logitst.cpu()
            all_labels.extend(labels.numpy())
            all_predictions.extend(logitst.numpy())

            loss = loss_fcn(logitst, labels.float())
            test_loss += loss.item()
            torch.cuda.empty_cache()

        test_loss /= len(test_set)
        mse = mean_squared_error(all_labels, all_predictions)
        print(f'Test Loss: {test_loss:.4f}, MSE: {mse:.4f}')

        # Extract and save embeddings
        drug_embeddings, cell_embeddings = [], []
        with torch.no_grad():
            net.eval()
            for u1, u2, c, r, fold in DataLoader(train_set, batch_size=args.batch_size, shuffle=False):
                _, combine_drug, cell_embs = net(u1, u2, c)
                drug_embeddings.append(combine_drug.cpu().numpy())
                cell_embeddings.append(cell_embs.cpu().numpy())
        torch.cuda.empty_cache()

        drug_embeddings = np.concatenate(drug_embeddings, axis=0)
        cell_embeddings = np.concatenate(cell_embeddings, axis=0)

        # Save the combined drug embeddings
        os.makedirs('/data1', exist_ok=True)
        np.save(os.path.join('/data1', 'drug_embeddings.npy'), drug_embeddings)
        np.save(os.path.join('/data1', 'cell_embeddings.npy'), cell_embeddings)





if __name__ == '__main__':
    seed = 55
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()