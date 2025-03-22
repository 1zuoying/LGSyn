import collections
import numpy as np


def construct_kg(kgTriples):
    print('生成知识图谱索引图')
    kg = dict()
    for triple in kgTriples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg





def getKgIndexsFromKgTriples(kg_triples):
    kg_indexs = collections.defaultdict(list)
    for h, r, t in kg_triples:
        kg_indexs[str(h)].append([int(t), int(r)]) # 头实体对应的尾实体和关系
    return kg_indexs


def filetDateSet(dataSet, user_pos):
    return [i for i in dataSet if str(i[0]) in user_pos]


# 根据kg邻接列表，得到实体邻接列表和关系邻接列表
def construct_adj(neighbor_sample_size, kg_indexes, entity_num):
    print('生成实体邻接列表和关系邻接列表')
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
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


import numpy as np
import scipy.sparse as sp

from codes.models import osUtils as ou
import sys
import random
import copy


def normalize_adj(mx):
    """Row Normalized Sparse Matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def readKGData(path='/gemini/code/data/kg.txt'):
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


def readRecData(path='/gemini/code/data/synergy.txt', test_ratio=0.2):
    print('Read Drug Combination Synergy Data...')
    drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, i, r, flod in ou.readTriple(path, sep='\t'):
        drug_set1.add(int(d1))
        drug_set2.add(int(d2))
        cell_set.add(int(i))
        triples.append((int(d1), int(d2), int(i), float(r), int(flod)))


    return list(drug_set1), list(drug_set2), list(cell_set), triples

