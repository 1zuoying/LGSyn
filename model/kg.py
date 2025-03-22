import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np


class KG(nn.Module):

    def __init__(self, args, n_entitys, n_relations, e_dim, r_dim, adj_entity, adj_relation,
                 agg_method='Bi-Interaction'):
        super(KG, self).__init__()
        self.entity_embs = nn.Embedding(n_entitys, e_dim, max_norm=1)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)
        self.dropout = args.dropout
        self.n_iter = args.n_iter
        self.dim = e_dim
        self.WW = nn.Linear(256, 128, bias=False)
        self.n_heads = args.n_heads
        self.adj_entity = torch.tensor(adj_entity, dtype=torch.long)
        self.adj_relation = torch.tensor(adj_relation, dtype=torch.long)
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # 定义全连接层
        self.fc1 = nn.Linear(3776, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)

        self.relu = nn.ReLU()
        self._init_weight()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.agg_method = agg_method
        self.agg = 'concat'
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        if agg_method == 'concat':
            self.W_concat = nn.Linear(e_dim * 2, e_dim)
        else:
            self.W1 = nn.Linear(e_dim * self.n_heads, e_dim * 2)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(e_dim * self.n_heads, e_dim * 2)

        # 将模型参数和缓冲区移动到CUDA设备
        self.to('cuda')

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def get_neighbors_cell(self, cells):
        with torch.no_grad():
            cells = cells.view((-1, 1))
            entities = [cells]
            relations = []
            for h in range(self.n_iter):
                neighbor_entities = self.adj_entity[entities[h]].view((entities[h].shape[0], -1)).to('cuda')
                neighbor_relations = self.adj_relation[entities[h]].view((entities[h].shape[0], -1)).to('cuda')
                entities.append(neighbor_entities)
                relations.append(neighbor_relations)
            neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
            neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
            return neighbor_entities_embs, neighbor_relations_embs


    def get_neighbors_drug(self, drugs):
        with torch.no_grad():
            drugs = drugs.view((-1, 1))
            entities = [drugs]
            relations = []
            for h in range(self.n_iter):
                neighbor_entities = self.adj_entity[entities[h]].view((entities[h].shape[0], -1)).to('cuda')
                neighbor_relations = self.adj_relation[entities[h]].view((entities[h].shape[0], -1)).to('cuda')
                entities.append(neighbor_entities)
                relations.append(neighbor_relations)
            neighbor_entities_embs = [self.entity_embs(entity) for entity in entities]
            neighbor_relations_embs = [self.relation_embs(relation) for relation in relations]
            return neighbor_entities_embs, neighbor_relations_embs


    def sum_aggregator(self, embs):
        e_u = embs[0]
        if self.agg == 'concat':
            for i in range(1, len(embs)):
                e_u = torch.cat((embs[i], e_u), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(embs)):
                e_u += self.WW(embs[i])
        return e_u

    def GATMessagePass(self, h_embs, r_embs, t_embs):
        muti = []
        for i in range(self.n_heads):
            att_weights = self.attention(torch.cat((h_embs, r_embs), dim=-1)).squeeze(-1)
            att_weights_norm = F.softmax(att_weights, dim=-1)
            emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_embs)
            Wx = nn.Linear(self.dim, self.dim).to('cuda')  # 保留这行，将Wx实例化后移到CUDA上
            emb_i = Wx(emb_i.sum(dim=1))
            muti.append(emb_i)
        all_emb_i = torch.cat([c for c in muti], dim=-1)
        return all_emb_i

    def aggregate(self, h_embs, Nh_embs, agg_method='Bi-Interaction'):
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        else:
            return self.leakyRelu(self.W1(h_embs + Nh_embs))


    def forward(self, u1, u2, c):
        # 将输入张量移到CUDA设备
        c = c.to('cuda')
        u1 = u1.to('cuda')
        u2 = u2.to('cuda')

        # cell line embedding learning process
        t_embs, r_embs = self.get_neighbors_cell(c)
        h_embs = self.entity_embs(c)
        t_vectors_next_iter = [h_embs]
        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs = torch.cat([torch.unsqueeze(h_embs, 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                h_embs = torch.cat([h_embs for _ in range(self.n_heads)], dim=1)
                cell_embs_1 = self.aggregate(h_embs, vector, self.agg_method)
                t_vectors_next_iter.append(cell_embs_1)
            else:
                h_broadcast_embs = torch.cat([torch.unsqueeze(torch.sum(t_embs[i], dim=1), 1) for _ in range(t_embs[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs, r_embs[i], t_embs[i + 1])
                vector = self.leakyRelu(vector)
                embs = torch.cat([torch.sum(t_embs[i], dim=1) for _ in range(self.n_heads)], dim=1)
                cell_embs_1 = self.aggregate(embs, vector, self.agg_method)
                t_vectors_next_iter.append(cell_embs_1)
        Nh_embs_list = t_vectors_next_iter
        self.cell_embs = self.sum_aggregator(Nh_embs_list)

        # drug2 embedding learning process
        t_embs_drug2, r_embs_drug2 = self.get_neighbors_drug(u2)
        h_embs_drug2 = self.entity_embs(u2)
        drug2_t_vectors_next_iter = [h_embs_drug2]
        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug2 = torch.cat([torch.unsqueeze(h_embs_drug2, 1) for _ in range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug2 = torch.cat([h_embs_drug2 for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(h_embs_drug2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)
            else:
                h_broadcast_embs_drug2 = torch.cat([torch.unsqueeze(torch.sum(t_embs_drug2[i], dim=1), 1) for _ in range(t_embs_drug2[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug2, r_embs_drug2[i], t_embs_drug2[i + 1])
                vector = self.leakyRelu(vector)
                embs_d2 = torch.cat([torch.sum(t_embs_drug2[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug2_embs_1 = self.aggregate(embs_d2, vector, self.agg_method)
                drug2_t_vectors_next_iter.append(drug2_embs_1)
        Nh_embs_drug2_list = drug2_t_vectors_next_iter
        self.drug2_embs = self.sum_aggregator(Nh_embs_drug2_list)

        # drug1 embedding learning process
        t_embs_drug1, r_embs_drug1 = self.get_neighbors_drug(u1)
        h_embs_drug1 = self.entity_embs(u1)
        drug1_t_vectors_next_iter = [h_embs_drug1]
        for i in range(self.n_iter):
            if i == 0:
                h_broadcast_embs_drug1 = torch.cat([torch.unsqueeze(h_embs_drug1, 1) for _ in range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                h_embs_drug1 = torch.cat([h_embs_drug1 for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(h_embs_drug1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)
            else:
                h_broadcast_embs_drug1 = torch.cat([torch.unsqueeze(torch.sum(t_embs_drug1[i], dim=1), 1) for _ in range(t_embs_drug1[i + 1].shape[1])], dim=1)
                vector = self.GATMessagePass(h_broadcast_embs_drug1, r_embs_drug1[i], t_embs_drug1[i + 1])
                vector = self.leakyRelu(vector)
                embs_d1 = torch.cat([torch.sum(t_embs_drug1[i], dim=1) for _ in range(self.n_heads)], dim=1)
                drug1_embs_1 = self.aggregate(embs_d1, vector, self.agg_method)
                drug1_t_vectors_next_iter.append(drug1_embs_1)
        Nh_embs_drug1_list = drug1_t_vectors_next_iter
        self.drug1_embs = self.sum_aggregator(Nh_embs_drug1_list)

        # Calculate combination prediction score
        combine_drug = torch.max(self.drug1_embs, self.drug2_embs)
        logits = (combine_drug * self.cell_embs).sum(dim=1)
        del t_embs, r_embs, t_embs_drug1, r_embs_drug1, t_embs_drug2, r_embs_drug2
        torch.cuda.empty_cache()

        return logits, combine_drug, self.cell_embs


