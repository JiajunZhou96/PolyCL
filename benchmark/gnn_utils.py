import os
os.chdir(os.pardir) # to the parent dir

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import utils

config = utils.get_config(path = './other_evals/config_graph.json', print_dict = False)

ATOM_LIST = list(range(0,config["num_atom_type"]))    # 0 to index [*]

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE, 
    Chem.rdchem.BondType.DOUBLE, 
    Chem.rdchem.BondType.TRIPLE, 
    Chem.rdchem.BondType.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

num_atom_type = config["num_atom_type"]
num_chirality_tag = config["num_chirality_tag"]

num_bond_type = config["num_bond_type"]
num_bond_direction = config["num_bond_direction"]


def construct_graph_data(smile, y):

    mol = Chem.MolFromSmiles(smile)
    
    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
    
    # bond attributes
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y)
    
    return data

class Graph_Dataset(Dataset):
    def __init__(self, smiles, labels):
        super(Dataset, self).__init__()
        self.smiles = smiles
        self.labels = labels

    def __getitem__(self, index):
        
        smile = self.smiles[index]
        label = self.labels[index]
        data = construct_graph_data(smile, label)
        return data
    
    def __len__(self):
        return len(self.smiles)
    

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

class GCNConv(MessagePassing):
    
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def gcn_norm(self, edge_index, num_nodes, dtype):

        edge_weight = torch.ones((edge_index.size(1), ), dtype = dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        norm = self.gcn_norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)
        
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_embeddings, aggr = self.aggr, norm = norm) # 这里的 dim 更改过，要知道一下
        
        return out

    def message(self, x_j, edge_attr, norm):

        return norm.view(-1, 1) * (x_j + edge_attr)

class GINConv(MessagePassing):
    
    # https://arxiv.org/abs/1810.00826
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        self.mlp = nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]
        
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_embeddings, aggr = self.aggr)
        
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(nn.Module):
    def __init__(self, num_layer, jk, emb_dim, drop_ratio=0, gnn_type = "gcn"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jk = jk

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))
            
            elif gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
                
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layer - 1:
                h = nn.functional.dropout(h, self.drop_ratio)
            else:
                h =nn.functional.dropout(nn.functional.relu(h), self.drop_ratio)

            h_list.append(h)
        
        if self.jk == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.jk == "last":
            node_representation = h_list[-1]
        elif self.jk == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.jk == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, JK, emb_dim, drop_ratio, gnn_type = gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        x= data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))
    
