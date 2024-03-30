# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:19:10 2023

@author: c
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import (GCNConv,
                                GATConv,
                                SAGEConv,
                                TransformerConv)


from retnet import RetNet



class RGT(nn.Module):
    
    def __init__(self,
                 GCN_model,
                 GCN_layers,
                 retnet_layers,
                 input_dim,
                 hidden_dim,
                 fnn_dim,
                 out_dim,
                 heads = 2,
                 T_fold = 5,
                 double_v_dim=False,
                 concat = True):
        
        super().__init__()
        
        self.concat = concat
        
        assert hidden_dim % heads == 0, "hidden_size must be divisible by heads"
        
        self.retnet = RetNet(retnet_layers, hidden_dim, fnn_dim, heads, double_v_dim=double_v_dim)
        
        
        if GCN_model == 'GCN':
            
            self.GCN = nn.ModuleList([
                GCNConv(input_dim,hidden_dim) 
                for _ in range(GCN_layers)])
            
        elif GCN_model == 'GAT':
            
            self.GCN = nn.ModuleList([
                GATConv(input_dim,hidden_dim,heads=heads,concat=False) 
                for _ in range(GCN_layers)])
            
        elif GCN_model == 'SAGE':
            
            self.GCN = nn.ModuleList([
                SAGEConv(input_dim,hidden_dim, Aggregation='mean') 
                for _ in range(GCN_layers)])
            
        elif GCN_model == 'Transformer':
            
            self.GCN = nn.ModuleList([
                TransformerConv(input_dim,hidden_dim,heads=heads,concat=False) 
                for _ in range(GCN_layers)])
            
        else:
            raise ValueError('GCN model is not in the default list')
          

        
        if self.concat:
            self.lin1 = nn.Linear(input_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.lin3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.lin4 = nn.Linear(hidden_dim // 2, out_dim)

        self.T = T_fold
        
    def forward(self,x,edge_index,batch,t):
           
        # edge_index = edge_index[:,t]
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        
        if self.concat:
            news = x[root]
            news = self.lin1(news)
        
        edge_index = self.dynamic_edge_index(edge_index.cpu(),root.cpu(),x.shape[0],t,self.T)
        
        for i in range(len(self.GCN)):
            x = self.GCN[i](x, torch.cat(edge_index,dim=-1).to(x.device)).relu()

        x = [x[i.unique(),:] for i in edge_index]
        
        x = pad_sequence(x,batch_first=True,padding_value=0)
        
        x = self.retnet(x)
        
        x,_ = x.max(dim=-2)
        
        if self.concat:
            x = self.lin2(torch.cat([news,x],dim=-1)).relu()
        
        x = self.lin3(x).relu()
        
        x = self.lin4(x)
        
        return x.log_softmax(dim=-1)
            
    def recurrent_forward(self,x,edge_index,batch,s_n_1,t):
        
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            
        if self.concat:
            news = x[root]
            news = self.lin1(news)
        
        edge_index = self.dynamic_edge_index(edge_index.cpu(),root.cpu(),x.shape[0],t,self.T)

        for i in range(len(self.GCN)):
            x = self.GCN[i](x, torch.cat(edge_index,dim=-1).to(x.device)).relu()

        x = [x[i.unique(),:] for i in edge_index]
        
        x = pad_sequence(x,batch_first=True,padding_value=0)
        
        x,s_n = self.retnet.forward_recurrent(x, s_n_1, t)
        
        x,_ = x.max(dim=-2)
        
        if self.concat:
            x = self.lin2(torch.cat([news,x],dim=-1)).relu()
        
        x = self.lin3(x).relu()
        
        x = self.lin4(x)
        
        return x.log_softmax(dim=-1),s_n
    
    
    def dynamic_edge_index(self,edge_index,root,total_num,t,T):
        
        if t == 0:
            root = torch.cat([root,torch.tensor([total_num+1])])
        
            edge_batch = ((edge_index[0].view(-1,1)>=root[:-1]) 
                          & (edge_index[0].view(-1,1)<root[1:])).nonzero()[:,1]
            
            start = (edge_batch[1:] - edge_batch[:-1]).nonzero(as_tuple=False).view(-1)
            start = torch.cat([start.new_zeros(1), start + 1], dim=0)
            
            num = edge_batch.unique(return_counts=True)[1]
            
            # end = start + num -1
            
            self.edge_batch = start,num
               
        
        edge_root,num = self.edge_batch
        
        start = edge_root + torch.floor(num*t/T).int()
        end   = edge_root + torch.floor(num*(t+1)/T).int()
        
        idx = [torch.arange(start[i],end[i]) for i in range(num.shape[0])]
        
        edge_index = [edge_index[:,i] for i in idx]
        
        return edge_index
        
        
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
import torch.nn.functional as F
import os.path as osp
import argparse
from sklearn.metrics import f1_score,accuracy_score

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact',
                        choices=['politifact', 'gossipcop'])
    parser.add_argument('--feature', type=str, default='bert',
                        choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--model', type=str, default='GAT',
                        choices=['GCN', 'GAT', 'SAGE','Transformer'])
    
    parser.add_argument('--T', type=int, default = 5,
                        help= 'T-stage split dynamic propagation network')
    
    parser.add_argument('--retnet-layers', type=int, default=2,
                        help = 'Rentive Network dimension')
    
    parser.add_argument('--ret-dim', type=int, default=128,
                        help = 'dim of feedforward layer')
        
    parser.add_argument('--fnn-dim', type=int, default=64,
                        help = 'dim of feedforward neural networks')
    
    parser.add_argument('--heads', type=int, default=2,
                        help = 'multi heads for Attention machanism and RetNet')
    
    args = parser.parse_args()

    path = osp.join( '..', 'data', 'UPFD')
    train_dataset = UPFD(path, args.dataset, args.feature, 'train', ToUndirected())
    val_dataset = UPFD(path, args.dataset, args.feature, 'val', ToUndirected())
    test_dataset = UPFD(path, args.dataset, args.feature, 'test', ToUndirected())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGT(     GCN_model=args.model,
                     GCN_layers=1,
                     retnet_layers=args.retnet_layers,
                     input_dim=train_dataset.num_features,
                     hidden_dim=args.ret_dim,
                     fnn_dim=args.fnn_dim,
                     out_dim=train_dataset.num_classes,
                     heads= args.heads
                     ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            for t in range(args.T):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch,t)
                loss = F.nll_loss(out, data.y.int())
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.num_graphs

        return total_loss / len(train_loader.dataset)/args.T
    
    @torch.no_grad()
    def test(loader):
        model.eval()
    
        total_correct = total_examples = 0
        
        y_true = []
        y_pred = []
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch,0).argmax(dim=-1)
            
            total_correct += int((pred == data.y).sum())
            total_examples += data.num_graphs
            
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
            
        return total_correct / total_examples,f1_score(y_true,y_pred,zero_division=1)
    
    @torch.no_grad()
    def dynamic_test(loader):
        model.eval()
    
        y_true = [[] for i in range(args.T)]
        y_pred = [[] for i in range(args.T)]
        
        for batch,data in enumerate(loader):
            
            data = data.to(device)
            
            # retnet_layers, heads, [batch_size, ret_dim // args.heads, ret_dim // args.heads]
            s_n_1 = [
                [
                torch.zeros(args.ret_dim // args.heads, args.ret_dim // args.heads).unsqueeze(0).repeat(data.num_graphs, 1, 1)
                for _ in range(args.heads)
                ]
            for _ in range(args.retnet_layers)
            ]
            
            for t in range(args.T):
            
                pred,s_n = model.recurrent_forward(data.x, data.edge_index, data.batch,s_n_1,t)
                
                y_true[t].extend(data.y.tolist())
                y_pred[t].extend(pred.argmax(dim=-1).tolist())
        
        return accuracy_score(y_true[4], y_pred[4]),f1_score(y_true[4], y_pred[4])
    for epoch in range(1, 61):
        loss = train()
        train_acc,_ = test(train_loader)
        val_acc,_ = test(val_loader)
        test_acc,F1 = dynamic_test(test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f},F1: {F1:.4f}')
            
        