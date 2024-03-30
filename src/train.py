# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:02:47 2023

@author: c
"""

from result_saver import result_saver

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from Dataset import Dataset
from model import RGT

from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score

def run(args):
    
    
    dataset = Dataset(args.path)
    train_dataset = dataset.load(args.dataset, args.feature, 'train')
    val_dataset = dataset.load(args.dataset, args.feature, 'val')
    test_dataset = dataset.load(args.dataset, args.feature, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(args.device)
    
    model = RGT(     GCN_model = args.model,
                     GCN_layers = args.GCN_layers,
                     retnet_layers = args.retnet_layers,
                     input_dim = train_dataset.num_features,
                     hidden_dim = args.hidden_dim,
                     fnn_dim = args.fnn_dim,
                     out_dim = train_dataset.num_classes,
                     heads = args.heads,
                     T_fold = args.T
                     ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)

            for t in range(args.T):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index.to(torch.int64), data.batch,t)
                loss = F.nll_loss(out, data.y.to(torch.int64))
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.num_graphs

        return total_loss / len(train_loader.dataset)/args.T
    
    
    def recurrent_train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            s_n_1 = [
                [
                torch.zeros(args.hidden_dim // args.heads, args.hidden_dim // args.heads).unsqueeze(0).repeat(data.num_graphs, 1, 1)
                for _ in range(args.heads)
                ]
                for _ in range(args.retnet_layers)
                ]
            
            for t in range(args.T):
                optimizer.zero_grad()
                out,s_n = model.recurrent_forward(data.x, data.edge_index, data.batch,s_n_1,t)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
                s_n_1 = s_n
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
            
        return total_correct / total_examples
    
    @torch.no_grad()
    def test_all(loader):
        model.eval()
        
        total_correct = total_examples = 0
    
        for data in loader:
            data = data.to(device)
            s_n_1 = [
                [
                torch.zeros(args.hidden_dim // args.heads, args.hidden_dim // args.heads).unsqueeze(0).repeat(data.num_graphs, 1, 1)
                for _ in range(args.heads)
                ]
                for _ in range(args.retnet_layers)
                ]
            
            for t in range(args.T):
                pred,s_n = model.recurrent_forward(data.x, data.edge_index.to(torch.int64), data.batch,s_n_1,t)
                s_n_1 = s_n
                total_correct += int((pred.argmax(-1) == data.y).sum())
                total_examples += data.num_graphs
            
        return total_correct / total_examples
    
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
                torch.zeros(args.hidden_dim // args.heads, args.hidden_dim // args.heads).unsqueeze(0).repeat(data.num_graphs, 1, 1)
                for _ in range(args.heads)
                ]
                for _ in range(args.retnet_layers)
                ]
            for t in range(args.T):
            
                pred,s_n = model.recurrent_forward(data.x, data.edge_index.to(torch.int64), data.batch,s_n_1,t)
                y_true[t].extend(data.y.tolist())
                y_pred[t].extend(pred.argmax(dim=-1).tolist())
                
                s_n_1 = s_n
                
        return y_pred, y_true
    
    saver = result_saver(args.save_path)
    
    for epoch in range(1, args.epoch_size+1):
       loss = train()
       train_acc = test_all(train_loader)
       val_acc = test_all(val_loader)
       t_acc = test_all(test_loader)
       y_pred,y_true = dynamic_test(test_loader)
       
       test_acc = [accuracy_score(y_true[i], y_pred[i]) for i in range(args.T)]
       test_f1  = [f1_score(y_true[i], y_pred[i],zero_division=1) for i in range(args.T)]
       test_prec = [precision_score(y_true[i], y_pred[i],zero_division=1) for i in range(args.T)]
       test_rec = [recall_score(y_true[i], y_pred[i],zero_division=1) for i in range(args.T)]
       
       if (epoch-1)%1 ==0:
           print('='*80)
           print(f"Epoch: {epoch:02d}\n"
                 f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f},test: {t_acc:.4f} ')
           print("stage acc: ",end = '')
           for t in range(args.T):
               print(f" T{t+1}:{test_acc[t]:.4f}",end = '  ')
           print("\nstage F1:  ",end = '')
           for t in range(args.T):
               print(f" T{t+1}:{test_f1[t]:.4f}",end = '  ')
            
           print('\n')
       saver.input_res(loss = loss, train_acc = train_acc,test_acc = t_acc,
                       stage_acc = test_acc, stage_f1 = test_f1, 
                       stage_prec = test_prec, stage_rec = test_rec)
       
    if args.save_path is not None:
        saver.save("_".join([args.feature, args.dataset,args.model,str(args.T)]))
        