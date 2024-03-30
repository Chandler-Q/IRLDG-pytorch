#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:12:45 2023

@author: yons
"""
import os
import argparse

import torch
from train import run

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, default= os.path.join('..','data'),
                        help='datasets locating path')
        
    parser.add_argument('--save-path', type=str, default= os.path.join('..','result'),
                        help='result save path')
    # model design
    parser.add_argument('--dataset', type=str, default='gossipcop',
                        choices=['politifact', 'gossipcop','Weibo'])
    
    parser.add_argument('--feature', type=str, default='bert',
                        choices=['profile', 'spacy', 'bert', 'content'])
    
    parser.add_argument('--model', type=str, default='SAGE',
                        choices=['GCN', 'GAT', 'SAGE','Transformer'])
    
    parser.add_argument('--reweight', type=bool, default = False)
    
    # train setup
    parser.add_argument('--device', type=str, default = 'cuda' if torch.cuda.is_available() else 'cpu',
                        help = 'device choice')
    
    parser.add_argument('--epoch-size', type=int, default = 60,
                        help = 'number of epoch')
    
    parser.add_argument('--batch-size', type=int, default = 128,
                        help = 'batch_size')
    
    parser.add_argument('--lr', type=float, default = 0.005,
                        help = 'learning rate')
    
    parser.add_argument('--weight-decay', type=float, default = 0.001,
                        help = 'weight_decay')
    
    
    # hyper-parameter setup
    parser.add_argument('--T', type=int, default = 1,
                        help= 'T-stage: split the dynamic propagation network')
    
    parser.add_argument('--heads', type=int, default = 2,
                        help = 'multi heads for Attention machanism and RetNet')
    
    parser.add_argument('--hidden-dim', type=int, default = 128,
                        help = 'hidden-dim in RetNet')
    
    parser.add_argument('--GCN-layers', type=int, default = 1,
                        help = '# of GCNs Layers')
    
    parser.add_argument('--retnet-layers', type=int, default = 2,
                        help = 'Rentive Network dimension')
    
    parser.add_argument('--fnn-dim', type=int, default=256,
                        help = 'dim of feedforward neural networks')

    args = parser.parse_args()

    run(args)




