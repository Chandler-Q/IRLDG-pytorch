# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:26:05 2023

@author: c
"""
import os.path as osp

from torch_geometric.transforms import ToUndirected
from torch_geometric.datasets import UPFD
from Weibo import Weibo

class Dataset:
    
    def __init__(self,path):
        
        self.path = path
        
    def load(self,name,feature,spilt):
        
        if name=='Weibo':
            
            return Weibo(self.path, name, feature, spilt, ToUndirected())
        
        else:
            
            return UPFD(osp.join(self.path,'UPFD'), name, feature, spilt, ToUndirected())