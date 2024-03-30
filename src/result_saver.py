#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:03:36 2023

@author: yons
"""

import pandas as pd
import os.path as osp

class result_saver:
    
    
    def __init__(self,save_path,
                 **kwargs):
        
        self.save_path = save_path
        self.result = dict()
        
        
    def input_res(self,**kwargs):
        
        if self.result != {}:
            for key,value in kwargs.items():
                self.result[key].append(value)
                
        else:
            for key,value in kwargs.items():
                self.result[key] = [value]
    
    def save(self,file_name):
        
        data = pd.DataFrame.from_dict(self.result)
        
        data.to_csv(osp.join(self.save_path,file_name + '.csv'))
    