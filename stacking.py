#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:03:08 2021

@author: makangbo
"""

import pandas as pd 
import numpy as np
import csv
import pdb


k_efficientnet_vote_matrix1 = np.zeros([16724,155])
k_efficientnet_vote_matrix2 = np.zeros([16724,155])
k_efficientnet_vote_matrix3 = np.zeros([16724,155])

t_efficientnet_vote_matrix1 = np.zeros([4500,155])
t_efficientnet_vote_matrix2 = np.zeros([4500,155])
t_efficientnet_vote_matrix3 = np.zeros([4500,155])

k_efficientnet_vote_matrix = [k_efficientnet_vote_matrix1,k_efficientnet_vote_matrix2,k_efficientnet_vote_matrix3]
t_efficientnet_vote_matrix = [t_efficientnet_vote_matrix1,t_efficientnet_vote_matrix2,t_efficientnet_vote_matrix3]


def read_csv(file_name,index_matrix,prob_matrix):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        for row in reader:

            index_matrix.append(row[:5])
            prob_matrix.append(row[5:])
        
        return  np.array(index_matrix),np.array(prob_matrix)

#按照三折划分的顺序
def filling_matrix(vote,index_matrix,prob_matrix,Lx):

    for j in range(index_matrix.shape[0]): 
        
        index1 = round(float(index_matrix[j][0]))
        index2 = round(float(index_matrix[j][1]))
        index3 = round(float(index_matrix[j][2]))
        index4 = round(float(index_matrix[j][3]))
        index5 = round(float(index_matrix[j][4]))
        
        prob1 = prob_matrix[j][0]
        prob2 = prob_matrix[j][1]
        prob3 = prob_matrix[j][2]
        prob4 = prob_matrix[j][3]
        prob5 = prob_matrix[j][4]
        
        '''
        if j == 0:
            print(index1,index2,index3,index4,index5)
            print(prob1,prob2,prob3,prob4,prob5)
            print('============================')
        '''
        
        vote[Lx[j],index1] += float(prob1)
        vote[Lx[j],index2] += float(prob2)
        vote[Lx[j],index3] += float(prob3)
        vote[Lx[j],index4] += float(prob4)
        vote[Lx[j],index5] += float(prob5)
               
    return vote

#按照正常顺序
def order_filling_matrix(vote,index_matrix,prob_matrix):
    for j in range(index_matrix.shape[0]):

        index1 = round(float(index_matrix[j][0]))
        index2 = round(float(index_matrix[j][1]))
        index3 = round(float(index_matrix[j][2]))
        index4 = round(float(index_matrix[j][3]))
        index5 = round(float(index_matrix[j][4]))
        
        prob1 = prob_matrix[j][0]
        prob2 = prob_matrix[j][1]
        prob3 = prob_matrix[j][2]
        prob4 = prob_matrix[j][3]
        prob5 = prob_matrix[j][4]
        
        '''
        if j == 0:
            print(index1,index2,index3,index4,index5)
            print(prob1,prob2,prob3,prob4,prob5)
            print('============================')
        '''
        
        vote[j,index1] += float(prob1)
        vote[j,index2] += float(prob2)
        vote[j,index3] += float(prob3)
        vote[j,index4] += float(prob4)
        vote[j,index5] += float(prob5)
        

    return vote


L1 = np.load('/Users/makangbo/Desktop/L1.npy')
L2 = np.load('/Users/makangbo/Desktop/L2.npy')
L3 = np.load('/Users/makangbo/Desktop/L3.npy')

k_path ={'efficientnet':['/Users/makangbo/Desktop/3foldmodel/k1_efficientnet_top5_result_eval_6807.csv',
                      '/Users/makangbo/Desktop/3foldmodel/k2_efficientnet_top5_result_eval_6884.csv',
                      '/Users/makangbo/Desktop/3foldmodel/k3_efficientnet_top5_result_eval_6912.csv'],
        
        'shiftgcn':['/Users/makangbo/Desktop/3foldmodel/1-fold-shift-gcn-validation-top5.csv',
                    '/Users/makangbo/Desktop/3foldmodel/2-fold-shift-gcn-validation-top5.csv',
                    '/Users/makangbo/Desktop/3foldmodel/3-fold-shift-gcn-validation-top5.csv'],
        
        'resgcn':['/Users/makangbo/Desktop/3foldmodel/k1_resgcn_top5_result_eval_6568.csv',
                  '/Users/makangbo/Desktop/3foldmodel/k2_resgcn_top5_result_eval_6593.csv',
                  '/Users/makangbo/Desktop/3foldmodel/k3_resgcn_top5_result_eval_6664.csv']
                
    }

test_path ={'efficientnet':['/Users/makangbo/Desktop/3foldmodel/efficientnet-top5_result_test_k1.csv',
                      '/Users/makangbo/Desktop/3foldmodel/efficientnet-top5_result_test_k2.csv',
                      '/Users/makangbo/Desktop/3foldmodel/efficientnet-top5_result_test_k3.csv'],

            'shiftgcn':['/Users/makangbo/Desktop/3foldmodel/1-fold-shift-gcn-6640-test-top5.csv',
                        '/Users/makangbo/Desktop/3foldmodel/2-fold-shift-gcn-6640-test-top5.csv',
                        '/Users/makangbo/Desktop/3foldmodel/3-fold-shift-gcn-6640-test-top5.csv'
                    ],
            
            'resgcn':['/Users/makangbo/Desktop/3foldmodel/resgcn_top5_result_test_k1.csv',
                      '/Users/makangbo/Desktop/3foldmodel/resgcn_top5_result_test_k2.csv',
                      '/Users/makangbo/Desktop/3foldmodel/resgcn_top5_result_test_k3.csv'
                    ]
}


L  = [L1,L2,L3]

index_matrix = {'efficientnet':[[],[],[]],'shiftgcn':[[],[],[]],'resgcn':[[],[],[]]}
prob_matrix = {'efficientnet':[[],[],[]],'shiftgcn':[[],[],[]],'resgcn':[[],[],[]]}

vote_matrix = {'efficientnet':[np.zeros([16724,155]),np.zeros([16724,155]),np.zeros([16724,155])],
               'shiftgcn':[np.zeros([16724,155]),np.zeros([16724,155]),np.zeros([16724,155])],
               'resgcn':[np.zeros([16724,155]),np.zeros([16724,155]),np.zeros([16724,155])]}

t_index_matrix = {'efficientnet':[[],[],[]],'shiftgcn':[[],[],[]],'resgcn':[[],[],[]]}
t_prob_matrix = {'efficientnet':[[],[],[]],'shiftgcn':[[],[],[]],'resgcn':[[],[],[]]}

t_vote_matrix = {'efficientnet':[np.zeros([4500,155]),np.zeros([4500,155]),np.zeros([4500,155])],
            'shiftgcn':[np.zeros([4500,155]),np.zeros([4500,155]),np.zeros([4500,155])],
            'resgcn':[np.zeros([4500,155]),np.zeros([4500,155]),np.zeros([4500,155])]}



for k in ['efficientnet', 'shiftgcn','resgcn']:
    #training data部分
    for i in range(3):
        index_matrix[k][i],prob_matrix[k][i] =  read_csv(k_path[k][i],index_matrix[k][i],prob_matrix[k][i])       
        vote_matrix[k][i] = filling_matrix(vote_matrix[k][i],index_matrix[k][i],prob_matrix[k][i],L[i])

    #test data部分
    for i in range(3):
        t_index_matrix[k][i],t_prob_matrix[k][i] =  read_csv(test_path[k][i],t_index_matrix[k][i],t_prob_matrix[k][i])
        t_vote_matrix[k][i] = order_filling_matrix(t_vote_matrix[k][i],t_index_matrix[k][i],t_prob_matrix[k][i])

#New_feature training data部分三个矩阵
k_efficientnet_vote_matrix[0]  = vote_matrix['efficientnet'][0] +  vote_matrix['efficientnet'][1] + vote_matrix['efficientnet'][2]
k_efficientnet_vote_matrix[1]  = vote_matrix['shiftgcn'][0] +  vote_matrix['shiftgcn'][1] + vote_matrix['shiftgcn'][2]
k_efficientnet_vote_matrix[2]  = vote_matrix['resgcn'][0] +  vote_matrix['resgcn'][1] + vote_matrix['resgcn'][2] 
     
#New_feature test data部分三个矩阵
t_efficientnet_vote_matrix[0] = (t_vote_matrix['efficientnet'][0] + t_vote_matrix['efficientnet'][1] + t_vote_matrix['efficientnet'][2] )/3
t_efficientnet_vote_matrix[1] = (t_vote_matrix['shiftgcn'][0] + t_vote_matrix['shiftgcn'][1] + t_vote_matrix['shiftgcn'][2] )/3
t_efficientnet_vote_matrix[2] = (t_vote_matrix['resgcn'][0] + t_vote_matrix['resgcn'][1] + t_vote_matrix['resgcn'][2] )/3


t_efficientnet_vote_matrixx = t_efficientnet_vote_matrix[0] + t_efficientnet_vote_matrix[1] + t_efficientnet_vote_matrix[2]
k_efficientnet_vote_matrixx = k_efficientnet_vote_matrix[0] + k_efficientnet_vote_matrix[1] + k_efficientnet_vote_matrix[2]

print(t_efficientnet_vote_matrixx.shape)
print(k_efficientnet_vote_matrixx.shape)

import pickle
fr=open('/Users/makangbo/Desktop/val_label.pkl','rb')
inf = pickle.load(fr)
doc = open('1.txt', 'a')

#print(np.array(inf[0]).shape)
#print(inf[1]) (1832,)
















