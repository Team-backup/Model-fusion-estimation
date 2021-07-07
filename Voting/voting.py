#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 20:52:23 2021

@author: makangbo
"""

import pandas as pd 
import numpy as np
import csv
import pdb

index_matrix1 = []
prob_matrix1= []

index_matrix2 = []
prob_matrix2 = []

index_matrix3 = []
prob_matrix3 = []

index_matrix4 = []
prob_matrix4 = []

index_matrix5 = []
prob_matrix5 = []

vote_matrix = np.zeros([1832,155])
vote_matrix1 = np.zeros([1832,155])
vote_matrix2 = np.zeros([1832,155])
vote_matrix3= np.zeros([1832,155])
vote_matrix4= np.zeros([1832,155])
vote_matrix5= np.zeros([1832,155])


def read_csv(file_name,index_matrix,prob_matrix):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            index_matrix.append(row[:5])
            prob_matrix.append(row[5:])
        
        return  np.array(index_matrix),np.array(prob_matrix)

def filling_matrix(vote,index_matrix,prob_matrix):
    for j in range(1832):

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

model_num = 4

index_matrix1,prob_matrix1 = read_csv('/Users/makangbo/Desktop/modelzoo/1019_EfficientGCN-B4_uav-random_sample_top5_result_eval_0.7057860262008734.csv',index_matrix1,prob_matrix1)
index_matrix2,prob_matrix2 = read_csv( '/Users/makangbo/Desktop/modelzoo/1018_pa-resgcn-b19_uav-random_sample_top5_result_eval_0.6872270742358079.csv',index_matrix2,prob_matrix2)
index_matrix3,prob_matrix3 = read_csv('/Users/makangbo/Desktop/modelzoo/shift-gcn-top5-6905.csv',index_matrix3,prob_matrix3)
index_matrix4,prob_matrix4 = read_csv('/Users/makangbo/Desktop/modelzoo/ddnet_top5_47.csv',index_matrix4,prob_matrix4)
#index_matrix5,prob_matrix5 = read_csv('/Users/makangbo/Desktop/modelzoo/MS-G3D-top5-66.27.csv',index_matrix5,prob_matrix5)

weight_matrix1 = 1 - np.load('/Users/makangbo/Desktop/Efficientgcn_err1.npy')  #1 - error
weight_matrix2 = 1 -np.load('/Users/makangbo/Desktop/resgcn_err2.npy')
weight_matrix3 = 1 - np.load('/Users/makangbo/Desktop/shiftgcn_err3.npy')
weight_matrix4 = 1 - np.load('/Users/makangbo/Desktop/shiftgcn_err3.npy')
#weight_matrix5 = 1 - np.load('/Users/makangbo/Desktop/shiftgcn_err3.npy')

windex1 = np.load('/Users/makangbo/Desktop/Efficientgcn_sums1.npy')
windex2 = np.load('/Users/makangbo/Desktop/resgcn_sums2.npy')
windex3 = np.load('/Users/makangbo/Desktop/shiftgcn_sums3.npy')
windex4 = np.load('/Users/makangbo/Desktop/shiftgcn_sums3.npy')
#windex5 = np.load('/Users/makangbo/Desktop/shiftgcn_sums3.npy')


def cal_weight(windex,weight_matrix): 
    
    ########1
    windex = (windex > 8)

    for i  in range(155):
        if windex[i]  == False:
            weight_matrix[i] = 0.6
        else:
            pass
        
    ########2
    ratio_err = 1 - weight_matrix  #error
    
    trust_threhold = 0.3
    untrust_threhold = 0.6
    
    trust_index = ratio_err<trust_threhold
    untrust_index = ratio_err>untrust_threhold
    uncertain_index = [not trust_index[i] and not untrust_index[i] for i in range(num_cla)]
    
    trust_cla = np.array(index)[trust_index]
    uncertain_cla = np.array(index)[uncertain_index]
    untrust_cla = np.array(index)[untrust_index]
    
    cla_matrix = np.zeros(155)
    cla_matrix[trust_cla] = 0.9
    cla_matrix[untrust_cla] = 0.2
    cla_matrix[uncertain_index] = 0.5
    
    return weight_matrix,cla_matrix

def add_weight(windex,weight_matrix,vote_matrix,index_matrix,prob_matrix,choice = '1'):
    weight_matrix,cla_matrix = cal_weight(windex,weight_matrix)

    vote_matrix = filling_matrix(vote_matrix,index_matrix,prob_matrix)
    
    if choice == '1':
        vote_matrix = np.multiply(vote_matrix,exp(weight_matrix))
    elif choice == '2':
        vote_matrix = np.multiply(vote_matrix,cla_matrix)
    elif choice == '3':
        vote_matrix = np.multiply(vote_matrix,exp(weight_matrix))
        vote_matrix = np.multiply(vote_matrix,cla_matrix)
    elif choice == '0':
        vote_matrix = vote_matrix

    else:
        print('wrong!')
    
    return vote_matrix


vote_matrix1 = add_weight(windex1,weight_matrix1,vote_matrix1,index_matrix1,prob_matrix1,'0')
vote_matrix2 = add_weight(windex2,weight_matrix2,vote_matrix2,index_matrix2,prob_matrix2,'0')
vote_matrix3 = add_weight(windex3,weight_matrix3,vote_matrix3,index_matrix3,prob_matrix3,'0')
vote_matrix4 = add_weight(windex4,weight_matrix4,vote_matrix4,index_matrix4,prob_matrix4,'0')
#vote_matrix5 = add_weight(windex5,weight_matrix5,vote_matrix5,index_matrix5,prob_matrix5,'0')

vote_matrix = vote_matrix1 + vote_matrix2 + vote_matrix3 + vote_matrix4 + vote_matrix5

vote_matrix = vote_matrix/model_num 


category = []
for i in range(1832):
    category.append(np.where(vote_matrix[i]==np.max(vote_matrix[i])))
    
category = [n for a in category for n in a ]
category = np.array(category)
category = [n for a in category for n in a ]


import pickle
fr=open('/Users/makangbo/Desktop/val_label.pkl','rb')
inf = pickle.load(fr)
doc = open('1.txt', 'a')

print(inf[1], file=doc)
#print(inf[1])

cat = np.array(category)
gt = np.array(inf[1])

rank = 0
for i in range(len(cat)):
    if cat[i] == gt[i]:
        rank += 1

accuray = rank/len(cat)
print('accuray',accuray)

