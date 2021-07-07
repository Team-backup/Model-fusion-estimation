#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-07-29 08:56
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""

import pandas as pd
import numpy as np
import lightgbm as lgb  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pickle
import pdb
from sklearn import metrics

## load data
# # 加载数据
# iris = load_iris()
train_data = np.load('/home/data2_8t/dataset/UAV-Human/skeleton/random_sample/train_data_joint.npy')
test_data = np.array(pickle.load(open('/home/data2_8t/dataset/UAV-Human/skeleton/random_sample/train_label.pkl','rb')))

print(train_data.shape)
B,C,T,V,H = train_data.shape

train_data = train_data.reshape(14892,-1)

X_train, X_test, y_train, y_test = train_test_split(train_data, test_data[1], shuffle = False, random_state = 2019)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

trn_data = lgb.Dataset(X_train[:1100], y_train[:1100],silent=True)
val_data = lgb.Dataset(X_test[:50], y_test[:50],silent=True)


params = {  
    'boosting_type': 'gbdt',  
    'objective': 'multiclass',  
    'num_class': 155,  
    'metric': 'multi_error',  
    "n_estimators": [50, 100, 150],
    "max_depth": [4,5, 7],
    "num_leaves": [300,900,1200],
    'min_data_in_leaf': 500,  
    "learning_rate" : [0.01,0.05,0.1], 
    'feature_fraction': 0.8,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 5,  
    'lambda_l1': 0.4,  
    'lambda_l2': 0.5,  
    'min_gain_to_split': 0.2,  
    'verbose': 0, 
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 2,
    }


# parameters = {
#               'max_depth': [15, 20, 25, 30, 35],
#               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#               'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#               'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#               'bagging_freq': [2, 4, 5, 6, 8],
#               'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
#               'lambda_l2': [0, 10, 15, 35, 40],
#               'cat_smooth': [1, 10, 15, 20, 35]
# }

parameters = {
              'max_depth': [15, 20],
              'learning_rate': [0.01, 0.02],
              'feature_fraction': [0.6, 0.7],
              'bagging_fraction': [0.6, 0.7],
              'bagging_freq': [2, 4],
              'lambda_l1': [0, 0.1],
              'lambda_l2': [0, 10],
              'cat_smooth': [1, 10]
}


gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'multiclass',
                         num_class =  155,  
                         metric =  'multi_error',  
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0,
                         silent=False,
                         min_gain_to_split =  0.2,  
                         verbose= 0, 
                         device='gpu',
                         gpu_platform_id= 0,
                         gpu_device_id=2)

# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
print('Training...')
gsearch.fit(X_train[:500], y_train[:500])

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


gbm.save_model('model.txt') 
print('Predicting...')
y_prob = gbm.predict(X_test, num_iteration=clf.best_iteration)
y_pred = [list(x).index(max(x)) for x in y_prob]
print("AUC score: {:<8.5f}".format(metrics.accuracy_score(y_pred, y_test)))


'''
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=params, 
                            cv = 5, scoring="roc_auc", verbose=5)
gsearch.fit(X_0_train, Y_0_train)


clf = lg.train(params, 
                trn_data, 
                
                num_boost_round = 1000,
                valid_sets = [trn_data,val_data], 
                verbose_eval = 100, 
                early_stopping_rounds = 100,
                )

clf.save_model('model.txt') 

print('Predicting...')
y_prob = clf.predict(X_test, num_iteration=clf.best_iteration)
y_pred = [list(x).index(max(x)) for x in y_prob]


print("AUC score: {:<8.5f}".format(metrics.accuracy_score(y_pred, y_test)))
'''

