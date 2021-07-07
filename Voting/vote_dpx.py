import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import csv
import pdb

def read_csv(file_name,index_matrix,prob_matrix):
    
#     print(file_name)
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            index_matrix.append(row[:5])
            prob_matrix.append(row[5:])
        
        return  np.array(index_matrix),np.array(prob_matrix)

def get_weight(path,trust_threhold,untrust_threhold,num_cls):

    data=pd.read_csv(path,header=None).values[1:, :]

    label = data[:, 1]
    predict = data[:, 2]
    num_cls = num_cls

    count = np.zeros((num_cls, ))
    sums = np.zeros((num_cls, ))
    for i in range(data.shape[0]):
        sums[int(label[i])] += 1
        if label[i] != predict[i]:
            count[int(label[i])] += 1

    num_cla = num_cls
    num_err = count
    num_sum = sums
    ratio_err = count/sums
    
    return ratio_err,num_sum

def cal_therhold_weight(weight_matrix,trust_threhold,untrust_threhold,ratio_trust,ratio_untrust,ratio_uncertain,num_cls): 
    
    ratio_err = 1 - weight_matrix  #error
    
    trust_index = ratio_err<trust_threhold
    untrust_index = ratio_err>untrust_threhold
    uncertain_index = [not trust_index[i] and not untrust_index[i] for i in range(num_cls)]
    
    cla_matrix = np.zeros(num_cls)
    cla_matrix[trust_index] = ratio_trust
    cla_matrix[untrust_index] = ratio_untrust
    cla_matrix[uncertain_index] = ratio_uncertain
    
    return cla_matrix

def filling_matrix(vote,index_matrix,prob_matrix,num_test):
    for j in range(num_test):

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

def cal_uncer_weight(windex,num_threhold,weight,num_cls): 
    
    ########1
    # 求可信任的类别数
    num = num_threhold
    windex = (windex > num)
    
    weight_vector = np.ones(num_cls)
    
    for i in range(num_cls):
        if windex[i]  == False:
            weight_vector[i] = weight
        else:
            pass
    return weight_vector

def filling_matrix(vote,index_matrix,prob_matrix,num_test):
    for j in range(num_test):
        
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

def get_info(weight_path,index_path,trust_threhold,untrust_threhold,num_threhold,weight,ratio_trust,ratio_untrust,ratio_uncertain,num_test,num_cls):
    
    cla_index, cla_prob = read_csv(index_path,[],[])

    ratio_err, num_sums = get_weight(weight_path,trust_threhold,untrust_threhold,num_cls)

    # 正确率
    cor_ratio = 1 - ratio_err
    
    # 阈值划分的正确率
    thr_ratio = cal_therhold_weight(cor_ratio,trust_threhold,untrust_threhold,ratio_trust,ratio_untrust,ratio_uncertain,num_cls)

    # 不信任的类别
    unc_ratio = cal_uncer_weight(num_sums,num_threhold,weight,num_cls)
    
    # 加上正确率的比例
    vot_ratio = np.zeros([num_test,num_cls])
    vot_ratio = filling_matrix(vot_ratio,cla_index,cla_prob,num_test)
  
    return cor_ratio,thr_ratio,unc_ratio,vot_ratio

def get_accuracy(vote_matrix,num_test):
    
    category = []
    for i in range(num_test):
        category.append(np.where(vote_matrix[i]==np.max(vote_matrix[i])))

    category = [n for a in category for n in a ]
    category = np.array(category)
    category = [n for a in category for n in a ]


    import pickle
    fr=open('./modelzoo/eval_label.pkl','rb')
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
#     print('accuray',accuray)
    
    return accuray

def get_result(choice,cor_ratio_list,thr_ratio_list,unc_ratio_list,vot_ratio_list):
    
    if choice == 0:
        vot_ratio_list_1 = vot_ratio_list
        
    elif choice == 1:
        vot_ratio_list_1 = [ vot * unc_ratio_list[index] for index,vot in enumerate(vot_ratio_list)]
        
    elif choice == 2:
        vot_ratio_list_1 = [ vot * unc_ratio_list[index] * thr_ratio_list[index]**0.6 for index,vot in enumerate(vot_ratio_list)]

    elif choice == 3:
        vot_ratio_list_1 = [ vot * cor_ratio_list[index]**0.6 for index,vot in enumerate(vot_ratio_list)]
    
    elif choice == 4:
        vot_ratio_list_1 = [ vot * thr_ratio_list[index]**0.6 for index,vot in enumerate(vot_ratio_list)]
        
    elif choice == 5:
        vot_ratio_list_1 = [ vot * unc_ratio_list[index] * cor_ratio_list[index]**0.6 for index,vot in enumerate(vot_ratio_list)]
        
    all_vot_ratio_1 = sum(vot_ratio_list_1)
    all_vot_ratio_1 = all_vot_ratio_1/model_num
    acc = get_accuracy(all_vot_ratio_1,num_test)
    
    return acc

    weight_path_all = ['./modelzoo/1019_EfficientGCN-B4_uav-random_sample_top1_result_eval_0.7057860262008734.csv',
                   './modelzoo/1018_pa-resgcn-b19_uav-random_sample_top1_result_eval_0.6872270742358079.csv',
                   './modelzoo/shift-gcn-top1-6905.csv',
                   './modelzoo/ddnet_top1_47.csv',
                   './modelzoo/shift-gcn-top1-6905.csv',
                   './modelzoo/MS-G3D-top1-66.27.csv',
                   './modelzoo/1018_pa-resgcn-b19_uav-random_sample_top1_result_eval_0.6839519650655022.csv'
                   
                  ]


index_path_all  = ['./modelzoo/1019_EfficientGCN-B4_uav-random_sample_top5_result_eval_0.7057860262008734.csv',
                   './modelzoo/1018_pa-resgcn-b19_uav-random_sample_top5_result_eval_0.6872270742358079.csv',
                   './modelzoo/shift-gcn-top5-6905.csv',
                   './modelzoo/ddnet_top5_47.csv',
                   './modelzoo/multi-input-shift-gcn-top5-6954.csv',
                   './modelzoo/MS-G3D-top5-66.27.csv',
                    './modelzoo/1018_pa-resgcn-b19_uav-random_sample_top5_result_eval_0.6839519650655022.csv'


                    ]

index = [1,1,1,1,1,0,0]

weight_path = [weight_path_all[i] for i,value in enumerate(index) if value==1]
index_path = [index_path_all[i] for i,value in enumerate(index) if value==1]
model_num = len(weight_path)

voxel = range(40,80,1)
voxel = [i/100.0 for i in voxel]
print(voxel)

for trust_threhold in [0.8]:

    for untrust_threhold in [0.5]:
        
        for ratio_trust in [0.7]:
            
            for ratio_uncertain in [0.5]:
                
                for ratio_untrust in [0.4]:
                    
                    for num_threhold in [8]:
                        
                        for weight in [0.49]:

                            print('-------------------------------')
                            print('trust_threhold:',trust_threhold)
                            print('untrust_threhold:',untrust_threhold)
                            print('ratio_trust:',ratio_trust)
                            print('ratio_uncertain:',ratio_uncertain)
                            print('ratio_untrust:',ratio_untrust)

                            print('num_threhold:',num_threhold)
                            print('weight:',weight)

                            num_test=1832
                            num_cls=155
                            
                            cor_ratio_list = []
                            thr_ratio_list = []
                            unc_ratio_list = []
                            vot_ratio_list = []



                            for i in range(model_num):
                                cor_ratio,thr_ratio,unc_ratio,vot_ratio = get_info(weight_path[i],index_path[i],trust_threhold,untrust_threhold,num_threhold,weight,ratio_trust,ratio_untrust,ratio_uncertain,num_test,num_cls)

                                cor_ratio_list.append(cor_ratio)
                                thr_ratio_list.append(thr_ratio)
                                unc_ratio_list.append(unc_ratio)
                                vot_ratio_list.append(vot_ratio)

                            for i in [0,1,2,3,4,5]:
                                acc = get_result(i,cor_ratio_list,thr_ratio_list,unc_ratio_list,vot_ratio_list)
                                print('第{}个acc:{}'.format(i,acc))
                                