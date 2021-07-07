import numpy as np
import pdb

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        #返回边界关系数组
        self.get_edge(layout)

        #返回临接矩阵，shape[25,25]
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)

        #返回A矩阵, 按照论文介绍的方法，有3种构造策略， 所以shape[3,25,25]
        self.get_adjacency('spatial')


    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1

        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]

            # print(neighbor_link)
            # [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5),
            #  (7, 6), (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
            #   (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), 
            #   (21, 22), (22, 7), (23, 24), (24, 11)]
   

            self.edge = self_link + neighbor_link
            self.center = 21 - 1  #每个数据集有一个人体中心点标号，这个数据集是20号

            # print(self.edge)  #
            # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), 
            # (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
            #  (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), 
            #  (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), 
            #  (23, 23), (24, 24), 

            #  (0, 1), (1, 20), (2, 20), 
            # (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), 
            #  (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
            #   (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            #    (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]

            # print(self.center) #20

    
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")


    def get_adjacency(self, strategy):
        
        valid_hop = range(0, self.max_hop + 1, self.dilation) #[0, 1]
        adjacency = np.zeros((self.num_node, self.num_node)) #(25, 25)的空矩阵
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1  #self.hop_dis->恢复程0，1矩阵

        # print(adjacency)
        # [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
        #   0.]
        #  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
        #   0.]
        #  [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
        #   0.]
        #  [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0.]
        #  [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
        #   0.]
        #  [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0.]
        # ......

        # pdb.set_trace()

        normalize_adjacency = normalize_digraph(adjacency) #标准化

        if strategy == 'uniform': #只有1个类
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance': #按照距离来分子集，分为root点和其他点

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial': #文章里的方法
            
            A = []
            for hop in valid_hop:
                #构造三个空矩阵 a_root， a_close， a_further
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                #把normalize_adjacency按照近心点、远心点、自身点，拆分成分三类
                #如果i,j临接点距离等于i到中心店距离，划分到a_root，小->a_close，大->a_further
                #体现了我们对不同类型结点的划分权值子集的思想，同时也体现了人体关节之间结点的空间联系

                for i in range(self.num_node): 
                    for j in range(self.num_node):  
                        if self.hop_dis[j, i] == hop:    #self.hop_dis 0,1,inf
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            # print(normalize_adjacency)
            # [[0.25       0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.33333333 0.         0.         0.         0.33333333 0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.25       0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.33333333 0.5        0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.33333333 0.5        0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.33333333 0.33333333
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.33333333 0.33333333
            #   0.33333333 0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.33333333
            #   0.33333333 0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #   ......

  
            # print(a_root)
            # [0.         0.         0.         0.         0.         0.
            #   0.33333333 0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.33333333 0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.33333333 0.33333333 0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.33333333 0.33333333 0.33333333 0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.33333333 0.33333333 0.33333333
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.33333333 0.33333333
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.33333333]
            #  [0.25       0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.33333333 0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #   ......


            # print(a_close)
            # [[0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.33333333 0.         0.         0.         0.33333333 0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.33333333 0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
              # ......

            # print(a_further)
            # [[0.         0.33333333 0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.33333333 0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.2        0.         0.         0.
            #   0.        ]
            #  [0.         0.         0.         0.         0.33333333 0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.         0.         0.         0.         0.         0.
            #   0.        ]
            #   ......


            A = np.stack(A) #(3, 25, 25)

            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):

    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # print(A) A 是自身和临接关系都为1的矩阵，shape[25,25]
 #  [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
 #  0.]
 # [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
 #  0.]
 # [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
 #  0.]
 # [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 #  0.]
 # [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
 #  0.]
 # [0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 #  0.]
 #  ......


    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf

 # print(hop_dis) 全inf矩阵，shape[25,25]
 # [[inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 #  inf inf inf inf inf inf inf]
 # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 #  inf inf inf inf inf inf inf]
 # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 #  inf inf inf inf inf inf inf]
 # ......

   
    #矩阵连乘 np.linalg.matrix_power(a, n)矩阵连乘函数 a^n，
    # 这里其中max_hop= 1，d = 0,1 ，d = 0，生成[25,25]的对角矩阵，d=1生成A^1也就是A
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]

    # print(transfer_mat)
    #对角
#     array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 0.],
#    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 0.],
#    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 0.],
#    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 0.],
#    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     ........
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 1.]]), 
    
#     #A
#     array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
#     1., 0., 0., 0., 0., 0., 0., 0., 0.],
#    [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 1., 0., 0., 0., 0.],
#    [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#     0., 0., 0., 0., 1., 0., 0., 0., 0.],
      # ........
    


    #将#True，False化 
    arrive_mat = (np.stack(transfer_mat) > 0)  
 #    print(arrive_mat)
 #    [[[ True False False ... False False False]
 #  [False  True False ... False False False]
 #  [False False  True ... False False False]
 #  ...
 #  [False False False ...  True False False]
 #  [False False False ... False  True False]
 #  [False False False ... False False  True]]

 # [[ True  True False ... False False False]
 #  [ True  True False ... False False False]
 #  [False False  True ... False False False]
 #  ...
 #  [False False False ...  True False False]
 #  [False False False ... False  True  True]
 #  [False False False ... False  True  True]]]
 

    #将hop_dis矩阵变成对角线是0，临接是1，非连接处是inf ，shape [25,25]
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    # pdb.set_trace()
    return hop_dis   


def normalize_digraph(A):
    
    Dl = np.sum(A, 0)  #(25,)
    num_node = A.shape[0]  #25
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) #-1次方

    AD = np.dot(A, Dn) #
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD