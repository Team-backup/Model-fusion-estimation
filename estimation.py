1.	top-1，top-5 accuracy
  out, _ , f_fea = self.model(x) 
                # out [64, 155]
                # f_fea [64, 256]

                # print(subjs)
                # subjs_unsqueeze = torch.unsqueeze(subjs,dim=1)
                # knn_fea = torch.cat((f_fea,subjs_unsqueeze),dim=1)
                # knn_fea_to_store[num*f_fea.size(0):(num+1)*f_fea.size(0),:]= knn_fea.cpu().data.numpy()

                # Getting Loss
               
                loss = self.loss_func(out, y)  #1.9105
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0) #64
                reco_top1 = out.max(1)[1] # (value index)，此处index 每行max
                num_top1 += reco_top1.eq(y).sum().item()  #True False
                reco_top5 = torch.topk(out,5)[1]  #input中 5个最大值的index  [64, 5]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))]) #54

  
2.	混淆矩阵
                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1  #(155, 155) 
可视化示例
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define sample labels
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))

3.	AP
  def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

   	    output = torch.tensor([-0.5816, -0.3873, -1.0215, -1.0145,  0.4053])
sorted, indices = torch.sort(output, descending=True, dim=-1)
target = torch.tensor([0,1,0,0,0])
pos_count = 0.000000001
total_count = 0.
precision_at_i = 0.
reprecision_at_i = average_precision(output,target)


4.ROC
# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pdb

#标签二值化
y_score =       np.array( [[-3.58, -0.31,  1.78,-0.18, -0.77],
       [-2.15,  1.11, -2.39,-3.19, -0.16],
       [ 1.89, -3.9 , -6.3 ,1.26, -0.38],
       [-4.53, -0.63,  1.96,-2.23, -6.71],
       [ 1.4 , -1.78, -6.26,-1.35,  1.1],
       [-4.3 , -1.45,  3.29,-1.26, -0.32]])

y_test =        np.array([[0, 0, 1,0,0],
       [0, 1, 0,0,0],
       [0, 0, 0,1,0],
       [0, 0, 1,0,0],
       [0, 0, 0,0,1],
       [0, 0, 1,0,0]])

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes): # 遍历三个类别
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


