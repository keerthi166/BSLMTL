import numpy as np
import csv
from sklearn import metrics
from collections import defaultdict

print 'BioBaseline based on the reviewer suggestion'
nRun=10
taskIds=['ebola','flu','flavi']
aupr_score=np.zeros((len(taskIds), nRun))
auroc_score=np.zeros((len(taskIds), nRun))

# Assumptions: No duplicate entry and user-user graph file has entry for both directions 

# Dictionaries to store the bipartite graphs
vhg_dict=defaultdict(list) # Use set instead of list, if there are any duplicates
hvg_dict=defaultdict(list)
vvg_dict=defaultdict(lambda: defaultdict()) # Dictionary of Dictionaries
hhg_dict=defaultdict(lambda: defaultdict()) # Dictionary of Dictionaries

# Populate the training data in dictionaries
# Read Virus-Virus Graph
with open('data/virus_virus_similarities.txt', 'rb') as csvfile:
  vvg_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in vvg_reader:
    vvg_dict[row[0]][row[1]]=float(row[2])

with open('data/human_human_similarities.txt', 'rb') as csvfile:
  hhg_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in hhg_reader:
    hhg_dict[row[0]][row[1]]=float(row[2])
    
# Read Virus-Human Graph
with open('data/virus_human_graph.txt', 'rb') as csvfile:
  vhg_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in vhg_reader:
    vhg_dict[row[0]].append(row[1])
    hvg_dict[row[1]].append(row[0])
tt=0
for taskId in taskIds:   
  for runId in range(nRun):
    # Read training data
    trainvhg_dict=defaultdict(list)
    trainhvg_dict=defaultdict(list)
    
    with open('data/10fold_data/{0}_train{1}.txt'.format(taskId,str(runId+1)), 'rb') as csvfile:
      task_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in task_reader:
        trainvhg_dict[row[0]].append(row[1])
        trainhvg_dict[row[1]].append(row[0])
    # Read Test Data
    y_true=list()
    y_pred=list()
    with open('data/10fold_data/{0}_test{1}.txt'.format(taskId,str(runId+1)), 'rb') as csvfile:
      task_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in task_reader:
        
        # row=[v,u]=[virus,human]=[pathogen,host]
        y_true.append(int(row[2]))
        # Fetch Host proteins Hv related to virus v
        Hv=vhg_dict[row[0]]+trainvhg_dict[row[0]]
        max_score=0; 
        for h in Hv:
          # Measure similarity score between u and Hv
          ss=hhg_dict[row[1]].get(h,0)
          if ss>max_score:
            max_score=ss
        # Fetch Pathogen proteins Pu related to host u
        Pu=hvg_dict[row[1]]+trainhvg_dict[row[1]]    
        for p in Pu:
          # Measure similarity score between u and Hv
          ss=vvg_dict[row[0]].get(p,0)
          if ss>max_score:
            max_score=ss
        y_pred.append(max_score)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred,pos_label=1)
    #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    aupr_score[tt][runId]=metrics.average_precision_score(y_true, y_pred)
    auroc_score[tt][runId]=metrics.roc_auc_score(y_true, y_pred)
      
  tt+=1 
  
print 'AUPR Mean: ',np.mean(aupr_score,axis=1)
print 'AUPR STD: ',np.std(aupr_score,axis=1)
print 'AUROC Mean: ',np.mean(auroc_score,axis=1)
print 'AUROC STD: ',np.std(auroc_score,axis=1)




