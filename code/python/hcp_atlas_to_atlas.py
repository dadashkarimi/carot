# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from numpy import linalg as la
import scipy.sparse
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from scipy.io import loadmat
import os

from scipy.io import loadmat
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pandas as pd
import random
import scipy.io as sio

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pandas as pd
import random
from numpy import savetxt
import sys
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr   
#num_time_points = 1
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from time import sleep
from sklearn import linear_model

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize
import numpy as np
import scipy.io as sio
import h5py

from numpy import genfromtxt
import time
import csv

import sys

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s','--source',type=str,help="source atlas")
parser.add_argument('-t','--target',type=str,help="target atlas")
parser.add_argument('-task','--task',type=str,default="all",help="task")
parser.add_argument('-c','--cost',type=str,default="functional",help="cost matrix")
parser.add_argument('-id','--id',type=bool,default=False,help="id rate")
parser.add_argument('-num_iters','--num_iters',default=100,type=int,help="number of iterations")
parser.add_argument('-test_size','--test_size',default=100,type=int,help="test size")
parser.add_argument('-id_direction','--id_direction',type=str,default="ot-ot",help="direction of id rate: orig-orig ot-ot orig-ot ot-orig")

args = parser.parse_args()


#argv = sys.argv[1:]

all_data = {}
atlases = ['dosenbach','schaefer','brainnetome','power','craddock','shen','shen368','craddock400']
#atlases = ['craddock','shen','shen368','craddock400']
#atlases = ['shen','shen368']

tasks = ["rest1","rest2","gambling","wm","motor","lang","social","relational","emotion"]
#tasks = ["rest1","rest2"]#,"gambling","wm","motor","lang","social","relational","emotion"]
#tasks = ["gambling","wm","motor","lang","social","relational","emotion"]

path = '/data_dustin/store4/Templates/HCP'

coord = {}
coord['schaefer'] =pd.read_csv('/data_dustin/store4/Templates/schaefer_coords.csv', sep=',',header=None)
coord['brainnetome'] =pd.read_csv('/data_dustin/store4/Templates/brainnetome_coords.csv', sep=',',header=None)
coord['shen'] =pd.read_csv('/data_dustin/store4/Templates/shen_coords.csv', sep=',',header=None)
coord['shen368'] =pd.read_csv('/data_dustin/store4/Templates/shen_368_coords.csv', sep=',',header=None)
coord['power'] =pd.read_csv('/data_dustin/store4/Templates/power_coords.txt', sep=',',header=None)
coord['dosenbach'] =pd.read_csv('/data_dustin/store4/Templates/dosenbach_coords.txt', sep=',',header=None)
coord['craddock'] =pd.read_csv('/data_dustin/store4/Templates/craddock_coords.txt', sep=',',header=None)
coord['craddock400'] =pd.read_csv('/data_dustin/store4/Templates/craddock_400_coords.txt', sep=',',header=None)

# Loading Atlas ...
if args.id == True:
    tasks = ["rest1","rest2"]
if args.task !="all":
    tasks = [args.task]

atlases = [args.source,args.target] 
for atlas in tqdm(atlases,desc = 'Loading Atlases ..'):
    zero_nodes = set()

    for task in tasks:
        data = sio.loadmat(os.path.join(path,atlas,task+'.mat'))
        x = data['all_mats']
        idx = np.argwhere(np.all(x[..., :] == 0, axis=0))
        p = [p1  for (p1,p2) in idx]
        zero_nodes.update(p)

    for task in tasks:
        data = sio.loadmat(os.path.join(path,atlas,task+'.mat'))
        x = data['all_mats']
        np.delete(x,list(zero_nodes),1)
        all_data[(atlas,task)] = x



all_behav = genfromtxt('data/268/all_behav.csv', delimiter=',')
all_sex = genfromtxt('data/268/gender.csv', delimiter=',')


def normalize(x):
    if all(v == 0 for v in x):
        return x
    else:
        if max(x) == min(x):
            return x
        else:
            return (x-min(x))/(max(x)-min(x))

from scipy import stats
def corr2_coeff(A, B):
	# Rowwise mean of input arrays & subtract from input arrays themeselves
	A_mA = A - A.mean(1)[:, None]
	B_mB = B - B.mean(1)[:, None]    # Sum of squares across rows
	ssA = (A_mA**2).sum(1)
	ssB = (B_mB**2).sum(1)
	return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)+1e-6
    s_y = y.std(1, ddof=n - 1)+1e-6
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis],mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


#atlases =['aal', 'zalesky','power','harvard_oxford_subcortical','harvard_oxford_cortical','harvard_oxford','dosenbach','craddock'] 
def atlas_ot(source,target,train_index,test_index,task,n_iters):

    n =all_data[(source,task)].shape[0] 
    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    num_time_points = t
    test_size = len(test_index)


    from collections import defaultdict
    G =  sinkhorn(source,target,train_index,task)
    results_pearson = defaultdict(list)
    results_mse = defaultdict(list)

    test_time_series = np.zeros((n,p2,num_time_points))

    c1_train = np.zeros((test_size,p2*p2))
    c2_train = np.zeros((test_size,p2*p2))
    step = 0
    id_count_ot = 0
    id_count_orig = 0
    id_conn = {}

    for i in test_index:
        for j in range(num_time_points):

            a = all_data[(source,task)][i,:,j]
            a = a + 1e-6
            a = normalize(a)
            b= np.transpose(G[j]).dot(a)
            test_time_series[i,:,j] = b

        time_series_orig = all_data[(target,task)][i,:,:]
        time_series_pred = test_time_series[i,:,:]
        c1 = generate_correlation_map(time_series_orig,time_series_orig)
        c2 = generate_correlation_map(time_series_pred,time_series_pred)


        c1 = c1.flatten()
        c2 = c2.flatten()

        c1[np.isnan(c1)] = 0
        c2[np.isnan(c2)] = 0


        c1_train[step,:] = c1
        c2_train[step,:] = c2
        step = step + 1
        
        
        corr = spearmanr(c1,c2)
        mse = ((c1 - c2)**2).mean(axis=0)
        results_mse[(source,target)].append(mse)
        results_pearson[(source,target)].append(corr[0])
        


    results = np.zeros((n_iters,2))
    for i in range(n_iters):
        s1, s2 = evaluate(c1_train,c2_train,all_behav[test_index],test_size)
        results[i,:] = (s1,s2)
        #print("IQ Prediction: orig: {0}, ot: {1}".format(s1,s2))
    df = pd.DataFrame(results, columns=['orig','ot'])
    df.to_csv(source+"_"+target+"_"+task+"_iq.csv")
    
    print("Mean IQ Performance of task={0}: orig: {1}, ot: {2}".format(task,np.mean(results[:,0]),np.mean(results[:,1])))
    print("Std IQ Performance of task={0}: orig: {1}, ot: {2}".format(task,np.std(results[:,0]),np.std(results[:,1])))
    #print(dict(results))
    for source,target in results_pearson:
        print('source: {0} target: {1} Task: {2} MSE: {3}'.format(source,target,task,np.mean(results_mse[(source,target)])))
        print('source: {0} target: {1} Task: {2} Pearson: {3}'.format(source,target,task,np.mean(results_pearson[(source,target)])))
    print("###################################################################################")    


def atlas_ot1(source,target,train_index,task,is_id,n_iters):
   #atlases =['power','dosenbach','craddock'] 


    from collections import defaultdict



    n =all_data[(source,task)].shape[0] 
    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    num_time_points = t

    print("source: {0} target: {1} task: {2} p1: {3} p2: {4} ".format(source,target,task,p1,p2))
    print("time points: {0} ".format(num_time_points))

    frame_size = num_time_points


    #ids = np.arange(n)
    #np.random.shuffle(ids)
    #train_test_fraction = 0.7
    #train_size = int(train_test_fraction*n)
    #test_size = n - train_size
    #train_index = ids[0:train_size]
    test_index = ids[train_size:]



    G = np.zeros((num_time_points,p1,p2))
    M = ot.dist(coord[source],coord[target],metric='euclidean')
    try:
        if args.cost == "euclidean":
            M = ot.dist(coord[source],coord[target],metric='euclidean')
            M /= M.max()
        elif args.cost == "functional":
            M = np.zeros((p1,p2))
            for i in train_index:
                a = all_data[(source,task)][i,:,:]
                b = all_data[(target,task)][i,:,:]

                Mi = generate_correlation_map(a,b)
                Mi = (Mi - np.min(Mi))/np.ptp(Mi)
                Mi = 1-Mi
                M = M + Mi
            M = M/train_size
    except ValueError:
        print("oops! cost measure is wrong.")

    for i in tqdm(train_index,desc="OT .."):
        for j in range(0,num_time_points,frame_size):
            lambd = 0.001
            a = all_data[(source,task)][i,:,j]
            b = all_data[(target,task)][i,:,j]
            a = normalize(a) + 1e-6
            b = normalize(b) + 1e-6
            a = a/np.sum(a)
            b = b/np.sum(b)
            T = ot.emd(a, b, M)#,lambd,numItermax=2000,verbose=False,method = "sinkhorn")
            #method = "SAG"
            #T = ot.stochastic.solve_semi_dual_entropic(a, b, M, lambd, method, numItermax=1000)
            T = T/T.sum(axis=0,keepdims=1)
            G[j:j+frame_size] = G[j:j+frame_size]+ T
        G[j:j+frame_size] = G[j:j+frame_size]/train_size

    G = G/G.sum(axis=1,keepdims=1)

    from scipy.stats.stats import spearmanr

    results_pearson = defaultdict(list)
    results_mse = defaultdict(list)

    test_time_series = np.zeros((n,p2,num_time_points))

    c1_train = np.zeros((test_size,p2*p2))
    c2_train = np.zeros((test_size,p2*p2))
    step = 0
    id_count_ot = 0
    id_count_orig = 0
    id_conn = {}
    if is_id:
        source_tasks = [x for x in tasks if x!=task]  
        source_task = source_tasks[0]
        for ii in tqdm(test_index,desc="correlation .. "):
            time_series_orig = all_data[(target,source_task)][ii,:,:]
            c1 = generate_correlation_map(time_series_orig,time_series_orig)
            c1 = c1.flatten()
            id_conn[ii] = c1


    for i in test_index:
        for j in range(num_time_points):

            a = all_data[(source,task)][i,:,j]
            a = a + 1e-6
            a = normalize(a)
            b= np.transpose(G[j]).dot(a)
            test_time_series[i,:,j] = b

        time_series_orig = all_data[(target,task)][i,:,:]
        time_series_pred = test_time_series[i,:,:]
        c1 = generate_correlation_map(time_series_orig,time_series_orig)
        c2 = generate_correlation_map(time_series_pred,time_series_pred)


        c1 = c1.flatten()
        c2 = c2.flatten()

        c1[np.isnan(c1)] = 0
        c2[np.isnan(c2)] = 0


        c1_train[step,:] = c1
        c2_train[step,:] = c2
        step = step + 1
        
        
        corr = spearmanr(c1,c2)
        mse = ((c1 - c2)**2).mean(axis=0)
        results_mse[(source,target)].append(mse)
        results_pearson[(source,target)].append(corr[0])
        

        if is_id:
            source_tasks = [x for x in tasks if x!=task]  
            source_task = source_tasks[0]
            id_map_ot = {}
            id_map_orig = {}
            for ii in test_index:
                id_map_ot[ii] = spearmanr(id_conn[ii],c2)
                id_map_orig[ii] = spearmanr(id_conn[ii],c1)
            closest_id_ot = max(id_map_ot, key=id_map_ot.get)
            closest_id_orig = max(id_map_orig, key=id_map_orig.get)
            if closest_id_ot == i:
                id_count_ot = id_count_ot + 1
            if closest_id_orig == i:
                id_count_orig = id_count_orig + 1
            print("orig = {:.6f}".format(float(id_count_orig)/step))
            print("ot = {:.6f}".format(float(id_count_ot)/step))


    results = np.zeros((n_iters,2))
    for i in range(n_iters):
        s1, s2 = evaluate(c1_train,c2_train,all_behav[test_index],test_size)
        results[i,:] = (s1,s2)
        #print("IQ Prediction: orig: {0}, ot: {1}".format(s1,s2))
    df = pd.DataFrame(results, columns=['orig','ot'])
    df.to_csv(source+"_"+target+"_"+task+"_iq.csv")
    
    df = pd.DataFrame(np.array([float(id_count_orig)/test_size,float(id_count_ot)/test_size]).reshape(1,2))
    df.to_csv(source+"_"+target+"_"+task+"_id.csv")

    print("Mean IQ Performance of task={0}: orig: {1}, ot: {2}".format(task,np.mean(results[:,0]),np.mean(results[:,1])))
    print("Std IQ Performance of task={0}: orig: {1}, ot: {2}".format(task,np.std(results[:,0]),np.std(results[:,1])))
    #print(dict(results))
    for source,target in results_pearson:
        print('source: {0} target: {1} Task: {2} MSE: {3}'.format(source,target,task,np.mean(results_mse[(source,target)])))
        print('source: {0} target: {1} Task: {2} Pearson: {3}'.format(source,target,task,np.mean(results_pearson[(source,target)])))
    print("###################################################################################")    

def evaluate(c1,c2,y,test_size):

    ids = np.arange(test_size)
    np.random.shuffle(ids)
    train_test_fraction = 0.9
    g3_train_size = int(train_test_fraction*test_size)
    g3_test_size = test_size - g3_train_size

    g3_train_index = ids[0:g3_train_size]
    g3_test_index = ids[g3_train_size:]

    clf = Ridge(alpha=1.0)
    clf.fit(c1[g3_train_index,:], y[g3_train_index])
    all_pred_1 = clf.predict(c1[g3_test_index])

    clf.fit(c2[g3_train_index,:], y[g3_train_index])
    all_pred_2 = clf.predict(c2[g3_test_index])

    #print(all_pred_1.shape,y[g3_test_size])
    s1 = np.corrcoef(all_pred_1, y[g3_test_index])[0, 1]
    s2 = np.corrcoef(all_pred_2, y[g3_test_index])[0, 1]
    return s1,s2
        
def sinkhorn(source,target,train_index,task):

    from collections import defaultdict



    n =all_data[(source,task)].shape[0] 
    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    num_time_points = t


    frame_size = num_time_points
    train_size = len(train_index)


    

    G = np.zeros((num_time_points,p1,p2))
    M = ot.dist(coord[source],coord[target],metric='euclidean')
    try:
        if args.cost == "euclidean":
            M = ot.dist(coord[source],coord[target],metric='euclidean')
            M /= M.max()
        elif args.cost == "functional":
            M = np.zeros((p1,p2))
            for i in train_index:
                a = all_data[(source,task)][i,:,:]
                b = all_data[(target,task)][i,:,:]

                Mi = generate_correlation_map(a,b)
                Mi = (Mi - np.min(Mi))/np.ptp(Mi)
                Mi = 1-Mi
                M = M + Mi
            M = M/train_size
    except ValueError:
        print("oops! cost measure is wrong.")

    for j in tqdm(range(0,num_time_points,frame_size),desc="OT .."):
        for i in train_index:
            lambd = 0.001
            a = all_data[(source,task)][i,:,j]
            b = all_data[(target,task)][i,:,j]
            a = normalize(a) + 1e-6
            b = normalize(b) + 1e-6
            a = a/np.sum(a)
            b = b/np.sum(b)
            T = ot.sinkhorn(a, b, M,lambd,verbose=False)#,lambd,numItermax=2000,verbose=False,method = "sinkhorn")
            #method = "SAG"
            #T = ot.stochastic.solve_semi_dual_entropic(a, b, M, lambd, method, numItermax=1000)
            T = T/T.sum(axis=0,keepdims=1)
            G[j:j+frame_size] = G[j:j+frame_size]+ T
        G[j:j+frame_size] = G[j:j+frame_size]/train_size

    G = G/G.sum(axis=1,keepdims=1)
    return G

from scipy.stats.stats import spearmanr
def id_rate(G1,G2,source,target,test_index,task,direction):
    
    n =all_data[(source,task)].shape[0] 

    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    num_time_points = t
    test_size = len(test_index)

    test_time_series_1 = np.zeros((n,p2,num_time_points))
    test_time_series_2 = np.zeros((n,p2,num_time_points))

    c1_train = np.zeros((test_size,p2*p2))
    c2_train = np.zeros((test_size,p2*p2))
    step = 0
    id_count_ot = 0
    id_count_orig = 0
    id_conn_2 = {}

    source_tasks = [x for x in tasks if x!=task]  
    source_task = source_tasks[0]

    for i in test_index:
        for j in range(num_time_points):

            a = all_data[(source,task)][i,:,j]
            a = a + 1e-6
            a = normalize(a)
            b= np.transpose(G1[j]).dot(a)
            test_time_series_1[i,:,j] = b

            a = all_data[(source,source_task)][i,:,j]
            a = a + 1e-6
            a = normalize(a)
            b= np.transpose(G2[j]).dot(a)
            test_time_series_2[i,:,j] = b



        time_series_orig_1 = all_data[(target,source_task)][i,:,:]
        time_series_orig_2 = all_data[(target,task)][i,:,:]

        time_series_pred_1 = test_time_series_1[i,:,:]
        time_series_pred_2 = test_time_series_2[i,:,:]
        #c1 = generate_correlation_map(time_series_orig_2,time_series_orig_2)
        if direction == "ot-ot":
            c1 = generate_correlation_map(time_series_pred_1,time_series_pred_1)
            c2 = generate_correlation_map(time_series_pred_2,time_series_pred_2)
        elif direction == "ot-orig":
            c1 = generate_correlation_map(time_series_pred_1,time_series_pred_1)
            c2 = generate_correlation_map(time_series_orig_2,time_series_orig_2)
        elif direction == "orig-orig":
            c1 = generate_correlation_map(time_series_orig_1,time_series_orig_1)
            c2 = generate_correlation_map(time_series_orig_2,time_series_orig_2)
        elif direction == "orig-ot":
            c1 = generate_correlation_map(time_series_orig_1,time_series_orig_1)
            c2 = generate_correlation_map(time_series_pred_2,time_series_pred_2)
        else:
            print("Dirction of ID rate is wrong!")

        c1 = c1.flatten()
        c2 = c2.flatten()

        c1[np.isnan(c1)] = 0
        c2[np.isnan(c2)] = 0


        c1_train[step,:] = c1
        c2_train[step,:] = c2
        step = step + 1
        
        
    def argmax(lst):
        return lst.index(max(lst))

    id_map_ot = {}
    for ii in tqdm(range(step)):
        a = [spearmanr(c2_train[ii,:],c1_train[x,:]) for x in range(step)]
        closest_id_ot = argmax(a)
        if closest_id_ot == ii:
            id_count_ot = id_count_ot + 1
        print("ot = {:.6f}".format(float(id_count_ot)/(ii+1)))

    df = pd.DataFrame(np.array([float(id_count_orig)/test_size,float(id_count_ot)/test_size]).reshape(1,2))
    df.to_csv(source+"_"+target+"_"+task+"_"+direction+"_id.csv")





import sys
if __name__ == "__main__":
    source = args.source
    target= args.target
    task = args.task
    is_id = args.id
    num_iters = args.num_iters
    test_size = args.test_size
    id_direction = args.id_direction

    
    n =all_data[(source,tasks[0])].shape[0] 
    ids = np.arange(n)
    np.random.shuffle(ids)
    train_test_fraction = 0.3
    train_size = int(train_test_fraction*n)
    train_index = ids[0:train_size]

    if test_size < 100:
        test_index = ids[train_size:(train_size+test_size)]
    else:
        test_index = ids[train_size:]


    if is_id:
        tasks = ["rest1","rest2"]
        for task in tasks:
            G1 = sinkhorn(source,target,train_index,task)
            source_tasks = [x for x in tasks if x!=task]  
            source_task = source_tasks[0]
            G2 = sinkhorn(source,target,train_index,source_task)
            id_rate(G1,G2,source,target,test_index,task,id_direction)

        sys.exit()

    if task == "all":
        for task in tasks:
            atlas_ot(source,target,train_index,test_index,task,num_iters)
    else:
        atlas_ot(source,target,train_index,test_index,task,num_iters)

