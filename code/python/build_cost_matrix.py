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
from scipy.spatial.distance import cdist

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
import os
import numpy as np
import sys
import argparse
import ConfigParser

config = ConfigParser.RawConfigParser()
config.read('config.properties')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s','--source',type=str,help="source atlas")
parser.add_argument('-s1','--source1',type=str,help="source atlas 1")
parser.add_argument('-s2','--source2',type=str,help="source atlas 2")
parser.add_argument('-t','--target',type=str,help="target atlas")
parser.add_argument('-task','--task',type=str,default="rest1",help="task")
parser.add_argument('-c','--cost',type=str,default="functional",help="cost matrix")
parser.add_argument('-id','--id',type=bool,default=False,help="id rate")
parser.add_argument('-save_model','--save_model',type=bool,default=False,help="saving OT model")
parser.add_argument('-use_mean_map','--use_mean_map',type=bool,default=False,help="Use the avrage OT model across tasks")
parser.add_argument('-intrinsic','--intrinsic',type=bool,default=False,help="intrinsic evaluation")
parser.add_argument('-simplex','--simplex',type=int,default=2,help="simplex evaluation")
parser.add_argument('-cross_task','--cross_task',type=str,default=None,help="cross task optimal transport")
parser.add_argument('-num_iters','--num_iters',default=100,type=int,help="number of iterations")
parser.add_argument('-test_size','--test_size',default=100,type=int,help="test size")
parser.add_argument('-k','--k',default=0,type=int,help="K nearest neighbour argument")
parser.add_argument('-lamda','--lamda',default=0.5,type=float,help="Lambda, linear interpolation between func and euc distance")
parser.add_argument('-sample_atlas','--sample_atlas',default=0,type=int,help="leave an atlas out")
parser.add_argument('-id_direction','--id_direction',type=str,default="ot-ot",help="direction of id rate: orig-orig ot-ot orig-ot ot-orig")

args = parser.parse_args()


#argv = sys.argv[1:]
def shuffle_list(l):
    ids = np.arange(len(l))
    np.random.shuffle(ids)
    print(ids)
    return np.array(l)[ids]




atlases = ['dosenbach','schaefer','brainnetome','power','shen','craddock']
atlases = shuffle_list(atlases)
tasks = ["rest1"]#,"gambling","wm","motor","lang","social","relational","emotion"]
tasks = shuffle_list(tasks)


coord = {}
all_data = {}

# Loading Atlas ...
tasks = [args.task]
atlases = [args.source,args.target] 

dataset_path = {}
for atlas in tqdm(atlases,desc = 'Loading config file ..'):
    dataset_path[atlas]=config.get('path',atlas)
    coord[atlas]= pd.read_csv(config.get('coord',atlas), sep=',',header=None)

for atlas in tqdm(atlases,desc = 'Loading Atlases ..'):
    zero_nodes = set()

    data = sio.loadmat(dataset_path[atlas])
    for task in tasks:
        x = data['all_mats']
        idx = np.argwhere(np.all(x[..., :] == 0, axis=0))
        p = [p1  for (p1,p2) in idx]
        zero_nodes.update(p)

    for task in tasks:
        x = data['all_mats']
        print(atlas,task,x.shape)
        np.delete(x,list(zero_nodes),1)
        all_data[(atlas,task)] = x

        data = {"all_mats":x[0:10,:,:]}
        sio.savemat(atlas+".mat",data)



all_behav = genfromtxt(config.get('behavior','iq'), delimiter=',')
all_sex = genfromtxt(config.get('behavior','gender'), delimiter=',')


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


       
def one_way_ot(source,target,task):
    from collections import defaultdict
    train_index = range(1,n)
    num_time_points =all_data[(source,task)].shape[2] 
    M =  build_cost(source,target,train_index,task,num_time_points)
    df = pd.DataFrame(M)
    df.to_csv("cost_"+source+"_"+target+".csv")

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
 
       

def build_cost(source,target,train_index,task,frame_size):

    from collections import defaultdict



    n =all_data[(source,task)].shape[0] 
    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    num_time_points = t


    train_size = len(train_index)


    

    G = np.zeros((num_time_points,p1,p2))
    M = ot.dist(coord[source],coord[target],metric='euclidean')

    if args.cost == "interpolate":
        M1 = ot.dist(coord[source],coord[target],metric='euclidean')
        M1 = (M1 - np.min(M1))/np.ptp(M1)
        
        lambd = 0.001
        M2 = np.zeros((p1,p2))
        for i in train_index:
            a = all_data[(source,task)][i,:,:]
            b = all_data[(target,task)][i,:,:]

            Mi = generate_correlation_map(a,b)
            Mi = (Mi - np.min(Mi))/np.ptp(Mi)
            Mi = 1-Mi
            M2 = M2 + Mi
        M2 = M2/train_size
        M = args.lamda*M1 + (1-args.lamda)*M2

    elif args.cost == "euclidean":
        lambd = 0.01
        M = ot.dist(coord[source],coord[target],metric='euclidean')
        if args.k>0:
            T = np.ones((M.shape[0],M.shape[1]))#+1e-6
            for ii in range(M.shape[0]):
                index = np.argsort(M[ii,:])[:args.k]
                T[ii,index] = 0#float(1.0/args.k)
            M = T
        M = (M - np.min(M))/np.ptp(M)

    elif args.cost == "functional" or args.cost == "vae":
        lambd = 0.001
        M = np.zeros((p1,p2))
        for i in train_index:
            a = all_data[(source,task)][i,:,:]
            b = all_data[(target,task)][i,:,:]

            Mi = generate_correlation_map(a,b)
            Mi = (Mi - np.min(Mi))/np.ptp(Mi)
            Mi = 1-Mi
            M = M + Mi
        M = M/train_size

    return M

from scipy.stats.stats import spearmanr
import sys

if __name__ == "__main__":
    source = args.source
    target= args.target
    task = args.task
    is_id = args.id
    num_iters = args.num_iters
    test_size = args.test_size
    id_direction = args.id_direction
    cross_task = args.cross_task
    random.seed(3000)
    
    n =all_data[(atlases[0],tasks[0])].shape[0] 
    one_way_ot(source,target,task)

