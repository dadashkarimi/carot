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
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

import sys

import argparse

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{0} is not a valid path".format(path))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s','--source',type=str,help="source atlas")
parser.add_argument('-t','--target',type=str,help="target atlas")
parser.add_argument('-task','--task',type=str,default="rest1",help="task")
parser.add_argument('-m','--mapping',help="path to mapping")

args = parser.parse_args()


#argv = sys.argv[1:]
def shuffle_list(l):
    ids = np.arange(len(l))
    np.random.shuffle(ids)
    print(ids)
    return np.array(l)[ids]




atlases = ['dosenbach','schaefer','brainnetome','power','craddock','shen','shen368','craddock400']
atlases = ['dosenbach','schaefer','brainnetome','power','shen','craddock']
atlases = shuffle_list(atlases)
tasks = ["rest1","gambling","wm","motor","lang","social","relational","emotion"]
tasks = shuffle_list(tasks)


path = '/data_dustin/store4/Templates/HCP'

coord = {}
all_data = {}
coord['schaefer'] =pd.read_csv('/data_dustin/store4/Templates/schaefer_coords.csv', sep=',',header=None)
coord['brainnetome'] =pd.read_csv('/data_dustin/store4/Templates/brainnetome_coords.csv', sep=',',header=None)
coord['shen'] =pd.read_csv('/data_dustin/store4/Templates/shen_coords.csv', sep=',',header=None)
coord['shen368'] =pd.read_csv('/data_dustin/store4/Templates/shen_368_coords.csv', sep=',',header=None)
coord['power'] =pd.read_csv('/data_dustin/store4/Templates/power_coords.txt', sep=',',header=None)
coord['dosenbach'] =pd.read_csv('/data_dustin/store4/Templates/dosenbach_coords.txt', sep=',',header=None)
coord['craddock'] =pd.read_csv('/data_dustin/store4/Templates/craddock_coords.txt', sep=',',header=None)
coord['craddock400'] =pd.read_csv('/data_dustin/store4/Templates/craddock_400_coords.txt', sep=',',header=None)

# Loading Atlas ...
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
        print(atlas,task,x.shape)
        np.delete(x,list(zero_nodes),1)
        all_data[(atlas,task)] = x


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


       
from collections import defaultdict
def atlas_ot(source,target,task):
    print("Atlas OT .. ")
    n =all_data[(source,task)].shape[0] 
    t =all_data[(source,task)].shape[2] 
    p1 =all_data[(source,task)].shape[1] 
    p2 =all_data[(target,task)].shape[1] 
    n2 =all_data[(target,task)].shape[0] 
    num_time_points = t
    
    mapping_file = args.mapping#os.chdir(args.mapping)
    G = genfromtxt(mapping_file,delimiter=',',skip_header=1,usecols=range(1,1+coord[args.target].shape[0]))

    test_time_series_pred = np.zeros((n2,p2,num_time_points))
    test_time_series  = []
    step = 0
    id_count_ot = 0
    id_count_orig = 0
    id_conn = {}
    for i in tqdm(range(n2),desc="applying OT .."):
        for j in range(num_time_points):

            a = all_data[(source,task)][i,:,j]
            a = a + 1e-6
            a = normalize(a)
            b= np.transpose(G).dot(a)
            test_time_series_pred[i,:,j] = b
        test_time_series.append(test_time_series_pred[i,:,:])

    return G, test_time_series_pred


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
 
       

from scipy.stats.stats import spearmanr
import sys

if __name__ == "__main__":
    source = args.source
    target= args.target
    task = args.task
    random.seed(3000)
    
    n =all_data[(atlases[0],tasks[0])].shape[0] 
    G, test_time_series_pred = atlas_ot(source,target,task)
    data = {"data":test_time_series_pred}
    sio.savemat("time_series_"+target+"_ot.mat",data)
    #df = pd.DataFrame(test_time_series_pred)
    #df = pd.Panel().to_frame(),stacj().reset_index()#DataFrame(test_time_series_pred)
    #df.to_csv("time_series_"+target+"_ot.csv")

