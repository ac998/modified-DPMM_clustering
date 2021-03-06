# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:13:40 2019

@author: adyasha
"""

import numpy as np
import matplotlib.pyplot as plt
from cluster_init_revised_sampling import dpmm_clustering
import time


setNum = 2 # set number
raw_data = np.loadtxt('newSet%d.csv' % setNum, usecols=(0,1)) # load dataset

#raw_data = np.loadtxt('T15_track_obs.txt', usecols=(1,2,3,4))
M = raw_data.shape[0] # number of points

############################# specify parameters ######################################
beta = 90 # setting parameters
r = 20 # learning rate

###################### plot original data ##########################################
#print(raw_data.shape)
fig = plt.figure(figsize = (8.0, 8.0))
plt.plot(raw_data[:,0], raw_data[:,1], 'k.')
plt.show()


############################# add index and initial label ##############################
idx = np.arange(M, dtype=np.int16)
initial_labels = np.full((M, 1), None)
dataset = np.hstack((np.reshape(idx, (M, 1)), raw_data, initial_labels))
dataset = dataset.astype(np.float32)
print(dataset.shape)
#print(dataset[1:5])


clustering = dpmm_clustering(dataset, beta = beta, r = r, max_cluster=200)
total_t = 0


############################################# stage 1 - find initial number of clusters ###################################################
t0 = time.time()
d_set1, labels, final_counts, final_centres = clustering.assign_labels()
t1 = time.time()
total_t += (t1 - t0)

########### plot after stage 1 ###############
print("After stage 1....\n")
np.random.seed(2021)
Colors = np.random.rand(len(labels),3)

fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = final_centres[:,0].astype(int), final_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
#print("stage 1 : ",(t1 - t0)*10**6)
plt.show()

######################################################### stage 2 - pre convergence ######################################################
t0 = time.time()
d_set1, labels, final_counts, final_centres, indices = clustering.assign_labels(label_ids=labels, prev_counts = final_counts, prev_centres = final_centres)
t1 = time.time()
total_t += (t1 - t0)

############ plot at stage 2 - before convergence #############
print("After stage 2, before convergence....\n")
fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = final_centres[:,0].astype(int), final_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
#plt.show()

########################################################## stage 2 - convergence ########################################################
# numIter = 10 # max no of iterations to converge
t0 = time.time()
d_set1, labels, final_counts, final_centres = clustering.assign_labels(label_ids=labels, prev_counts = final_counts, prev_centres = final_centres, converge=True, numIter=10, indices=indices)
t1 = time.time()
print("stage 2 - after convergence : ",t1 - t0)
print(len(labels))
print(labels)
print(final_counts)

total_t += (t1 - t0)

fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = final_centres[:,0].astype(int), final_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
plt.show()

print("Total time (in ms) = ", total_t * 1000)