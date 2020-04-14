import numpy as np
import matplotlib.pyplot as plt
from cluster_init_revised_sampling import modified_dpmm
import time


setNum = 2 # set number
raw_data = np.loadtxt('newSet%d.csv' % setNum, usecols=(0,1)) # load dataset

#raw_data = np.loadtxt('T15_track_obs.txt', usecols=(1,2,3,4))
M = raw_data.shape[0] # number of points


# specify parameters 
beta = 90 # initial search radius
r = 20 # learning rate



# plot unclustered points
#print(raw_data.shape)
fig = plt.figure(figsize = (8.0, 8.0))
plt.plot(raw_data[:,0], raw_data[:,1], 'k.')
plt.title("Unclustered points")
plt.show()


# add index and initial label 
idx = np.arange(M, dtype=np.int16)
initial_labels = np.full((M, 1), None)
dataset = np.hstack((np.reshape(idx, (M, 1)), raw_data, initial_labels))
dataset = dataset.astype(np.float32)
print(dataset.shape)
#print(dataset[1:5])


clustering = modified_dpmm(dataset, beta = beta, r = r, max_cluster=200)
total_t = 0


# stage 1 - find initial number of clusters 
t0 = time.time()
result = clustering.assign_labels()
t1 = time.time()
total_t += (t1 - t0)

# plot after stage 1
print("After stage 1....\n")
np.random.seed(2021)
Colors = np.random.rand(len(result.cluster_labels),3)

fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = result.cluster_centres[:,0].astype(int), result.cluster_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
plt.title("Stage 1")
#print("stage 1 : ",(t1 - t0)*10**6)
plt.show()

# stage 2 - pre convergence 
t0 = time.time()
result = clustering.assign_labels(label_ids=result.cluster_labels, prev_counts = result.cluster_counts, prev_centres = result.cluster_centres)
t1 = time.time()
total_t += (t1 - t0)

# plot at stage 2 - before convergence 
print("After stage 2, before convergence....\n")
fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = result.cluster_centres[:,0].astype(int), result.cluster_centres[:,1].astype(int)
plt.title("Before convergence of stage 2")
plt.scatter(cx, cy, marker='x', color='k')
plt.show()

# stage 2 - convergence 
# numIter = 10 # max no of iterations to converge
t0 = time.time()
result = clustering.assign_labels(label_ids=result.cluster_labels, prev_counts = result.cluster_counts, prev_centres = result.cluster_centres, converge=True, maxIter=10)

t1 = time.time()
total_t += (t1 - t0)

# plot at stage 2 - before convergence 
print("stage 2 - after convergence : ",t1 - t0)

fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,3] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = result.cluster_centres[:,0].astype(int), result.cluster_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
plt.title("After convergence of stage 2")
plt.show()

print("Total time (in ms) = ", total_t * 1000)