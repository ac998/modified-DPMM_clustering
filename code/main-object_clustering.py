import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from PIL import Image
import time 
from videoClusterUtils import get_pixels
from cluster_init_revised_sampling import modified_dpmm


# specify parameters 
beta = 40
r = 18
randomize = False



# load foreground pixels and corresponding optical flow for given frame 
frame_no = 6030
set_name = 'VIRAT'
parent_folder = os.path.dirname(os.getcwd())
foreground_file = 'foregrounds/FG{}.jpg'.format(frame_no)
optical_flow_file = 'optical_flows/flow{}.txt'.format(frame_no)
data_folder_path = os.path.join(parent_folder, 'data', 'video', set_name)


with ZipFile(os.path.join(data_folder_path, 'foregrounds.zip')) as myzip:
    file = myzip.open(foreground_file)
    raw_img_mask = Image.open(file)
    print(raw_img_mask.size)
    #raw_img_mask.show()
    img_mask = np.array(raw_img_mask)
    print(img_mask.shape)

with ZipFile(os.path.join(data_folder_path, 'optical_flows.zip')) as myzip:
    file = myzip.open(optical_flow_file)
    opt_flow = np.loadtxt(file, dtype=np.float32)
    #print(opt_flow.shape)

pixels = get_pixels(img_mask)
pixels = pixels.astype(np.float32)
print(pixels.shape)



# plot unclustered points 
fig = plt.figure(figsize=(8,8))
plt.scatter(pixels[:,0], pixels[:,1], marker='.')
plt.show()

# add extra columns for labels and indexes
M = pixels.shape[0]
idx = np.arange(M, dtype=np.int16)
initial_labels = np.full((M, 1), None)
dataset = np.hstack((np.reshape(idx, (M, 1)), pixels, opt_flow, initial_labels))
dataset = dataset.astype(np.float32)
N = dataset.shape[1]
print(dataset.shape)

# whether to process the frame scan line wise or randomly
if randomize:
    rand_indexes = np.arange(M)
    np.random.shuffle(rand_indexes)
    dataset = dataset[rand_indexes]

# stage 1 - find initial number of clusters 
clustering = modified_dpmm(dataset, beta = beta, r = r, max_cluster=200)
total_t = 0


t0 = time.time()
result = clustering.assign_labels()
t1 = time.time()
total_t += (t1 - t0)

#plot
print("After stage 1....\n")
np.random.seed(2021)
Colors = np.random.rand(len(result.cluster_labels),3)


fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,-1] == c]
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

#plot
print("After stage 2, before convergence....\n")
fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,-1] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = result.cluster_centres[:,0].astype(int), result.cluster_centres[:,1].astype(int)
plt.title("Before convergence of stage 2")
plt.scatter(cx, cy, marker='x', color='k')
plt.show()



# stage 2 - convergence 
t0 = time.time()
result = clustering.assign_labels(label_ids=result.cluster_labels, prev_counts = result.cluster_counts, prev_centres = result.cluster_centres, converge=True, numIter=10)

t1 = time.time()
print("stage 2 - after convergence : ",t1 - t0)
total_t += (t1 - t0)

# plot
fig = plt.figure(figsize = (8.0,8.0))
for i,c in enumerate(result.cluster_labels):
    points = dataset[dataset[:,-1] == c]
    plt.scatter(points[:,1], points[:,2], marker = '.', color=Colors[i])
    
cx, cy = result.cluster_centres[:,0].astype(int), result.cluster_centres[:,1].astype(int)
plt.scatter(cx, cy, marker='x', color='k')
plt.title("After convergence of stage 2")
plt.show()

print("Total time (in ms) = ", total_t * 1000)
np.savetxt('clustering_result.csv' , dataset, delimiter='\t', fmt='%3.2f')
