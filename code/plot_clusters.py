import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image


frame_no = 6030
set_name = 'VIRAT'
parent_folder = os.path.dirname(os.getcwd())
data_folder_path = os.path.join(parent_folder, 'data', 'video', set_name)

frame_file = 'frames/frame{}.jpg'.format(frame_no)
result_file= 'clustering_result.csv'


with ZipFile(os.path.join(data_folder_path, 'frames.zip')) as myzip:
    file = myzip.open(frame_file)
    orig_frame = Image.open(file)
    print(orig_frame.size)
    orig_frame = np.array(orig_frame)


x_max = orig_frame.shape[1]
y_max = orig_frame.shape[0]


def plot_cluster(cluster_file_path, image):
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x_raw, y_raw, label_raw = np.loadtxt(cluster_file_path, usecols=(1, 2, 5), unpack=True, dtype='float')

    y_coord = x_raw.astype(int)
    x_coord = y_raw.astype(int)
    label_array = label_raw.astype(int)

    #adjust outofbound values
    x_coord[x_coord >= x_max] = x_max-1
    x_coord[x_coord >= x_max] = x_max-1
    y_coord[y_coord >= y_max] = y_max-1
    y_coord[y_coord >= y_max] = y_max-1
    x_coord[x_coord < 0] = 0
    x_coord[x_coord < 0] = 0
    y_coord[y_coord < 0] = 0
    y_coord[y_coord < 0] = 0

    num_data = len(x_coord)
    #print (num_data)
    t = np.arange(len(x_coord))
    hue = []
    sat = []
    val = []
    unique_labels = np.unique(label_array)
    max_color = len(unique_labels)
    #print (max_color)
   
    hue[:] = [(np.where(unique_labels == label_array[x])[0][0])*170.0/max_color for x in t]
    sat[:] = [255 for x in t]
    val[:] = [255 for x in t]

    hsv_frame[y_coord[:],x_coord[:] ,0] = hue[:]
    hsv_frame[y_coord[:],x_coord[:] ,1] = sat[:]
    hsv_frame[y_coord[:],x_coord[:] ,2] = val[:]

    image = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    bgrcopy = image[:, :, :].copy() 
    return image


def blend_plot_with_frame(cluster_plot, frame, beta=0.5):
    alpha = 1 - beta
    blended_frame = cv2.addWeighted(cluster_plot, alpha, frame, beta, 0.0)
    return blended_frame


plotted_frame = plot_cluster(result_file, orig_frame)
cv2.imshow('plotted_frame', plotted_frame)
cv2.waitKey(0)

blended_frame = blend_plot_with_frame(plotted_frame, orig_frame)
cv2.imshow('blended_frame', blended_frame)
cv2.waitKey(0)

cv2.destroyAllWindows()