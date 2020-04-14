import cv2
import numpy  as np

def get_mask_pixels(mask):
	# get the object pixels from the foreground
    maskSlice = mask[:,:,0]
    #maskSlice = mask
    x, y = np.nonzero(maskSlice)
    dataset = np.stack((x, y), axis=1)
    return dataset

def get_pixels(mask):
    # get the object pixels from the foreground
    #maskSlice = mask[:,:,0]
    maskSlice = mask
    x, y = np.nonzero(maskSlice)
    dataset = np.stack((x, y), axis=1)
    return dataset

def draw_Oflow(flow, mask, step):
	# draw optical flow vectors
    I,J = flow.shape[1], flow.shape[0]
    for x in range(0, flow.shape[1], step):
        for y in range(0, flow.shape[0], step):
            mask = cv2.line(mask, (x,y), (int(x + 0.5 + flow[y,x,0]), int(y + 0.5 + flow[y,x,1])), [0,255,0])
    return mask


def quantize_flow(flow):
	
    qFlow = np.zeros(flow.shape)
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
    
    # between 0 and 45
    idx = np.where((ang >= 0) & (ang < 45))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = 2, 0
    # between 45 and 90
    idx = np.where((ang >= 45) & (ang < 90))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = np.sqrt(2), np.sqrt(2)
    # between 90 and 135
    idx = np.where((ang >= 90) & (ang < 135))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = 0, 2
    # between 135 and 180
    idx = np.where((ang >= 135) & (ang < 180))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = -np.sqrt(2), np.sqrt(2)
    # between 180 and 225
    idx = np.where((ang >= 180) & (ang < 225))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = -2, 0
    # between 225 and 270
    idx = np.where((ang >= 225) & (ang < 270))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = -np.sqrt(2), -np.sqrt(2)
    # between 270 and 315
    idx = np.where((ang >= 270) & (ang < 315))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = 0, -2
    # between 315 and 360
    idx = np.where((ang >= 315) & (ang < 360))
    qFlow[:,:,0][idx], qFlow[:,:,1][idx] = np.sqrt(2), -np.sqrt(2)
    
    return qFlow
