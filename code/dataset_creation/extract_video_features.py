import cv2 
import numpy as np
import matplotlib.pyplot as plt
import shutil
from videoClusterUtils import drawOflow, quantizeFlow, get_mask_pixels

print(cv2.__version__)

dataset = 'VIRAT'
home_dir = os.getcwd()
parent_folder = os.path.dirname(os.getcwd())
video_path = os.path.join(parent_folder, 'data', 'video', dataset)
video_file = 'VIRAT_000589.mp4'
frame_path = os.path.join(video_path, 'frames')
fg_path = os.path.join(video_path, 'foregrounds')
oflow_path = os.path.join(video_path, 'optical_flows')

cap = cv2.VideoCapture(os.path.name(video_path, video_file))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.get(cv2.CAP_PROP_FPS))
print('Length is %f' % (1000 * cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)))

ret, ctr = 1, 0
font = cv2.FONT_HERSHEY_SIMPLEX

# save foreground mask and frames 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history = 100, nmixtures = 2, backgroundRatio = 0.7, noiseSigma = 0)
while(ret):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))) # for QML
        
        #fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(frame, learningRate = -1)
        
        fgmask = cv2.GaussianBlur(fgmask,(15,15),3.5)
        fgmask = cv2.GaussianBlur(fgmask,(15,15),3.5)
        
        ret_th, fgmask_th = cv2.threshold(fgmask ,10 ,255 ,cv2.THRESH_BINARY)
        
        im2, contours, hierarchy = cv2.findContours(fgmask_th, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        
        bg_final = fgmask_th.copy()
        bg_final[:,:] = 0
        
        #cv2.drawContours(fgmask_th, contours, -1, [255, 255, 255], cv2.FILLED)
        cv2.drawContours(bg_final, contours, -1, (255,255,255), cv2.FILLED)
        
        cv2.imwrite(os.path.join(fg_path, 'FG{}.jpg'.format(ctr)), bg_final)
        cv2.imwrite(os.path.join(frame_path, 'frame{}.jpg'.format(ctr)), frame)
        
        cv2.putText(frame,'frame %d' % ctr, (10,50), font, 2,(0,0,0),2,cv2.LINE_AA)        
        cv2.imshow('frame', frame)
        cv2.imshow('fg', bg_final)
        k = cv2.waitKey(30)
        if k == 27:
            break
        ctr += 1

cap.release()
cv2.destroyAllWindows()


# optical flow time averaged over 3 frames for moving objects in each frame
old_img = cv2.imread(videoPath + 'frames/frame0.jpg')
old_gray =  cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
for frameIdx in range(1, count):
    new_img = cv2.imread(videoPath + 'frames/frame%d.jpg' % frameIdx)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(videoPath + 'foregrounds/FG%d.jpg' % frameIdx)
    mask_idx = get_mask_pixels(mask)
    flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    if frameIdx == 1:
        qFlow = quantize_flow(flow)
        qFlowX, qFlowY = qFlow[:,:,0], qFlow[:,:,1]
        oflowX = qFlowX[mask_idx[:,0], mask_idx[:,1]]
        oflowY = qFlowY[mask_idx[:,0], mask_idx[:,1]]
        oFlow = np.stack((oflowX, oflowY), axis=1)
        #np.savetxt(videoPath + 'optical_flows/flow%d.txt' % frameIdx, oFlow)
        oldFlow2 = np.copy(flow)

    elif frameIdx == 2:
        qFlow = quantize_flow(flow)
        qFlowX, qFlowY = qFlow[:,:,0], qFlow[:,:,1]
        oflowX = qFlowX[mask_idx[:,0], mask_idx[:,1]]
        oflowY = qFlowY[mask_idx[:,0], mask_idx[:,1]]
        oFlow = np.stack((oflowX, oflowY), axis=1)
        #np.savetxt(videoPath + 'optical_flows/flow%d.txt' % frameIdx, oFlow)
        oldFlow1 = np.copy(flow)
        
    else:
        flow = (flow + oldFlow1 + oldFlow2) / 3
        qFlow = quantize_flow(flow)
        qFlowX, qFlowY = qFlow[:,:,0], qFlow[:,:,1]
        oflowX = qFlowX[mask_idx[:,0], mask_idx[:,1]]
        oflowY = qFlowY[mask_idx[:,0], mask_idx[:,1]]
        oFlow = np.stack((oflowX, oflowY), axis=1)
        #np.savetxt(videoPath + 'optical_flows/flow%d.txt' % frameIdx, oFlow)
        oldFlow2 = np.copy(oldFlow1)
        oldFlow1 = np.copy(flow)
        
    old_gray = new_gray
    #print('mask shape ', mask_idx.shape, ' flow shape ', oFlow.shape)
    mask = np.zeros_like(old_img)
    mask = draw_Oflow(flow, mask, step = 5)
    finalImg = cv2.add(new_img, mask)
    cv2.imshow('frame',finalImg)
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cv2.destroyAllWindows()


# write to frames, foreground masks and optical flows into zipfile archives
# frames
os.chdir(video_path)
with ZipFile('frames.zip', 'w') as zipObj:
    for file in os.listdir('frames'):
        filePath = os.path.join('frames', file)
        #print(filePath)
        zipObj.write(filePath) 
# foregrounds
with ZipFile('foregrounds.zip', 'w') as zipObj:
    for file in os.listdir('foregrounds'):
        filePath = os.path.join('foregrounds', file)
        #print(filePath)
        zipObj.write(filePath) 
# optical flow
with ZipFile('optical_flows.zip', 'w') as zipObj:
    for file in os.listdir('optical_flows'):
        filePath = os.path.join('optical_flows', file)
        #print(filePath)
        zipObj.write(filePath) 

#delete folders after making archives
try:
    shutil.rmtree('frames')
except:
    print('Directory "frames" not found')

try:
    shutil.rmtree('foregrounds')
except:
    print('Directory "foregrounds" not found')

try:
    shutil.rmtree('optical_flows')
except:
    print('Directory "optical_flows" not found')
