import math
import time
import threading
import airsim
import numpy as np
from moveCtrl import *
import airsim.utils as utils
import cv2 as cv

# necessary varibles
_WIDTH = 640
_HEIGHT = 480
_THROUGH = False
_order = 1
_end = False
cfg = "./yolov4.cfg"
weights = "./yolov4.weights"
img = None
scene = None


# main interpreter and analyzing class
class dbox(object):
    def __init__(self,box):
        a,b,c,d = box
        self.x = a
        self.y = b
        self.w = c
        self.h = d
        self.center = (self.x+self.w/2,self.y+self.h/2)
        self.y_bias = self.center[0] - _WIDTH/2
        self.z_bias = self.center[1] - _HEIGHT/2
class analyzer(object):
    def __init__(self):
        self.points = (
            (15.5, -19.6, -3.5, 5),
            (22, -41.2, -2.5, 5),
            (21, -61.5, -2, 3),
            (10, -78.2, -2, 3),
            (-9.3, -93, -2.5, 3),
            (-27, -98, -4, 3),
            (-50.1, -103, -5.7, 3)
        )
        self.yaws = (-90,-90, -100, -120, -170, -170, -180)
        self.mbox = None
    def InCenter(self):
        if self.mbox.center[0] - _WIDTH/2 < 20 and self.mbox.center[1] - _HEIGHT/2 < 15:
            return True
        else:
            return False
    def Adjust(self,mbox):
        self.mbox = mbox
        if max(self.mbox.w,self.mbox.h)==0:
            pass # do something 未检测到圈
        v_y, v_z = (self.mbox.y_bias,self.mbox.z_bias)
        # speed = max(self.mbox.h,self.mbox.w)
        client.moveByVelocityBodyFrameAsync(2,v_y*0.01,v_z*0.015,0.9)
def findBiggest(boxes):
    if len(boxes )==0:
        return [0,0,0,0]
    mbox = boxes[0]
    for box in boxes:
        if max(box[2] ,box[3]) >max(mbox[2] ,mbox[3]):
            mbox = box
    return mbox

# for constant img request
def askForImg():
    global img
    img = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)])

# for constant img decoding
def decodeImg():
    global img
    global scene
    while(True):
        try:
            scene = np.frombuffer(img[0].image_data_uint8, dtype=np.uint8)
            scene = scene.reshape(480, 640, 3)
        except:
            print('img empty')
# main
def main():
    global client
    global scene
    ana = analyzer()
    while (1):
        cv.imshow('a', scene)
        cv.waitKey(1)
        classes, confidences, boxes = net.detect(scene, confThreshold=0.8, nmsThreshold=0.4)
        mbox = findBiggest(boxes)
        ana.Adjust(dbox(mbox))

net = cv.dnn_DetectionModel(cfg, weights)
net.setInputSize(416 ,416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

client.moveToPositionAsync(15.5, -12.6 , -3.5, 5).join()
client.hoverAsync().join()
time.sleep(3)
client.rotateToYawAsync(-90).join()

try:
    ask = threading.Thread(target=askForImg)
    dec = threading.Thread(target=decodeImg())
    ask.start()
    dec.start()
except:
    print('img threads not established')

client.landAsync().join()