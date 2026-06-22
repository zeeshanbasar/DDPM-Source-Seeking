import numpy as np
import getCurrSensorMeas as gsm

H = 28
W = 28
dx = 0.01
dy = 0.01
searchArea = np.zeros((H, W)).astype(np.float32)
updatedBel = np.zeros((H, W)).astype(np.float32)

def updateSearchArea(currPos, field, prevBel):

    currSenseActual = gsm.sense(currPos, field, isSim = False)
    updatedBel = prevBel.copy()
    zero_mask = (prevBel == 0)
    updatedBel[zero_mask] = currSenseActual[zero_mask]

    return updatedBel

def createMask(updatedBel):
    newMask = (updatedBel == 0) ## maybe flipped
    newMask = 1 - np.array(newMask).astype(int)

    return newMask

def convertToIDX(currPos):
    xIDX = np.floor(currPos[0]/dx)
    yIDX = np.floor(currPos[1]/dy)

    return int(xIDX), int(yIDX)

# class createCurrBel():
#     def __init__(self):
#         H = 28
#         W = 28
#         searchArea = np.zeros((H, W)).astype(np.float32)
#         updatedBel = np.zeros((H, W)).astype(np.float32)

#     def updateSearchArea(self, currPos, field, prevBel = None):
#         if prevBel == None:
#             prevBel = np.zeros((H, W)).astype(np.float32)

#         currSenseActual = gsm.getCurrSensorMeas.sense(currPos, field, isSim = False)
#         updatedBel = prevBel.copy()
#         zero_mask = (prevBel == 0)
#         updatedBel[zero_mask] = currSenseActual[zero_mask]

#         return updatedBel
    
#     def createMask(self):
#         newMask = (updatedBel == 0)

#         return newMask

#     def convertToIDX(self, currPos):
#         xIDX = np.floor(currPos[0]/dx)
#         yIDX = np.floor(currPos[1]/dy)

#         return xIDX, yIDX