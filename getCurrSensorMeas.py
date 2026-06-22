import numpy as np

H = 28
W = 28
sensorSize = 5
dx = 0.01
dy = 0.01
# field = np.array(Image.open("data/images/test2/000001.png"))/255.0 # change this as and when needed


def sense(currPos, field, isSim):
    if isSim == False:
        # xIDX, yIDX = convertToIDX(currPos)
        currSenseActual = np.zeros((H, W))
        x = int(min(max([0,currPos[1] - np.floor(sensorSize/2)]),W)) ## coords flipped maybe
        y = int(min(max([0,currPos[0] - np.floor(sensorSize/2)]),H))

        currSenseActual[y:y+sensorSize, x:x+sensorSize] = field[y:y+sensorSize, x:x+sensorSize]

        return currSenseActual
        
    else:
        # xIDX, yIDX = convertToIDX(currPos)
        currSense = np.zeros((H, W))
        x = max(min([0,currPos[1] - sensorSize/2]),W) ## coords flipped maybe
        y = max(min([0,currPos[0] - sensorSize/2]),H)

        currSenseActual[y:y+sensorSize, x:x+sensorSize] = field[y:y+sensorSize, x:x+sensorSize]

        return currSense

def convertToIDX(currPos):
    xIDX = np.floor(currPos[0]/dx)
    yIDX = np.floor(currPos[1]/dy)

    return int(xIDX), int(yIDX)


# class getCurrSensorMeas():
#     def __init__(self):
#         H = 28
#         W = 28
#         sensorSize = 5
#         dx = 0.01
#         dy = 0.01
#         # field = np.array(Image.open("data/images/test2/000001.png"))/255.0 # change this as and when needed


#     def sense(self, currPos, field, isSim):
#         if isSim == False:
#             xIDX, yIDX = convertToIDX(currPos)
#             currSenseActual = np.zeros((H, W))
#             x = (min([0,xIDX - sensorSize/2]),W)
#             y = (min([0,yIDX - sensorSize/2]),H)

#             currSenseActual[x:x+sensorSize, y:y+sensorSize] = field[x:x+sensorSize, y:y+sensorSize]

#             return currSenseActual
            
#         else:
#             xIDX, yIDX = convertToIDX(currPos)
#             currSense = np.zeros((H, W))
#             x = (min([0,xIDX - sensorSize/2]),W)
#             y = (min([0,yIDX - sensorSize/2]),H)

#             currSense[x:x+sensorSize, y:y+sensorSize] = field[x:x+sensorSize, y:y+sensorSize]

#             return currSense

#     def convertToIDX(self, currPos):
#         xIDX = np.floor(currPos[0]/dx)
#         yIDX = np.floor(currPos[1]/dy)

#         return xIDX, yIDX