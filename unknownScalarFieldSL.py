import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import approxSearchArea as appr
import createCurrBel as cb
import takeOptimStep as optim

currPos = [20,20]
prevPos = [20,20]
posList = []
field = np.array(Image.open("data/images/test2/009880.png"))/255.0 # change this as and when needed
H = 28
W = 28

pk = np.unravel_index(np.argmax(field),(H,W))

currBel = np.zeros((H, W)).astype(np.float64)

parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
parser.add_argument('--config', dest='config_path',
                    default='config/default.yaml', type=str)
args = parser.parse_args()

numSteps = 0
while(1):
    currBel = cb.updateSearchArea(currPos=currPos, field=field, prevBel=currBel)
    currMask = cb.createMask(currBel)


    simField = appr.infer(args, currKnown=currBel, currMask=currMask)
    simField = simField[0][0]

    prevPos = currPos
    posList.append(prevPos)
    _,_,optimPath = optim.multiscale_maximum(simField,currPos)
    currPos = optimPath[-1]
    # currPos = optim.step(currPos=currPos, field=simField)
    print(f"\ncurrent postion: {currPos}")
    numSteps += 1

    if np.linalg.norm(currPos-prevPos) < 0.005 or numSteps > 20:
        break

print(f"\nActual peak = {pk}\n")

fig, ax = plt.subplots()

im = ax.imshow(field)

x,y = [], []
for i in range(len(posList)):
    x.append(int(np.floor(posList[i][1])))
    y.append(int(np.floor(posList[i][0])))

ax.plot(x, y, 
        linestyle='--', 
        color='red', 
        linewidth=2, 
        alpha=0.8,
        label='Path')

# Plot circles at each point
ax.scatter(x, y, 
          s=100,           # Size of circles
          c='white',       # Fill color
          edgecolors='red', # Edge color
          linewidth=2,     # Edge width
          alpha=0.9,
          label='Points',
          zorder=5)        # Ensure circles are on top

