import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import sys
module_path = 'code' 
if module_path not in sys.path:
    sys.path.append(module_path)

# Markus's code
from MM_Plot_Utils import plot, hist
from MM_Maze_Utils import *
from MM_Traj_Utils import *

# Draw maze with node numbers and grayed out blocks
from matplotlib.collections import LineCollection
from matplotlib import cm



# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Add the rectangle to the axis so that it appears on the plot
d7_tf=LoadTraj(f"D7-tf").ce[2]
ma = NewMaze(6)
for i, item in enumerate(d7_tf):

    points = Rectangle((ma.xc[item], ma.yc[item]), 1, 1, facecolor='red')
    ax.add_patch(points)

plt.show()