
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sys
module_path = 'code' 
if module_path not in sys.path:
    sys.path.append(module_path)

# Markus's code
from MM_Plot_Utils import plot, hist
from MM_Maze_Utils import *
from MM_Traj_Utils import *


# Set the depth of the tree
depth = 6

# Recursive function to plot nodes and edges.
# It takes an extra parameter "target" to decide which node to highlight.
def plot_tree(level, num, node_list, offset, target):
    if level == depth:
        return num
    new_nodes = np.empty((2**level, 2))
    for i, (x, y) in enumerate(node_list):
        for j in range(2):
            num += 1
            new_x, new_y = x + (j * 2 - 1) * offset, y - 1
            # Draw the connecting edge
            ax.plot([x, new_x], [y, new_y], color='black')
            # If the current node number equals the target, plot it in red; otherwise in black.
            if num == target:
                ax.scatter(new_x, new_y, color='red', s=100)
            else:
                ax.scatter(new_x, new_y, color='black', s=100)
            new_nodes[i * 2 + j] = (new_x, new_y)
    num = plot_tree(level + 1, num, new_nodes, offset / 2, target)
    return num

# Create the figure and axes.
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# The update function for the animation.
# Here, the current frame number is used as the target.
def update(entry):
    ax.clear()         # Clear the current axes content
    ax.axis('off')     # Hide the axes
    timestep = entry[1]
    frame = entry[0]
    # Draw the initial root node
    if frame==0:
        ax.scatter(0, 0, color='red', s=100)
    else:
        ax.scatter(0, 0, color='black', s=100)
    # Set the current target node from the frame number.
    current_target = frame
    # Draw the tree with the current target highlighted.
    plot_tree(level=1, num=0, node_list=np.array([[0, 0]]), offset=2**4, target=current_target)
    ax.set_title(f"Timestep is: {timestep}", fontsize=16)

# Create and start the FuncAnimation.
# Here we assume a maximum of 30 nodes get drawn.
d7_tf=LoadTraj(f"D7-tf")
frames = d7_tf.no[2]
ani = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)
plt.show()