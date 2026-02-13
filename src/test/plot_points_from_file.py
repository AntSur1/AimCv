import numpy as np
import re
import matplotlib.pyplot as plt

# Collect all points
all_points = []

with open("points.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"array\((\[.*?\])\)", r"\1", line)
        line = re.sub(r",\s*$", "", line)
        data_dict = eval(line)
        for v in data_dict.values():
            all_points.append(np.array(v))

all_points = np.stack(all_points)  # shape (num_points, 3)

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], s=10, c='blue', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('All 600 Points')
plt.show()
