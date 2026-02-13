import numpy as np
import re

# Initialize storage for each point
points_data = {}

with open("points.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Replace 'array([...])' with '[...]'
        line = re.sub(r"array\((\[.*?\])\)", r"\1", line)
        # Remove trailing commas
        line = re.sub(r",\s*$", "", line)
        # Evaluate the dictionary
        data_dict = eval(line)
        
        # Accumulate data per key
        for k, v in data_dict.items():
            points_data.setdefault(k, []).append(np.array(v))

# Compute average per coordinate for each point
averages_per_point = {k: np.mean(np.stack(vs), axis=0) for k, vs in points_data.items()}

# Print nicely
for k, avg in averages_per_point.items():
    print(f"Point {k} average (x, y, z): {avg}")
