---
title: Process Explanation
date: 2026-02-24
---

The gist of the program is as follows:

1. The program detects marker IDs and corners from a camera frame using `detect_markers`.  
2. Based on the IDs and corners, marker coordinates in camera coordinates are calculated using OpenCV's `estimatePoseSingleMarkers` function via `get_marker_positions`. 

    From now on, the problem is reduced to a series of linear algebra calculations.

3. Based on the coordinates, the aim direction can be calculated using `compute_marker_aim`.  
4. Finally, the intersection with the aim plane can be computed using `compute_aim_intersection`.  

After computing the intersection, it can be translated into screen coordinates.

**Step-by-Step Vector Calculations**

Compute the midpoint of the left and right markers:

$$
P_{mid} = \frac{P_{right} + P_{left}}{2}
$$

Compute the normalized side vector between the left and right rear markers:

$$
V_{side} = \frac{P_{right} - P_{left}}{\||P_{right} - P_{left}\||}
$$

Compute the normalized forward vector from the rear midpoint toward the forward marker:

$$
V_{forward} = \frac{P_{front} - P_{mid}}{\||P_{front} - P_{mid}\||}
$$

Compute the normalized up vector via the cross product of the side and forward vectors:

$$
V_{up} = \frac{V\_{side} \times V\_{forward}}{\||V\_{side} \times V\_{forward}\||}$$

Compute the virtual barrel base position by offsetting the rear midpoint along the up vector:

$$
P_{barrel} = P_{mid} - h \cdot V_{up}
$$

Compute the normalized barrel direction vector pointing from the virtual barrel base to the forward marker:

$$
V\_{barrel\\_aim} = \frac{P\_{front} - P\_{barrel}}{\||P\_{front} - P\_{barrel}\||}
$$

The intersection occurs at a scalar $t$ along the aim line defined by its origin $P_{barrel}$ and direction vector $V_{barrel\\_aim}$:

$$
t = \frac{Z_{plane} - P_{barrel, z}}{V_{barrel\\_aim, z}}
$$

Finally an intersection point with the plane can be found using the parameter $t$:

$$
P_{intersection} = P_{barrel} + t \cdot V_{barrel\\_aim}
$$