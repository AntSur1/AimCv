---
title: Process explanation
date: 2026-02-24
---

The gist of the program is as follows:

1. The program detects marker IDs and corners from a camera frame using `detect_markers`.  
2. Based on the IDs and corners, marker coordinates in camera coordinates are calculated using OpenCV's `solvePnP` function via `get_marker_positions`. 

    From now on, the problem is reduced to a series of linear algebra calculations.

3. Based on the coordinates, the aim direction can be calculated using `compute_marker_aim`.  
4. Finally, the intersection with the aim plane can be computed using `compute_aim_intersection`.  

After computing the intersection, it can be translated into screen coordinates.

**Step-by-Step Vector Calculations**

Compute the midpoint of the left and right markers:

$$
P_{mid} = \frac{P_{left} + P_{right}}{2}
$$

Compute the normalized side vector between the left and right rear markers:

$$
\mathbf{V}_{side} = \frac{P_{right\_rear} - P_{left\_rear}}{\|P_{right\_rear} - P_{left\_rear}\|}
$$

Compute the normalized forward vector from the rear midpoint toward the forward marker:

$$
\mathbf{V}_{forward} = \frac{P_{front} - P_{mid}}{\|P_{front} - P_{mid}\|}
$$

Compute the normalized up vector via the cross product of the side and forward vectors:

$$
\mathbf{V}_{up} = \frac{\mathbf{V}_{side} \times \mathbf{V}_{forward}}{\|\mathbf{V}_{side} \times \mathbf{V}_{forward}\|}
$$

Compute the virtual barrel base position by offsetting the rear midpoint along the up vector:

$$
P_{barrel} = P_{mid} - h \cdot \mathbf{V}_{up}
$$

Compute the normalized barrel direction vector pointing from the virtual barrel base to the forward marker:

$$
\mathbf{V}_{barrel\_aim} = \frac{P_{front} - P_{barrel}}{\|P_{front} - P_{barrel}\|}
$$

The intersection occurs at a scalar $t$ along the aim line defined by its origin $P_{barrel}$ and direction vector $\mathbf{V}_{barrel\_aim}$:

$$
t = \frac{Z_{plane} - P_{barrel, z}}{\mathbf{V}_{barrel\_aim, z}}
$$

Using the parameter $t$, the intersection point with the plane is:

$$
P_{intersection} = P_{barrel} + t \cdot \mathbf{V}_{barrel\_aim}
$$

