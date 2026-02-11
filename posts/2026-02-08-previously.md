---
title: Previously
date: 2026-02-08
---

My vision is to create an application where you can practice hunting. To do this I need to finish the first part of the plan, which is to create a way to detect where a user is pointing their gun on their screen.

To do this I decided to use OpenCV to detect so called aruco codes placed strategically on the gun.

A majority of this first step is completed. I have 3 views:
- 3d-view -- should be a projection of the markers in 3d space with a red "aim line". All of this is calculated.
- 2d-view -- shows the aim point on the screen.
- cam-view -- debug view for the camera. 

![Example view](/img/example.png)
*Figure: Camera view with detected ArUco markers*

**Todo**
* I need to verify the math, as it's wrong. When you aim at one point and then rotate the gun, the aim point moves, and the further to the side you aim the lower the aim point gets interpreted. I need to recalculate the math. I will present it here in the future.

* Apparently i can increse the camera view resolution, this increses code detection accuracy.

* I need to redo the codes, they're hand drawn and i marked the top left corner to know which one it is, but this has decresed acuracy because the CV gets confused by that mark.
