---
title: Aim Reliability
date: 2026-02-22
---

Turns out the math was correct, just that the cross product elements were calculated in the wrong order. Now, rotating the gun while aiming at a point won't move the point. The issue with aiming to the sides still persist, and I still suspect the  culprit to be the fish eye distortion. 

Another issue is that the aim point jumps a lot. The aim jitter is created by camera limitations. Increasing the camera resolution and having a good light might help. I will need to redo some calibration for this though, and print out new markers, as the current ones are hand drawn.

Another solution would be to move the camera from a position right in front of the barrel. OpenCV has a hard time to detect small angles of the markers, which might affect the exact position of the markers. So, moving the camera to have more of a side view of the markers will increase the angle and therefore be more reliable. This will require future experimentation.

Other than these observations, I updated the GUI to reflect the screen size better.

![New GUI view](posts/img/gui3.png)
*Figure: New size accurate intersection plot.*

**Todo**
* ~~Increase camera resolution and redo calibration.~~

* (Move camera)