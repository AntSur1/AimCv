---
title: Camera Feedback
date: 2026-02-28
---

So I have split the view into two a camera view with an indication if all markers are read properly. There are issues with having two separate display systems. Simply put, this makes the system less stable than if everything went through one single GUI system. Thankfully I only intend to use the plot GUI for debugging purposes, so this should not be a common problem.

The next step is to play around with overlaying the intersection point over a video. This should not be hard. The step after that would be to try to have some kind of "ideal aim" path that the user would try to follow and be "graded" based on. This can be done programmatically. Each frame of the video has to have two coordinates forming a box, in which the "correct" aim would be at. When the user aims at the screen, a simple check would be required to determine wether the aim is within the "allowed" bounds.

![New gui](posts/img/gui5.png)
*Figure: New GUI with separate windows.*

The camera view is smaller now. This is because it's intended only as a reference to the user for them to see how they should position themselves to include all markers. 

![Cam window](posts/img/cam.gif)
*Figure: Camera window functionality.*

**Todo**
* Overlay the dot information over a video
* Detect where on the video a user is aiming
