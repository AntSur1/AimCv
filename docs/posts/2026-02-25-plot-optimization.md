---
title: Plot Optimization
date: 2026-02-25
---

Turns out that the jitter is not that bad at all when you run on higher FPS. It was (still is) really expensive to redraw all the plots and the camera view each frame. I redid how the plots were drawn and turned off the camera view, which increased the smoothness of the plots drastically. Way back I did have the camera as a separate view, doing this in this way would keep the smoothness and the camera view. I'll do that in the future I think. 

I also redid the intersection plot into a proper screen coordinate plot. Now you have the exact screen coordinate you aim at!

Next I have to start thinking about some way to overlay this information over videos. Since this is supposed to be a hunting simulation, I'd like to have a few scenarios where one can practice no how to act lawfully. This is not intended to be very accurate, but enough to see how one handles different situations.

![GUI 4](posts/img/gui4.png)
*Figure: Updated GUI with screen coordinates.*

**Todo**
* ~~Make a separate camera view.~~
* Do video overlay
* Get hunting video?
