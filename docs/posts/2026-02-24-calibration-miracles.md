---
title: Calibration Miracles
date: 2026-02-24
---

This time I have redone the calibration with printed markers and higher resolution camera. And to my surprise, it worked better then I expected. The jitter has ben reduced greatly. Amazing! 

There is still some jitter. To reduce this, I want to implement a sliding window that filters outliers and then calculates an average. Since the camera records at 30 fps, there are a few frames to play around with.

I also did some refactoring of the code, and wrote the math explanation below.

Next I'd like to add a function that gives me exact screen pixel coordinates based on the intersection. This should not be hard. With this I will be able to have limited interaction with the screen, solely based on aim. Previously I had not considered the possibility to also detect the moment the trigger is pulled. On first thought some kind of infrared light could work perhaps. The camera could maybe detect something like this. I'll have to give this some consideration.

![Colorful calibration](posts/img/calibration.png)
*Figure: This is an example calibration frame.*

**Todo**
* Implement window filter to reduce jitter.
* Create a function that translates coordinates into pixel coordinates.
* Think about a way to detect a trigger pull.

