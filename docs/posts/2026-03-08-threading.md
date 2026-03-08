---
title: Threading
date: 2026-03-08
---

Adding a video while manipulating the frames and doing all previously described camera operations is heavy, especially for python.

I decided to split the heavy lifting into two threads to avoid buffering and lag. One dedicated to reading and processing camera frames, and the other for reading and manipulating video frames. One issue I encountered was that open CVs `imShow` functions are not thread safe. I worked around this by having the threads only prepare each frame and I'm displaying it (running the `imShow`) in the main thread. Works like a charm, and I now have a tasty lookin' boar running around my forest!

Another familiar issue I encountered is that when I move the rifle side to side, the aim point also moves upward proportionally to the speed I move. This is weird.

It would be nice to have a dedicated prop gun that I could use for this project. It could be a more permanent solution to my ad hoc creation. I'm thinking about 3d-printing something like this. It would be lighter too for future users.

Regarding the trigger detection. I think if I had a prop gun, I could implement some hardware to the mix. Detecting a button press and sending communicating over WiFi or bluetooth with my program does not sound like that bad of an idea. I'll have to look more into this.

![Hog in Forest](posts/img/svin.gif)
*Figure: Working views and dot overlay.*

**Todo**
* Figure out the movement issues
* CAD a prop rifle
* Consider hardware for the prop rifle
