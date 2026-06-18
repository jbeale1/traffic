High quality motion-triggered images using a Raspberry Pi HQ or global shutter camera. The 12 Mpixel HQ camera delivers much better resolution, but it suffers from significant rolling-shutter artifact on faster vehicles. The 1.6 Mpixel IMX296 sensor has a global shutter so there is no trapezoidal distortion, regardless of speed, and the 3.45 um pixels give it better low-light performance, other things being equal.

It is possible to estimate a vehicle's speed using a pair of images taken from the side, if you know the camera-vehicle distance and therefor the image scale in mm per pixel, and the image frame rate. Then you can match up keypoints on each frame, or do a template match between images using the area of the vehicle. The match quality and the CV (coefficient of variation) of the individual keypoints gives an indication of the reliability of the speed estimates.


