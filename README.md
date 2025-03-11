# 🖐️ PalmAuthAI - Real-Time Palm Recognition System  

## Overview  
PalmAuthAI is a **real-time palm recognition system** that uses **computer vision and deep learning** for biometric authentication. This project aims to create a **secure and scalable palm-based identity verification** system using AI. 

## Python version requirement
3.11.4

## Feature extraction
We will start the feature extraction by detecting the keypoint localization of 21 hand-knuckle coordinates within the detected hand regions. We will call them **21 Palm Landmark Points** for simplicity. This method is introduced by Google's MediaPipe.
If you want to know more about **21 Palm Landmark Points**  refer to this: [Hand Landmark Detection Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

So the current feature extraction does the following:
- Extracts 21 Palm Landmark points realtime from camera
- Flatten them into a 1D array
- Normalizes the co-ordinate values (subtract the mean & scale by standard deviation) to make them independent of hand size & position.