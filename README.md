# Bone-Fracture-Detection 
Data augmentation and Preprocessing for bone fracture detection in Xray images :raising_hand:

## 1. Distal radius fractures

![alt text](https://www.gchandtherapy.com/wp-content/uploads/2017/04/fractureddistalradius.png)

## 1.1 Introduction :smiley:

- This part is about using faster RCNN to detect distal identify and locate distal radius fractures in anteroposterior X-ray images.  (38 images - resolution of up to 1600×1600pixels for training). The result (ACC=0.96 & mAP=0.866) is significantly more accurate than the detection achieved by physicians and radiologists (only 0.7 ACC)

- Some challenges:

 - In many casesthe fracture’s size is small and hard to detect.
 - The fractures have a wide range of different shapes

- The advantage of Faster R-CNN is that it can handle high-resolution images. Also, Faster R-CNN can be trained toa high accuracy in detecting objects with a small number of images. Two clear tasks:

 - Classifying if there is a fracture in the distal radius. 
 - Finding the fracture’s location. 

