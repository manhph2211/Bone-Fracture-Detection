# Bone-Fracture-Detection 
Data augmentation and Preprocessing for bone fracture detection in Xray images :raising_hand:

## 1. Distal radius fractures

![alt text](https://www.gchandtherapy.com/wp-content/uploads/2017/04/fractureddistalradius.png)

## 1.1 Introduction :smiley:

- This part is about using faster RCNN to detect distal identify and locate distal radius fractures in anteroposterior X-ray images.  (38 images - resolution of up to 1600×1600pixels for training). The result (ACC=0.96 & mAP=0.866) is significantly more accurate than the detection achieved by physicians and radiologists (only 0.7 ACC)

- Some challenges:

  - In many cases,the fracture’s size is small and hard to detect.
  - The fractures have a wide range of different shapes

- The advantage of Faster R-CNN is that it can handle high-resolution images. Also, Faster R-CNN can be trained to a high accuracy in detecting objects with a small number of images. Two clear tasks:

  - Classifying if there is a fracture in the distal radius. 
  - Finding the fracture’s location. 

## 1.2 Faster RCNN 

- Faster RCNN has 3 parts:
  -  A convolutional deep neural network for classification and generating a feature map. 
  -  A regional proposal network, generating region proposals.
  -  A regressor, finding by regression and additional convolutional layers, the precise location of each object and its classification.

![alt text](https://www.researchgate.net/profile/Zhipeng-Deng-2/publication/324903264/figure/fig2/AS:640145124499471@1529633899620/The-architecture-of-Faster-R-CNN.png)

- Base VGG16 containing 16 layers including 3x3 convolution layers, 2x2 pooling layers and fully connected layers with over 144 million parameters.

- At each window location (using a sliding window NxN), up to 9 anchors with different aspect ratios and scales give region proposals. 

- The RPN keeps anchors that either has the highest intersection over union (IOU) with the ground truth box or anchors that have IOU overlap of at least 70% with any positive ground truth.

- Find more about the model by reading [this](https://arxiv.org/pdf/1506.01497.pdf)


## 1.3 Pre-processing 

- For data augmentation: 
  - Using mirroring, sharpness, brightness and contrast augmentation.
  - Don't use  shear, strain or spot noise augmentation since these could cause a normal hand image to be classified as a hand with a fracture.

- 80% for training and 20% for the test

## 1.4 Some methods

- The object detection neural network had better results when trained only on AP images instead of AP and lateral images together

- To increase the classification accuracy of finding if fractures appear or
not in the X-ray image the X-ray images were tagged with two labels one
for images with fractures and one for hand images with no fractures.

- To increase the detection accuracy, four types of image augmentation
were created: sharpness, brightness, contrast, and mirror symmetry.
