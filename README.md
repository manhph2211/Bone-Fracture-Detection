# Bone-Fracture-Detection 😟
Data augmentation and Preprocessing for bone fracture detection in Xray images :raising_hand:

## 1. Distal radius fractures 😢

<img src="https://www.gchandtherapy.com/wp-content/uploads/2017/04/fractureddistalradius.png" width="400" height="200">

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

## 1.3 Methods 🙂

- First, for data augmentation: 

  - Using mirroring, sharpness, brightness and contrast augmentation.

  - Don't use  shear, strain or spot noise augmentation since these could cause a normal hand image to be classified as a hand with a fracture.

- The object detection neural network had better results when trained only on AP images instead of AP and lateral images together

- To increase the classification accuracy of finding if fractures appear or
not in the X-ray image the X-ray images were tagged with two labels one
for images with fractures and one for hand images with no fractures.

- To increase the detection accuracy, four types of image augmentation
were created: sharpness, brightness, contrast, and mirror symmetry.


# 2. Arm fracture detection in X-rays

## 2.1 Overview

- 3 main points:

  - Preprocessing method including: Opening operation, developing pixel value transformation to enhance the contrast of img.

  - New back-bone network based on feature pyramid architecture for gainning more fractual information
 
  - Anchor scale reduction and tiny RoIs expansion is exploited to find more fractures.

## 3.2 Preprocessing 

<img src="https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/preprocessing.png">

- 2 problems : noise, dark background:

  - The effects of noise can be mitigated by using morphological opening operation with a 21x21 kernel is adopted to process grayscale img. Also, the main area can be identified

  - Increasing brightness - pixel transformation - Using cumulative distribution function of the normal distribution to perform gray strech on the original image - Take the maximum pixel value of main area as the mean of the normal distribution to make the transformation sensitive to the fracture area

## 3.3 Network 

![](https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/img.png)  ![](https://github.com/manhph2211/Bone-Fracture-Detection/blob/main/imgrm/process.png )

- An improved two-stage R-CNN method: 

  - After preprocessing part, a novel backbone network: 
  
    - Resnet - is composed 5 stages, the feature map output from last layers of each 5 stages are denoted as C1,C2,C3,C4,C5, respectively. 
    
    - FPN - Feature Pyramid Network combines low-resolution, semantically strong features with high-resolution,semantically weak features that has rich semantics at all levels. The feature maps {C2;C3;C4;C5} are used to create the feature pyramid. C1 is not included in the pyramid due to its large memory footprint.  --> final feature maps {P2;P3;P4;P5}
    
    - P2,P3,P4,P5 are resized to the same size as P4 through max-pooling and interpolation. Second, integrated features are obtained by average the rescaled {P2;P3;P4;P5}. Third, we use the embedded Gaussiann on-local attention module to refine the integratedfeatures. Fourth, the refined features are then rescaled using the same but reverse procedure to strengthen the original features {P2;P3;P4;P5}, namely element-wise adding refine features to {P2;P3;P4;P5}. Finally, the outputs {S2;S3;S4;S5;S6} are used for object detection following the same pipeline in FPN. Here, S6 is maxpooled from S5. In this new architecture, each resolution in the feature pyramid gains the same information from other resolutions,balancing the flow of information and making the features more discriminating.
    
  - Feature map (S2,S3,S4,S5,S6) with 5 differences scales are fed into Region Proposal Network, which provide object proposals at each pixel position.

  - RPN generating 256 region of interests(RoIs). Hereafter, the receiptive field expansion is exploited to expand the tiny RoIs for the detetion of tiny fractures.( For each RoI, if its width is less than 30 pixels, then 20 pixels are added to its width. If its width is less than 40, then 10 pixels are added to its width. This rule also applies to the length adjustment of the RoIs)

  - RoI pooling layer unifies the size of the crooped features in RoIs into a small feature map with a fixed spatial extent of 7x7

  - The feature map with fixed 7×7 spatial extent is flattened to a feature vector, which is input into two 1024-way fully connectedlayers. Finally, the regressor regresses bounding boxes,and the classifier predicts classes.

## 3.4 Anchor scales reduction

- It is very important to set appropriate anchor scale for detection tasks. 

- Experiments indicates that anchor scale with respect to feature maps {P2;P3;P4;P5;P6} is {256;128;64;32;16} is the best. Those scales guarantee more foreground RoIs for RPN training.
