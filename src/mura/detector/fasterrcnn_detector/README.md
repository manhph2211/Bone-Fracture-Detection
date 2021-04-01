Faster RCNN for Detecting Bone Fracture :smile:
=====


# 0. Introduction :smiley:

- This implementation aims to predict bone fracture in Xray Image using Faster RCNN model.

# 1. Dependencies :smiley:

- Torch
- Torchvision
- Opencv-python
- Tqdm

# 2. Faster RCNN :smile:

<img src="https://i.stack.imgur.com/RUJ2b.png" width="800" height="400">

## 2.1 Region Proposal Networks :smiley:

### 2.1.1 Selective Search From RCNN

- One approach for localizing object is Exhaustive Search, which uses sliding window of different size... seems works but compute a lot as it searchs for object in thousands of windows even for small image size and more than that it is not efficent. Instead, RCNN uses RPN which based on Selective Search algorithm which uses both Exhaustive search and segmentation.

- Selective Search first initialize segmentation,then using Greedy Algorithm to combine similar regions to make better/larger regions, then using the segmented region proposals to generate candidate object locations - Bounding boxes. Selecsive Search in details:

  - From set of regions, choose two that are most similar.
  - Combine them into a single, larger region.
  - Repeat the above steps for multiple iterations.

- However, Due to number of windows it processed, it takes anywhere from 1.8 to 3.7 seconds (Selective Search Fast) to generate region proposal which is not good enough for a real-time object detection system.


<img src="https://arthurdouillard.com/figures/selective_search1.png" width="800" height="400">

### 2.1.2 RPN of Faster RCNN, What is the difference?

- Train a NN instead of Using selective search, following 3 steps:

  - The input image goes through a convolution network which will output a set of convlutional feature maps on the last convolutional layer.
  - Then a sliding window is run spatially on these feature maps. The size of sliding window is  n×n(3x3 for example). For each sliding window, a set of 9 anchors are generated which all have the same center  (xa,ya)  but with 3 different aspect ratios and 3 different scales. Anchors are labeled positive or negative based on IOU(mormaly positive if IOU>0.7 and Negative if IOU<0.3)
  - Finally, the  3×3  spatial features extracted from those convolution feature maps (shown above within red box) are fed to a smaller network which has two tasks: classification (cls) and regression (reg). 

## 2.2 RoI Pool - From Fast RCNN :sleepy:

- Each proposal will be of a different shape. So there is a need for converting all the proposals to fixed shape as required by fully connected layers. ROI Pooling is exactly doing this before moving to the FC layers

- ROI pooling produces the fixed-size feature maps from non-uniform inputs by doing max-pooling on the inputs. The number of output channels is equal to the number of input channels for this layer. ROI pooling layer takes two inputs:
  - feature map obtained from a Convolutional Neural Network after multiple convolutions and pooling layers.
  - ‘N’ proposals or Region of Interests from Region proposal network. Each proposal has five values, the first one indicating the index and the rest of the four are proposal coordinates. Generally, it represents the top-left and bottom-right corner of the proposal.


## 2.3 Sharing Features for RPN and Fast-RCNN :smiley:

- The RPN and Fast R-CNN, are independent networks. Each of them can be trained separately. However, for Faster R-CNN it is possible to build a unified network in which the RPN and Fast R-CNN are trained at once. The core idea is that both the RPN and Fast R-CNN share the same convolutional layers.

- The method is called alternating training:
  - The RPN is first trained to generate region proposals. The weights of the shared convolutional layers are initialized based on a pre-trained model on ImageNet. The other weights of the RPN are initialized randomly. 
  - After the RPN produces the boxes of the region proposals, the weights of both the RPN and the shared convolutional layers are tuned. The generated proposals by the RPN are used to train the Fast R-CNN module. In this case, the weights of the shared convolutional layers are initialized with the tuned weights by the RPN. 
  - The other Fast R-CNN weights are initialized randomly. While the Fast R-CNN is trained, both the weights of Fast R-CNN and the shared layers are tuned. The tuned weights in the shared layers are again used to train the RPN, and the process repeats.



# 3. Usage :smiley:

## 3.1 train

- Make sure you're in `fasterrcnn_detector` folder. Then run:

```
python3 utils.py
python3 train.py

```

## 3.2 Inference :smiley:

- Just try `python3 predict.py`


# 4. References :smiley:

- This implementation is strongly based on:  

  - [this paper](https://arxiv.org/pdf/1506.01497.pdf)
  - [this blog](https://www.quora.com/How-does-RPN-work-on-the-Faster-R-CNN?no_redirect=1)
  - [this blog](https://towardsdatascience.com/region-of-interest-pooling-f7c637f409af)