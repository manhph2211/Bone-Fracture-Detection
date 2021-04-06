# Solo for Xray segmentation

## Model
The general architecture is showed below:

![solo](images/solo.png)
### CoorConv layer

![CoorConv](images/CoorConv.jpeg)
#### Why not conventional convolution? :worried:

The author of the original [paper](https://arxiv.org/pdf/1807.03247.pdf) points out the falling of convolution in some simple trivial task:

- Supervised rendering: Given some Cartesian coordinate (i, j), highlight the square around this center with fixed size in the 'black' image (a grayscale image with value 0 in each pixel).
- Supervised coordinate classification: Given some Cartesian coordinate (i, j), provide its correspond one hot vector representation.
- Supervised regression: Same as *_Supervised coordinate classification_* but with the inverse direction.

#### Explanation :raised_hand:

- CNN is pixel-invariant.
- This is too hard for CNN to map data from lower dimensions to higher dimensions.
- CNN fails to learn a smooth function to represent a dataset.

#### **CoorConv** is the solution :boom:

 **CoorConv** concatenates additional dimensions to an input which annotate the coordinate of each pixel follow by normal convolution layer. By this
approach, a filter is able to know the position of each pixel and break down the invariant character. I think it's useful for some tasks which need the 
coordinate prediction like object detection, segmentation, ...
### Backbone
Model uses Resnet architecture for extract features.
### Neck
FPN is employed for multiple scales prediction.
### Head
![head](images/head.png)

This model gets each feature map from FPN as inputs. The head at the top layer uses
simultaneous 2 branches to predict each category and correspond mask.  
- Semantic category:
  With each grid, this branch predicts correspond logit vector and classify this grid to class which have the largest probability.
- Instance mask:
  Using **CoorConv** layer at first and apply convolution multi time, we achieve a tensor with each channel is a soft mask associated with each grid. **NMS** is also used to obtain the final result.
### Loss function
Training loss function is defined as follows:

![loss](images/loss1.png)

Model uses **Focal loss** for penalty classification loss.

The formulations of mask loss are represented as follows:

![](images/loss2.png)

where, the author chooses **Dice loss** as *_metric_*:

![](images/loss3.png)
![](images/loss4.png)

## Implementation

## Result
