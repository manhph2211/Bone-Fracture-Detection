from torchvision import transforms
import cv2
import torch
import matplotlib.pyplot as plt 
import random

trans = transforms.Compose([
	transforms.Pad(random.randrange(20)),
	transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 2.0)),
	])

img = cv2.imread('hand_patient10101_image3.png')
img = cv2.resize(img,(128,128))
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
equ = cv2.equalizeHist(gray)
equ = cv2.cvtColor(equ,cv2.COLOR_GRAY2BGR)
after_trans = torch.FloatTensor(equ)
after_trans = after_trans.permute(2,0,1)
after_trans = trans(after_trans).permute(1,2,0)/255


plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(equ)
plt.subplot(1,3,3)
plt.imshow(after_trans)
plt.show()

