import cv2
import matplotlib.pyplot as plt 
import os
import glob 

img = cv2.imread('./hand_patient10101_image3.png')
mask = cv2.imread('hand_patient10101_image3_mask.png')
mask[mask == 255] = 1
bg_subtracted_img = img * mask
# cv2.imwrite('./bg_subtracted_img.png',bg_subtracted_img)
mask[mask == 1] = 255
plt.subplot(1,4,1)
plt.imshow(img)
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(mask)
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(bg_subtracted_img)
plt.axis("off")

clahe = cv2.createCLAHE(clipLimit= 5, tileGridSize=(7, 7))
output = cv2.cvtColor(bg_subtracted_img, cv2.COLOR_BGR2GRAY)
output = clahe.apply(output)
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
# cv2.imwrite('./after_clahe.png',output)

plt.subplot(1,4,4)
plt.imshow(output)
plt.axis("off")


plt.show()