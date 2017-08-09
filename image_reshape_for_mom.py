import cv2

img = cv2.imread('20170808155519.jpg')
print(img.shape)
img = cv2.resize(img,(275,385))

cv2.imwrite('qwe.jpg',img)

