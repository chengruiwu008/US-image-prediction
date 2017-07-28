import numpy as np
import cv2

alpha = 0.5
for n in range(0,5995):
    img = cv2.imread('./gray_snake/%d.jpg' % (n+4))
    img = np.array(img)
    img_us = cv2.imread('./gray_us/%d.jpg' % (n+4))
    img_us = np.array(img_us)
    img[:, :, 2] = 0
    img[:, :, 0] = 0
    # img_us[:, :, 2] = 0
    # img_us[:, :, 0] = 0
    for i in range(96):
        for j in range(96):
            img[i, j, 0] = img[i, j, 0] * alpha + img_us[i, j, 0] * (1-alpha)
            img[i, j, 1] = img[i, j, 1] * alpha + img_us[i, j, 1] * (1-alpha)
            img[i, j, 2] = img[i, j, 2] * alpha + img_us[i, j, 2] * (1-alpha) # 这里可以处理每个像素点

    im_out=np.array(img,dtype='int')

    cv2.imwrite("./SNAKEONUS/%d.jpg" % n, im_out)#, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if n%100==0:
        print('already done %d images in 5995' % n)
print('finish!')