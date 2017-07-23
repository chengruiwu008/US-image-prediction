import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal

# 生成高斯算子的函数
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

# 生成标准差为2的5*5高斯算子
suanzi = np.fromfunction(func,(20,20),sigma=5)

# 打开图像并转化成灰度图像
image = Image.open('./cropedoriginalPixel2/1.jpg').convert("L")
image_array = np.array(image)

# 图像与高斯算子进行卷积
image2 = signal.convolve2d(image_array,suanzi,mode="same")

# 结果转化到0-255
image2 = (image2/float(image2.max()))*255

# 显示图像
plt.subplot(2,1,1)
plt.imshow(image_array,cmap=cm.gray)
plt.axis("off")
plt.subplot(2,1,2)
plt.imshow(image2,cmap=cm.gray)
plt.axis("off")

plt.show()