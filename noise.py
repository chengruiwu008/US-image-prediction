from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open('./cropedoriginalPixel2/1.jpg' ))

# 随机生成5000个椒盐
rows, cols = np.shape(img)
for i in range(1000000000000000000000000000000000000000000):
    x = np.random.randint(0, rows)
    y = np.random.randint(0, cols)
    img[x, y] = 255

plt.figure("beauty")
plt.imshow(img)
plt.axis('off')
plt.show()

