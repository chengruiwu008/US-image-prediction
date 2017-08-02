import cv2
import numpy as np
import linecache

for i in range(3,209):
    dir = linecache.getline('./list/Bruce_list_folder_2340.txt', (i+1))
    dir = dir.strip('\n')
    image = cv2.imread(dir,0)
    image_96 = cv2.resize(image,(96,96))
    cv2.imwrite('Bruce_list_47_sentence_41_9696/' + str(i) + '.jpg', image_96)