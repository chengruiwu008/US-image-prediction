import cv2
import numpy as np
import matplotlib.pyplot as plt

def loss(img_1,img_2):
    sum_ = 0.0
    for j in range(96*96):
        sum_ =sum_ + (img_1[j]-img_2[j])**2
    return sum_/(96*96)

loss_0=[]
loss_1=[]
loss_2=[]
loss_3=[]
loss_tar=[]
loss_aver=[]

# plt.axis('auto')
# plt.ion()
# plt.show()
for i in range(1,349):
    image_0=cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg' % i)
    image_0 = np.array(image_0, dtype='int').reshape(-1)
    image_1=cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg'% (i+1))
    image_1 = np.array(image_1, dtype='int').reshape(-1)
    # image_2=cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg' % (i+2))
    # image_2 = np.array(image_2, dtype='int').reshape(-1)
    # image_3=cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg' % (i+3))
    # image_3 = np.array(image_3, dtype='int').reshape(-1)
    image_tar=cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg' % (i+4))
    image_tar = np.array(image_tar, dtype='int').reshape(-1)
    image_pred=cv2.imread('./pred_result_challenge/%d.jpg' % (i+4))
    image_pred = np.array(image_pred, dtype='int').reshape(-1)
    image_aver = (image_0+image_1)/2
    # image_far = cv2.imread('./Bruce_list_1_sentence_50_9696/%d.jpg' % (i + 26))
    # image_far = np.array(image_far, dtype='int').reshape(-1)

    loss_0.append(loss(image_pred, image_0))
    loss_1.append(loss(image_pred, image_1))
    # loss_2.append(loss(image_pred, image_2))
    # loss_3.append(loss(image_pred, image_3))
    loss_tar.append(loss(image_pred, image_tar))
    loss_aver.append(loss(image_pred, image_aver))

    # list=['loss_0','loss_1','loss_2','loss_3','loss_tar']
    # print('prediction  %d'%i)
    # print('loss with input_0 and output  ',loss(image_pred,image_0))
    # sum=0.0
    # # for n in range(96*96):
    # #     sum=sum+(image_0[i]-image_pred[i])**2
    # # print(sum/(96*96))
    # print('loss with input_1 and output  ', loss(image_pred, image_1))
    # print('loss with input_2 and output  ', loss(image_pred, image_2))
    # print('loss with input_3 and output  ', loss(image_pred, image_3))
    # print('loss with target and output   ', loss(image_pred, image_tar))
    # print('loss with farget and output   ', loss(image_pred, image_far))
    plt.plot(loss_0, '-b')
    plt.plot(loss_1, '-y')
    # plt.plot(loss_2, '-y')
    # plt.plot(loss_3, '-p')
    plt.plot(loss_aver, '-g')
    plt.plot(loss_tar, '--r')
    # plt.draw()
    # plt.pause(0.01)
    #print(min(loss_tar,loss_3,loss_2,loss_1,loss_0))
plt.show()
print('sum(loss_tar) ',sum(loss_tar))
print('sum(loss_0)   ',sum(loss_0))
print('sum(loss_1)   ',sum(loss_1))
# print('sum(loss_2)   ',sum(loss_2))
# print('sum(loss_3)   ',sum(loss_3))
print('sum(loss_aver)',sum(loss_aver))