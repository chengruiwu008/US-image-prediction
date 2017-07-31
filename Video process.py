import numpy as np
import cv2

#def readfile(filename='a.avi'):
cap = cv2.VideoCapture('./Video Result/Bruce_list_1_sentence_50.avi')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (96, 96))
    cv2.imwrite('Bruce_list_1_sentence_50_9696/' + str(i) + '.jpg',frame)
    gray = cv2.cvtColor(frame, 0)#cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
cap.release()
cv2.destroyAllWindows()