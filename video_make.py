import cv2
import numpy as np
alpha = 0.5
# file_num=[362,358,211,321,239,302,311,310,240,340,248,268,240,389,348,293,287,298,314,336,261,273,351,302,272,284,
#           252,303,292,247,282,312,308,282,245,356,316,324,327,336,414,267,285,311,325,271,288,347,356,353]
#
# for i in range(50):
#     out = cv2.VideoWriter('./Video Result/Bruce_list_1_sentence_%d.avi'% (i+1),cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') , 20, (320, 240))
#     for j in range(file_num[i]):
#         # get a frame
#         frame = cv2.imread("./Bruce_list01/Bruce_list1_sent%d/image (%d).bmp" % ((i+1),(1+j)))
#         # save a frame
#         out.write(frame)
#         # show a frame
#         cv2.imshow("capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     out.release()
# cv2.destroyAllWindows()
out = cv2.VideoWriter('CNN_RNN_image_pred_snake.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') , 20, (96,96))
for n in range(4,9930):
        # get a frame
    frame = cv2.imread("./CNN_RNN_image_pred_snake/%d.jpg" % n)
        # save a frame
    out.write(frame)
        # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()