import cv2
import numpy as np

out = cv2.VideoWriter('./Video Result/US+US-pred.avi', -1, 20, (96*2,96))

for i in range(0,5995):
    # get a frame
    frame = cv2.imread("./packed_us/%d.jpg" % i)
    # save a frame
    out.write(frame)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()