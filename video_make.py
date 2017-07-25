import cv2
import numpy as np

out = cv2.VideoWriter('snake_on_us.avi', -1, 20, (96,96))

for i in range(610,6015):
    # get a frame
    frame = cv2.imread("./Snake_on_us/%d.jpg" % i)
    # save a frame
    out.write(frame)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()