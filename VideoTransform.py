import cv2
import numpy as np
import  ChamferMatching as CM

path='TestVideo/TestVid.mp4'

#read video
cap=cv2.VideoCapture(path)

#play video
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    # cv2.imshow("capture", frame)

    # resize
    img=cv2.resize(frame,(320,240),interpolation=cv2.INTER_CUBIC)
    # grayimage
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # extract edge features
    edge = cv2.Canny(img, 50, 150)
    # color reverse for edge
    imgrev = (255 - edge)
    # CDT
    CDTRes = CM.ChamferMatching.Chamfer_Dist_Transform(imgrev)
    # result
    cv2.imshow("CDT", CDTRes)
    # cv2.imshow("edge",imgrev)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

