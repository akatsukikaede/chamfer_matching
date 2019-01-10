import cv2
import ChamferMatching as CM


path='TestImage/railway2.jpg'

# read image
img = cv2.imread(path)

# reshape
img=p0=cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)

# grayimage
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# extract edge features
edge=cv2.Canny(img,50,150)

# color reverse for edge
imgrev=(255-edge)

# origin
cv2.imshow("origin",img)

# reversed edge
cv2.imshow("edge",imgrev)

# CDT
CDTRes=CM.ChamferMatching.Chamfer_Dist_Transform(imgrev)

# result
cv2.imshow("CDT",CDTRes)

cv2.waitKey(0)
