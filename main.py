import cv2
import ChamferMatching as CM
import numpy as np


path='TestImage/TestImg5.jpg'
templatePath='TestImage/Template.jpg'
# read image
img = cv2.imread(path)
tplt=cv2.imread(templatePath)

# # reshape
# img=p0=cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)

# grayimage
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
graytplt=cv2.cvtColor(tplt,cv2.COLOR_RGB2GRAY)

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

#CDT result
cv2.imshow("CDT",CDTRes)

#mean match with the template
MatchingRes=CM.ChamferMatching.MeanConv(imgrev,graytplt)

# get coordinates of the minimum value of MatchingRes
rol, column = MatchingRes.shape
_positon =np.argmin(MatchingRes)
print(_positon)
m, n = divmod(_positon, column)
print ("The rol is " ,m)
print ("The column is ",  n)
print ("The minimum of the a is ", MatchingRes[m , n])

# extract image from roi based on the coordinates above
rols,cols=graytplt.shape
roi=img[m:m+rols,n:n+cols]

cv2.imshow("Matching result", roi)




cv2.waitKey(0)
