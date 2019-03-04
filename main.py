import cv2
import ChamferMatching as CM
import numpy as np


path='TestImage/TestImg1.jpg'
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
# cv2.imshow("edge",edge)

# CDT
CDTRes=CM.ChamferMatching.Chamfer_Dist_Transform(imgrev[241:480,200:600])

#CDT result
# cv2.imshow("CDT",CDTRes)

HalfPic=imgrev[241:480,200:600]
#mean match with the template
MatchingRes=CM.ChamferMatching.MeanConv(HalfPic[80:239,0:400],graytplt)

# get coordinates of the minimum value of MatchingRes
rol, column = MatchingRes.shape
_positon =np.argmin(MatchingRes)
print(_positon)
m, n = divmod(_positon, column)
m=m+80
print ("The rol is " ,241+m)
print ("The column is ",  200+n)
print ("The minimum of the a is ", MatchingRes[m -80, n])
# 图像匹配窗左上角位于（241+m,200+n）处
#左侧直线两端点为（158，32）和（0，62）
# 右侧直线量端点为（158，133）和（0，98）
canvas=img;
green = (0, 255, 0)
red = (0, 0, 255)
cv2.line(canvas,(200+n+32-4,241+m+158),(200+n+62-4,241+m),red,2)
cv2.line(canvas,(200+n+133-4,241+m+158),(200+n+98-4,241+m),green,2)

# # 左侧线起点为（200+n+62,241+m），延伸32个垂直像素单位后处于第241+m-32行，269+n列
# # 右侧线起点为(200+n+98,241+m),延伸32个垂直像素单位后处于241+m-32行，200+n+98-7=291+n列
# cv2.line(canvas,(200+n+62,241+m),(269+n,209+m),red,3)
# cv2.line(canvas,(200+n+98,241+m),(291+n,209+m),green,3)

ndArrayL=np.zeros(32)
# 分32个角度将左侧的线往一边倾斜
for i in range(-16,16):
    height, width=CDTRes.shape
    blankCanvas=np.zeros(CDTRes.shape)
    endPixL=69+n+i-4
    # endPixR=91+n+i*2
    cv2.line(blankCanvas,(n+62-4,m),(endPixL,m-32),255,1)
    # cv2.line(blankCanvas,(200+n+62,m-32),endPixR,255,2)
    res=(np.sum(blankCanvas*CDTRes))/height/width
    index=i+16
    ndArrayL[index]=res
minResL=np.argmin(ndArrayL)
# 左侧匹配点
MatchingPosL=269+n+(minResL-16)

ndArrayR=np.zeros(32)
# 分32个角度将右侧的线往一边倾斜
for j in range(-16,16):
    # height, width=CDTRes.shape
    blankCanvas=np.zeros(CDTRes.shape)
    # endPixL=69+n+j*2
    endPixR=91+n+j-4
    # cv2.line(blankCanvas,(200+n+62,m-32),endPixL,255,2)
    cv2.line(blankCanvas,(n+62-4,m),(endPixR,m-32),255,1)
    res=(np.sum(blankCanvas*CDTRes))/height/width
    index = j + 16
    ndArrayR[index]=res
minResR=np.argmin(ndArrayR)
# 右侧匹配点
MatchingPosR=291+n+(minResL-16)

cv2.line(canvas,(200+n+62-4,241+m),(MatchingPosL,209+m),red,2)
cv2.line(canvas,(200+n+98-4,241+m),(MatchingPosR,209+m),green,2)

cv2.imshow("Result",canvas)

# extract image from roi based on the coordinates above
rols,cols=graytplt.shape
roi=img[241+m:241+m+rols,200+n:200+n+cols]

# cv2.imshow("Matching result", roi)




cv2.waitKey(0)
