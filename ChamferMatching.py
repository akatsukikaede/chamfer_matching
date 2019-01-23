import cv2
import numpy as np


class ChamferMatching:

    # calculate euclidean distance
    def Eucl_Distance(x1: float, x2: float, y1: float, y2: float) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5



    # chamfer distance transform, CDT for short
    def Chamfer_Dist_Transform(image: np.ndarray):
        # two distance value
        d1 = 1
        d2 = (d1 * 2) ** 0.5
        # convert pixel data type from int(0-255) to float(0-1.0)
        CDTRes = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        CDTRes[CDTRes!=0]=float("inf")
        # first pass: top-bottom; left-right
        for i in range(1, CDTRes.shape[0] - 1):
            for j in range(1, CDTRes.shape[1] - 1):
                # 3x3 window:
                # 1  2  3
                # 0  x  4
                # 5  6  7

                # pixel at x position(center distance)
                BWValCenter = CDTRes[i, j]

                #pixel 0
                BWVar0 = CDTRes[i, j - 1]
                dist0 = d1 + BWVar0

                # pixel 1
                BWVar1 = CDTRes[i - 1, j - 1]
                dist1 = d2 + BWVar1

                # pixel 2
                BWVar2 = CDTRes[i - 1, j]
                dist2 = d1 + BWVar2

                # pixel 3
                BWVar3 = CDTRes[i - 1, j + 1]
                dist3 = d2 + BWVar3

                # min value for all distances
                fdist = min(BWValCenter, dist0, dist1, dist1, dist2, dist3)

                # replace pixel x with min value
                CDTRes[i, j] = fdist

        # second pass: bottom-top; right-left
        # same method as the first pass
        for k in range(CDTRes.shape[0] - 2, 0,-1):
            for l in range(CDTRes.shape[1] - 2, 0,-1):
                # pixel at x position
                BWValCenter = CDTRes[k, l]

                # pixel 4
                BWVar4 = CDTRes[k, l + 1]
                dist4 =d1 + BWVar4

                # pixel 5
                BWVar5 = CDTRes[k + 1, l + 1]
                dist5 = d2 + BWVar5

                # pixel 6
                BWVar6 = CDTRes[k + 1, l]
                dist6 = d1 + BWVar6

                # pixel 7
                BWVar7 = CDTRes[k + 1, l - 1]
                dist7 =d2+ BWVar7

                # min dist
                fdist = min(BWValCenter, dist4, dist5, dist6, dist7)

                # replace pixel x with min value
                CDTRes[k, l] = fdist

        # normalized output (data type:float)
        CDTRes = cv2.normalize(CDTRes[1:CDTRes.shape[0]-2,1:CDTRes.shape[1] - 2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return CDTRes

    def MeanConv(image,weight):
        # size of image and weight
        height, width = image.shape
        h, w = weight.shape
        # size of the result after convolution
        new_h = height - h + 1
        new_w = width - w + 1
        new_image = np.zeros((new_h, new_w), dtype=np.float)
        # mean convolution
        for i in range(new_h):
            for j in range(new_w):
                new_image[i, j] = (np.sum(image[i:i + h, j:j + w] * weight))/w/h
                # if(j==new_h-1):
                #     print("Halt")
        return new_image


