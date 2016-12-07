import cv2
import numpy as np
import scipy
import scipy.signal as sp
import scipy.ndimage as nd
np.seterr(over='ignore')

image = cv2.imread('UBCampus.jpg',0)
np.set_printoptions(threshold=np.nan)

DoGKernel = np.array(([0,0,-1,-1,-1,0,0],
                      [0,-2,-3,-3,-3,-2,0],
                      [-1,-3,5,5,5,-3,-1],
                      [-1,-3,5,16,5,-3,-1],
                      [-1,-3,5,5,5,-3,-1],
                      [0,-2,-3,-3,-3,-2,0],
                      [0,0,-1,-1,-1,0,0]),dtype=np.float32)
DoGoutput = sp.convolve2d(image,DoGKernel)

zc = np.diff(np.sign(DoGoutput))
# print DoGoutput
ZCDoG = np.zeros(zc.shape)
for x in range(zc.shape[0]):
  for y in range(zc.shape[1]):
    if zc[x,y] != 0 and (DoGoutput[x,y] <25 or DoGoutput[x,y] > -25):
      ZCDoG[x,y]=0
    else:
      ZCDoG[x,y]=DoGoutput[x,y]
DoGoutput=np.absolute(DoGoutput)
DoGoutput= (DoGoutput-np.min(DoGoutput))/float(np.max(DoGoutput)-np.min(DoGoutput))

cv2.imshow("DoG Output",DoGoutput)
cv2.waitKey()
cv2.imwrite("DoG Output.jpg",DoGoutput)

cv2.imshow("Zero Crossing DoG Output",ZCDoG)
cv2.waitKey()
cv2.imwrite("Zero Crossing DoG Output.jpg",ZCDoG)

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)  # x
#cv2.imshow("Sobel dx",sobelx)
#cv2.waitKey()
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)  # y
#cv2.imshow("Sobel dy",sobely)
#cv2.waitKey()

Sobel = np.hypot(sobelx,sobely)
Sobel = np.array(Sobel,dtype="float32")
Sobel *= 255.0 / np.max(Sobel)
# print Sobel

threshold_value = 45
ret,Sobel_thresh = cv2.threshold(Sobel,threshold_value,255,cv2.THRESH_BINARY)
# print Sobel_thresh
#cv2.imshow("Sobel",Sobel_thresh)
#cv2.waitKey()

Sobel_thresh = scipy.misc.imresize(Sobel_thresh,ZCDoG.shape)
Sobel_thresh = np.array(Sobel_thresh,dtype="float32")
ZCDoG = np.array(ZCDoG,dtype="float32")
Strong_edge_dog = cv2.bitwise_and(ZCDoG,Sobel_thresh)
cv2.imshow("DoG Strong Edges",Strong_edge_dog)
cv2.waitKey()
cv2.imwrite("DoG Strong Edges.jpg",Strong_edge_dog)

LoGKernel = np.array(([0,0,1,0,0],
                      [0,1,2,1,0],
                      [1,2,-16,2,1],
                      [0,1,2,1,0],
                      [0,0,1,0,0]),dtype=np.float32)

LoGOutput = sp.convolve2d(image,LoGKernel)
#LoGOutput = cv2.filter2D(image,-1,LoGKernel)

zc = np.diff(np.sign(LoGOutput))

ZCLoG = np.zeros(zc.shape)
for x in range(zc.shape[0]):
  for y in range(zc.shape[1]):
    if zc[x,y] ==0  and (LoGOutput[x,y] < 10 or LoGOutput[x,y] > -10):
      ZCLoG[x,y]=LoGOutput[x,y]
    else:
      ZCLoG[x,y]=0

cv2.imshow("LoG Output",LoGOutput)
cv2.waitKey()
cv2.imwrite("LoG Output.jpg",LoGOutput)

cv2.imshow("Zero crossing LoG Output",ZCLoG)
cv2.waitKey()
cv2.imwrite("Zero crossing LoG Output.jpg",ZCLoG)

Sobel_thresh = scipy.misc.imresize(Sobel_thresh,ZCLoG.shape)
Sobel_thresh = np.array(Sobel_thresh,dtype="float32")
ZCLoG = np.array(ZCLoG,dtype="float32")
Strong_edge_log = cv2.bitwise_and(ZCLoG,Sobel_thresh)
cv2.imshow("LoG Strong Edges",Strong_edge_log)
cv2.waitKey()
cv2.imwrite("LoG Strong Edges.jpg",Strong_edge_log)

cv2.destroyAllWindows()