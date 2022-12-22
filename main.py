import timeit
start = timeit.default_timer()
import cv2
import matplotlib.pyplot as plt
import numpy as np
from my_func import my_gradian, my_NMS , my_feature_arr , my_show , my_distance , my_min_distance

#############  calc gradient and Ixx Iyy Ixy for im01 and im02 ########
print("progres (1 of 2)" , end=" : ")

# im1_bgr=cv2.imread('im01.jpg',cv2.IMREAD_COLOR)
im1_bgr=cv2.imread('1.jpg',cv2.IMREAD_COLOR)
im1=cv2.cvtColor(im1_bgr,cv2.COLOR_BGR2GRAY)

# im2_bgr=cv2.imread('im02.jpg',cv2.IMREAD_COLOR)
im2_bgr=cv2.imread('2.jpg',cv2.IMREAD_COLOR)
im2=cv2.cvtColor(im2_bgr,cv2.COLOR_BGR2GRAY)

[Ixx1,Iyy1,Ixy1,gradient1]=my_gradian(im1)
[Ixx2,Iyy2,Ixy2,gradient2]=my_gradian(im2)
        
####### show and save results ########
my_show(gradient1, 'gradient of im01', gradient2, 'gradient of im02')
cv2.imwrite('result/res01_grad.jpg',gradient1);
cv2.imwrite('result/res02_grad.jpg',gradient2);

gausskernel=cv2.getGaussianKernel(ksize=31,sigma=3)

###############   calc Sxx ####################

Sxx1=cv2.filter2D(Ixx1,ddepth=cv2.CV_64F,kernel=gausskernel)
Sxx2=cv2.filter2D(Ixx2,ddepth=cv2.CV_64F,kernel=gausskernel)

my_show(Sxx1, '$S_x^2$ for im01', Sxx2, '$S_x^2$ for im02')

##############    calc Syy    ##################

Syy1=cv2.filter2D(Iyy1,ddepth=cv2.CV_64F,kernel=gausskernel)
Syy2=cv2.filter2D(Iyy2,ddepth=cv2.CV_64F,kernel=gausskernel)

my_show(Syy1, '$S_y^2$ for im01', Syy2, '$S_y^2$ for im02')

##############    calc Sxy    ##################

Sxy1=cv2.filter2D(Ixy1,ddepth=cv2.CV_64F,kernel=gausskernel)
Sxy2=cv2.filter2D(Ixy2,ddepth=cv2.CV_64F,kernel=gausskernel)

my_show(Sxy1, '$S_{xy}$ for im01', Sxy2, '$S_{xy}$ for im02')

######## calc structure tensor " M " for im01 and im02 ########
M1_up=np.concatenate((Sxx1, Sxy1), axis=1)
M1_down=np.concatenate((Sxy1,Syy1), axis=1)
M1=np.concatenate((M1_up,M1_down), axis=0)

M2_up=np.concatenate((Sxx2, Sxy2), axis=1)
M2_down=np.concatenate((Sxy2,Syy2), axis=1)
M2=np.concatenate((M2_up,M2_down), axis=0)

##### calc det and trace ########

tmp1=np.multiply(Sxx1,Syy1)
tmp2=np.multiply(Sxy1,Sxy1)

det1=np.subtract(tmp1,tmp2)
trace1=np.add(Sxx1,Syy1)

tmp1=np.multiply(Sxx2,Syy2)
tmp2=np.multiply(Sxy2,Sxy2)

det2=np.subtract(tmp1,tmp2)
trace2=np.add(Sxx2,Syy2)

#### calc R and showing it ######
k=30
R1=cv2.subtract(det1,k*trace1)
R1_show=np.absolute(R1)
R1_show=np.uint8(R1_show)

R2=cv2.subtract(det2,k*trace2)
R2_show=np.absolute(R2)
R2_show=np.uint8(R2_show)

my_show(R1_show, '$R_1$', R2_show, '$R_2$')
cv2.imwrite('result/res03_score.jpg',R1_show);
cv2.imwrite('result/res04_score.jpg',R2_show);

#### thresholding and showing result and save it ######
my_threshold=100
_,th1=cv2.threshold(R1,my_threshold,255,cv2.THRESH_BINARY)
_,th2=cv2.threshold(R2,my_threshold,255,cv2.THRESH_BINARY)

my_show(th1, 'threshold for im01', th2, 'threshold for im02')

cv2.imwrite('result/res03_thresh.jpg',th1);
cv2.imwrite('result/res04_thresh.jpg',th2);

#################  non maximum suppresion ###############
size=17
res1=my_NMS(size, th1)
res2=my_NMS(size, th2)
                  
[m1,n1]=np.shape(res1)
[m2,n2]=np.shape(res2)

############# draw points in original images ##########

pnt1,t=[],0
tmp1=im1_bgr.copy()
for i in range(m1):
    for j in range(n1):
        if res1[i][j]==255 :
            t+=1
            pnt1.append([i,j,t])
            cv2.circle(tmp1,(j,i),radius=2,color=(0,255,0) ,thickness=2 )

pnt2,t=[],0          
tmp2=im2_bgr.copy()
for i in range(m2):
    for j in range(n2):
        if res2[i][j]==255 :
            t+=1
            pnt2.append([i,j,t])
            cv2.circle(tmp2,(j,i),radius=2,color=(0,255,0) ,thickness=2 )
cv2.imwrite('result/res07_harris.jpg',tmp1);
cv2.imwrite('result/res08_harris.jpg',tmp2);

####### finding feature array for each point ##############

n=20
feature1 = my_feature_arr(n, pnt1, im1_bgr, m1, n1)
feature2 = my_feature_arr(n, pnt2, im2_bgr, m2, n2)

l1=len(feature1)
l2=len(feature2)

####### find corresponding point in pic2 for pic1 ######
print ("done !")
print("progres (2 of 2) : (%) ")

dist=my_distance(feature1, feature2, l1, l2)
[cores1,cores2]=my_min_distance(dist, l1, l2)

####### find valid correspond point for each image #######

thresh=0.75
correspond1=[] # [pnt1 , pnt2] : pnt1 is from pic1 correspond to pnt2 from pic2
for row in cores1:
    if np.divide(row[2],row[4]) < thresh :
        correspond1.append([row[0],row[1]])
        
correspond2=[] # [pnt1 , pnt2] : pnt1 is from pic2 correspond to pnt2 from pic1
for row in cores2:
    if (np.divide(row[2],row[4])) < thresh :
        correspond2.append([row[0],row[1]])
        
final_cor=[]
for row in correspond1:
    for row2 in correspond2:
        if( row[0]==row2[1] and  row[1]==row2[0] ):
            final_cor.append(row) # [pic1 pic2]
            
for i,row in enumerate(final_cor):
    ch=row[1]
    for j,row2 in enumerate(final_cor):
        if  (row2[1]==ch and i!=j):
            final_cor[i].append(-1)
            
############ draw correspond point in each image #########

tmp1=im1_bgr.copy()
tmp2=im2_bgr.copy()
for row in final_cor:
    p1=row[0] # pnt in pic 1
    p2=row[1] # pnt in pic 2
    
    (i1,j1)=pnt1[p1-1][0:2] # find cordinates
    (i2,j2)=pnt2[p2-1][0:2]
    
    cv2.circle(tmp1,(j1,i1),radius=2,color=(0,255,0) ,thickness=4 )
    cv2.circle(tmp2,(j2,i2),radius=2,color=(0,255,0) ,thickness=4 )
    
cv2.imwrite('result/res09_corres.jpg',tmp1);
cv2.imwrite('result/res10_corres.jpg',tmp2);    


############## draw lines for correspond points ##########

tmp3=im1_bgr.copy()
tmp4=im2_bgr.copy()

tmp5=cv2.hconcat([tmp3,tmp4])
 
tmp1=im1_bgr.copy()
tmp2=im2_bgr.copy()
for i,row in enumerate(final_cor):

    if i%3!=0:
       continue
    
    p1=row[0] # pnt in pic 1
    p2=row[1] # pnt in pic 2

    (i1,j1)=pnt1[p1-1][0:2] # find cordinates
    (i2,j2)=pnt2[p2-1][0:2]
    j2+=n1
    if i%2==0:
        cv2.line(tmp5,(j1,i1),(j2,i2),color=(0,0,255),thickness=2)
        cv2.circle(tmp5,(j1,i1),radius=2,color=(0,255,0) ,thickness=10 )
        cv2.circle(tmp5,(j2,i2),radius=2,color=(0,255,0) ,thickness=10 )
    else:
        cv2.line(tmp5,(j1,i1),(j2,i2),color=(255,0,0),thickness=2)  
        cv2.circle(tmp5,(j1,i1),radius=2,color=(0,255,0) ,thickness=10 )
        cv2.circle(tmp5,(j2,i2),radius=2,color=(0,255,0) ,thickness=10 )
    
cv2.imwrite('result/res11.jpg',tmp5);

stop = timeit.default_timer()

print('\n Run-Time: ', (stop - start)/60)  
   


