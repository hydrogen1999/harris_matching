
import cv2
import matplotlib.pyplot as plt
import numpy as np


def my_gradian(im):
    img=im.copy()
    Ix=cv2.Sobel( img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1 );
    Iy=cv2.Sobel( img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1 );
    
    Ixx=np.multiply(Ix,Ix)
    Iyy=np.multiply(Iy,Iy)
    Ixy=np.multiply(Ix,Iy)
      
    tmp=np.add(Ixx,Iyy)
    gradient=np.sqrt(tmp)
    
    gradient=np.absolute(gradient)
    gradient=np.uint8(gradient)
    return [Ixx,Iyy,Ixy,gradient]



def my_NMS(size,th): 
    res=th.copy()
    [m,n]=np.shape(res)
    for i in range(0,m,size):
        for j in range(0,n,size):
            try:
                ch=0
                for k in range(size):
                    for l in range(size):
                        if ch==1:
                            res[i+k][j+l]=0
                        elif res[i+k][j+l]==255:
                            ch=1
                        else:
                            pass
            except:
                pass
    return res





def my_feature_arr(N,pnt,im_bgr,m,n):
    im1=im_bgr[:,:,0]
    im2=im_bgr[:,:,1]
    im3=im_bgr[:,:,2]
    
    feature_arr=[]
    for row in pnt:
        i=row[0]
        j=row[1]
        b0=i-int(N/2)
        b1=i+int(N/2)
        t1,t2,t3=[],[],[]
        for i in range(b0,b1):
            
            
            t1.append(im1[i%m])
            t2.append(im2[i%m])
            t3.append(im3[i%m])
        
        tmp1,tmp2,tmp3=[],[],[]
        for i in range(len(t1)):
            b0=(j-int(N/2))
            b1=(j+int(N/2))
            
            row1=t1[i]
            row2=t2[i]
            row3=t3[i]
            
            tm1,tm2,tm3=[],[],[]
            for i in range(b0,b1):
                tm1.append(row1[i%n])
                tm2.append(row2[i%n])
                tm3.append(row3[i%n])
                
            tmp1.append(tm1) 
            tmp2.append(tm2) 
            tmp3.append(tm3) 
            
            
        feature_arr.append([row[2],tmp1,tmp2,tmp3])
    return feature_arr


    
    
    
def my_show(img1,title1,img2,title2):
    plt.subplot(121); plt.axis(False); plt.title(title1)
    plt.imshow(img1,'gray')
    plt.subplot(122); plt.axis(False); plt.title(title2)
    plt.imshow(img2,'gray')
    plt.show()

    
    
def my_distance(feature1,feature2,l1,l2):
    dis=[]
    ch,p=0,-1
    total=l1*l2
    for i in range(l1):
        for j in range(l2):
            row1=feature1[i]
            row2=feature2[j]   
            
            a1=np.array(row1[1])
            b1=np.array(row2[1])  
            a2=np.array(row1[2])
            b2=np.array(row2[2]) 
            a3=np.array(row1[3])
            b3=np.array(row2[3]) 
           

            dis_1=(np.linalg.norm(a1-b1))
            dis_2=(np.linalg.norm(a2-b2))
            dis_3=(np.linalg.norm(a3-b3))
            dis_=int((dis_1+dis_2+dis_3)/3)
            
            dis.append([row1[0],row2[0],dis_]) # [pnt1 from im1 , pnt2 from im2 , distance]
           
            ch+=1
            prog=int(ch*100/total)
            if prog != p :
                p=prog
                print(str(p),end=" ")  
    return dis

    
def my_min_distance(dis,l1,l2):
    min_dis1=[]
    min_dis2=[]
    
    dis.sort(key= lambda x:x[0])
    
    
    for i in range(l1):
        tmp=dis[i*l2:(i+1)*l2]
        tmp.sort(key = lambda x:x[2])
        min_dis1.append([tmp[0][0],tmp[0][1],tmp[0][2],tmp[1][0],tmp[1][1],tmp[1][2]])
    
    dis.sort(key= lambda x:x[1])
    for i in range(l2):
        tmp=dis[i*l1:(i+1)*l1]
        tmp.sort(key = lambda x:x[2])
        min_dis2.append([tmp[0][1],tmp[0][0],tmp[0][2],tmp[1][1],tmp[1][0],tmp[1][2]])
    
    return [min_dis1,min_dis2]
    
    
    
    
    
    
    
    
    
    
    

