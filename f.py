# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:36:03 2016

@author: tobias hoelzer
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy as sp
import math
import f

#Functions doing the actual work

#Preprocessing_____________________


#Detectors__________________________
#This guy is an edge detector
def ShiTomasiDetector(image,Mask=None):
    '''
    image - image to give in (gray)
    Mask - places where the algorithm doesn't look for keypoints
    '''
    
    
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                          qualityLevel = 0.3,
                           minDistance = 7,
                          blockSize = 7 )


    points = cv2.goodFeaturesToTrack(image,mask=Mask,**feature_params)
    
    return points

#Drawing____________________________________________


def DrawFASTKeypointsAndFOV(grayimage,keypoints,imagein,FOV=None,cars=None,offcenterCamera=[0,0],savename=None):
    '''
    Draws teh keypoints, the field of view, the image center and the other cars,of the FAST algorithm
    grayimage   - input image
    keypoints   - keypoints found in FAST algorithm
    imagein     - input color image
    FOV         - Field of view as [linestart,lineend, columnstart,columnend], default [150,670,30,800]
    cars        - cars found with Haar cascade, default None
    offcenterCamera - how far is the center of the image away from the actual center 720/2, 960/2, default is [0,0]
    savename    - if defined, this is where the image is stored, default  = None
    
    FOV default is set

    e.g.: FOV_MASK = [150:670,30:800] = [y1:y2,x1:x2]
    -> FOV = [P1,P2] = [(30,150),(800,679)]
    P1 = (x1,y1)
    P2 = (x2,y2)
    '''    
    image = imagein.copy()
    

    
    #ColorMap. 
    cm = plt.get_cmap('viridis')
    colors = [cm((1.*i)/(len(keypoints)+2)) for i in range(2,len(keypoints)+2)]

    #check if type [[x,y]] or [cv2.KeyPoint]
    if isinstance(keypoints[0],np.ndarray):
        for i,point in enumerate(keypoints):

            col = tuple([colors[i][0]*255,colors[i][1]*255,colors[i][2]*255])
            cv2.circle(image,tuple(np.squeeze(point)),5,col)
    else:
        cv2.drawKeypoints(grayimage,keypoints,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cx = offcenterCamera[0]  
    cy = offcenterCamera[1]
    cv2.circle(image,(480+cx,360+cy),10,(255,0,0),-1)

    if FOV != None:
        P1 = (FOV[2],FOV[0])
        P2 = (FOV[3],FOV[1])   
        cv2.rectangle(image,P1,P2,(0,255,0)) 
        
    for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
     
    if savename: cv2.imwrite(savename,image)

    
    
    return

def Draw_moved_points(image1,image2,p1,p2,FOV=None,offcenterCamera=None,cars1=None,cars2=None,writename=None):
    '''
    Draws the two images with it's feature points and how they moved from one image to the other
    image1      - input image 1
    image2      - input image 2 changed keypoints    
    p1          - keypoints found in 1st image
    p2          - keypoints found in 2nd image
    FOV         - Field of View of the image, can be a list of 2 FOV (if the images are from two different cars). Default is None
    offcenterCamera - how far is the center of the image away from the actual center 720/2, 960/2, default is None can be a list of 2 offcenterCamera (if the images are from two different cars).
    FOV         - Field of view as [linestart,lineend, columnstart,columnend], default None
    cars1       - cars found in image1 with Haar cascade, default None
    cars2       - cars found in image2 with Haar cascade, default None
    writename   - if defined, this is where the image is stored, default  = None
   
    
    '''
    im1 = image1.copy()
    im2 = image2.copy()
    
    
    #just check
    if len(p1) != len(p2):
        print 'p1 and p2 must be of same size'
        return
        

    #Get and draw the fluchtpunkt of the camera
    if np.any(offcenterCamera):
        if np.shape(offcenterCamera) == (2,):
            cx = offcenterCamera[0]  
            cy = offcenterCamera[1]
            cv2.circle(im1,(480+cx,360+cy),10,(255,0,0),-1)
            cv2.circle(im2,(480+cx,360+cy),10,(255,0,0),-1)
        elif np.shape(offcenterCamera) == (2,2):
            cx = offcenterCamera[0][0]  
            cy = offcenterCamera[0][1]
            cv2.circle(im1,(480+cx,360+cy),10,(255,0,0),-1)
            cx = offcenterCamera[1][0]  
            cy = offcenterCamera[1][1]
            cv2.circle(im2,(480+cx,360+cy),10,(255,0,0),-1)

    
    #Draw Field of View in green
    if np.any(FOV):
        if len(FOV)== 1:
            P1 = (FOV[2],FOV[0])
            P2 = (FOV[3],FOV[1])    
            cv2.rectangle(im1,P1,P2,(0,255,0))        
            cv2.rectangle(im2,P1,P2,(0,255,0)) 
        elif len(FOV)== 2:
            P1 = (FOV[0][2],FOV[0][0])
            P2 = (FOV[0][3],FOV[0][1])    
            cv2.rectangle(im1,P1,P2,(0,255,0))
            P1 = (FOV[1][2],FOV[1][0])
            P2 = (FOV[1][3],FOV[1][1])
            cv2.rectangle(im2,P1,P2,(0,255,0))


    
    #Draw other recognized cars in red
    if np.any(cars1):
        for (x,y,w,h) in cars1:
            cv2.rectangle(im1,(x,y),(x+w,y+h),(0,0,255),1)
    if np.any(cars2):
        for (x,y,w,h) in cars2:
            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),1)
     
   
   #ColorMap. 
    cm = plt.get_cmap('viridis')
    colors = [cm((1.*i)/(len(p1)+2)) for i in range(2,len(p1)+2)]

    

    #Draw Points
    for i in range(len(p1)):
        col = tuple([colors[i][0]*255,colors[i][1]*255,colors[i][2]*255])  
        
                
        #Draw circles at the Keypoints And Arrows where they moved
        cv2.circle(im1,tuple(p1[i]),5,col,-1)
        cv2.arrowedLine(im1,tuple(p1[i]),tuple(p2[i]),col)
        
        cv2.circle(im2,tuple(p2[i]),5,col,-1)
        cv2.arrowedLine(im2,tuple(p1[i]),tuple(p2[i]),col)
        
        #number the keypoints matched
        #cv2.putText(im1,str(i),tuple(p1[i]),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,thickness=2)    
        #cv2.putText(im2,str(i),tuple(p2[i]),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,thickness=2)    

    #Put images together    
    im3 = np.concatenate((im1,im2),axis=1)
    
    #And print it
    if writename:cv2.imwrite(writename,im3)






#This dude is a custom build feature matching visualisation between two images. original one is buggy.
def show_matches(im1,kp1,im2,kp2,matches):
    '''im1 - 1st image
        kp1 - 1st image keypoints
        im2 - 2nd image 
        kp2 - 2nd image keypoints
        matches - matches object'''
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, :] = im1 
    view[:h2, w1:, :] = im2
    view[:, :, 1] = view[:, :, 0]  
    view[:, :, 2] = view[:, :, 0]
    #ColorMap. From Good to Bad
    cm = plt.get_cmap('viridis')
    colors = [cm((1.*i)/(len(matches)+2)) for i in range(2,len(matches)+2)]


    for (i,m) in enumerate(matches):
        # draw the keypoints
        #print m.queryIdx, m.trainIdx, m.distance
        
        color = tuple([colors[i][0]*255,colors[i][1]*255,colors[i][2]*255])  
        #color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])) , (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)


    cv2.imshow("view", view)
    cv2.waitKey()    



#This guy draws the epipolar lines on two images
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img1
        lines - corresponding epilines ''' 
    r,c,_ = np.shape(img1)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def visualize_2D(positions_ALL,saveas=None):
    '''
    Plots the x and z components of all positions and saves the plot
    positions_ALL   - all positions in xyz coordinates
    saveas          - will be the saved figure, if defined. default = None
    
    '''
    
    #Define colors
    cm = plt.get_cmap('viridis')
    colors = [cm((1.*i)/(len(positions_ALL)+2)) for i in range(2,len(positions_ALL)+2)]

    #start figure
    fig1    = plt.figure()
    ax1     = fig1.add_subplot(111)

    
    for i,positions in enumerate(positions_ALL):
          
        #in the camera coordinate system, y refers to the height
        x = [p[0] for p in positions]
        #y = [p[1] for p in positions]
        z = [p[2] for p in positions]

        ax1.plot(z,x,'o-',markersize=20,c=colors[i],linewidth = 3,label = 'Car {}'.format(i))

    ax1.set_title('2D Projection')
    ax1.set_xlabel('Z [aU]')    
    ax1.set_ylabel('X [aU]')
    ax1.set_xlim([-1,14])
    ax1.set_ylim([-3,3])
    #ax.set_zlabel('Y [aU]')
    ax1.grid()
    ax1.legend(loc='best',ncol=5)
    
    if saveas: fig1.savefig(saveas)    
    
    plt.show()
    
    return
    
    
def visualize_3D(positions_ALL,direction_ALL,saveas=None):
    '''
    Plots the x y and z psoitions of all images and saves the plot
    positions_ALL   - all positions in xyz coordinates [x,y,z]
    directions_ALL  - the directions in xyz coordinates [x,y,z]
    saveas          - will be the saved figure, if defined. default = None
    
    '''
    #Define colors
    cm = plt.get_cmap('viridis')
    colors = [cm((1.*i)/(len(positions_ALL)+2)) for i in range(2,len(positions_ALL)+2)]

    #start figure
    fig2    = plt.figure()
    ax2     = fig2.add_subplot(111, projection='3d')


    
    
    for i in range(len(positions_ALL)):
         
          
        #in the camera coordinate system, y refers to the height
        x = [p[0] for p in positions_ALL[i]]
        y = [p[1] for p in positions_ALL[i]]
        z = [p[2] for p in positions_ALL[i]]

        #get the direction
        xcomp = [p[0] for p in direction_ALL[i]]
        ycomp = [p[1] for p in direction_ALL[i]]
        zcomp = [p[2] for p in direction_ALL[i]]
    

        #xcomp = [[(x[n][0]-x[n-1][0])] for n in range(1,len(x))]
        #ycomp = [[(y[n][0]-y[n-1][0])] for n in range(1,len(y))]
        #zcomp = [[(z[n][0]-z[n-1][0])] for n in range(1,len(z))] 
    
        #normalize the arrowhead
        #length = [1.0/np.sqrt(np.power(xcomp[n],2)+np.power(ycomp[n],2)+np.power(zcomp[n],2)) for n in range(len(xcomp))]
        col=colors[i]
        
        #Actual Plotting
        ax2.quiver(z,x,y,zcomp,xcomp,ycomp,colors=col,label = 'Car {}'.format(i),pivot='tail')#,arrow_length_ratio=length)
        #ax2.quiver(z[:-1],x[:-1],y[:-1],zcomp,xcomp,ycomp,colors=col,label = 'Car {}'.format(i),pivot='tail')#,arrow_length_ratio=length)

    #Axisstuff
    ax2.set_title('3D Positions/Directions')
    ax2.set_xlabel('Z [aU]')
    ax2.set_ylabel('X [aU]')
    ax2.set_zlabel('Y [aU]')
    ax2.set_xlim3d(0, 9)
    ax2.set_ylim3d(-2,2)
    ax2.set_zlim3d(-2,2)
    ax2.grid()
    ax2.legend(loc='best')
    
    if saveas: fig2.savefig(saveas)    
    plt.show()
    
    
    return



#ImagePreprocessing________________________________________

  
def getmask(im,FOV,number=1):
    '''
    Takes the image and returns a matrix defining the mask where features are allowed to be
    im              - image with features
    FOV             - field of view (inside can be keypoints) format=[linebegin,lineend, columnbegin, columnend] in pixel
    offcenterCamera - how far is the image center away from the actual center in px format [cx,cy]
    number          - some algorithms need the mask to have 1's others need 255's. check your algorithm
    '''
    #Here will be the car detector in the future
    
    #only consider feature points in the middle
   
    mask = np.zeros_like(im,dtype=np.uint8)   
    
    mask[FOV[0]:FOV[1],FOV[2]:FOV[3]] = number
    
    #now check the image for Cars, because they move with us they are not allowed to be features
    #Found a pretrained haar cascade. it's not perfect but still okay.
    cascade = cv2.CascadeClassifier('./cars.xml')
    #detect cars
    cars = cascade.detectMultiScale(im,1.1,2)
    
    #Remove cars from Mask to look for features
    for (x,y,w,h) in cars:
        mask[y:(y+h),x:(x+w)] = 0        
        #cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
    
    
    #middle row
    #cy = 360+offcenterCamera[1]
    #mask[(cy-10):(cy+10),FOV[2]:FOV[3]]  = 0
   
    
    
    
    return mask,cars


#FeatureMatching____________________________________________

    
def BF_matcher(des1,des2,norm):
    '''
    Brute force feature matching. Returns a matches objects
    des1 - descriptors of first image
    des2 - descriptors of second image
    norm - cv2.NORM_L1 for SIFT/SURF, cv2.NORM_HAMMING for ORB Features
    '''    
    
    #Brute Force Matche Object [use hamming norm because we used ORB. SIFT/SURF require cv2.NORM_L1f]
    #cross check for fluctuations
    bf = cv2. BFMatcher(norm, crossCheck=True)

    #match descriptors stored
    #only returns the best match for each according to knn
    matches = bf.match(des1,des2)
    #sort matches for their distance
    matches = sorted(matches, key=lambda x:x.distance)
    n_matches = 80
    matches = matches[0:n_matches]
    return matches
    
def BF_matcher_general(des1,des2,params):
    '''
    Brute force feature matching. Returns a matches objects
    des1 - descriptors of first image
    des2 - descriptors of second image
    norm - cv2.NORM_L1 for SIFT/SURF, cv2.NORM_HAMMING for ORB Features
    '''    
    
    #Brute Force Matche Object [use hamming norm because we used ORB. SIFT/SURF require cv2.NORM_L1f]
    #cross check for fluctuations
    if params:
        bf = cv2. BFMatcher(**params)
    else:
        bf = cv2. BFMatcher()
    #match descriptors stored
    #only returns the best match for each according to knn
    matches = bf.match(des1,des2)
    #sort matches for their distance
    matches = sorted(matches, key=lambda x:x.distance)

    return matches
    
def FLANN_matcher(des1,des2):
    '''
    does not work for some reason in opencv 3.
    FLANN feature matching (faster than BF)
    Brute force feature matching
    des1 - descriptors of first image
    des2 - descriptors of second image
    norm - cv2.NORM_L1 for SIFT/SURF, cv2.NORM_HAMMING for ORB Features
    '''    

    #Flann matching is faste than BF
    #FLANN parameters
    #this is okay for ORB
    FLANN_INDEX_KDTREE  = 0
    FLANN_INDEX_LSH     = 6
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #alternative when using ORB descriptors
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    
    
    search_params = dict(checks=50)   # or pass empty dictionary
     
    #Create Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    

    #Match with k=2
    matches = flann.knnMatch(des1,des2,k=2)
    
    
    goodmatches = []

    # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            goodmatches.append(m)
    '''
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    '''
    #img3 = cv22.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    #plt.imshow(img3,),plt.show()
    if len(goodmatches)<10: print 'Only ',len(goodmatches),' found. Be careful.'
    
    return goodmatches

def matchpoints_BF(kps1,kps2, des1, des2,norm):
    '''
    Brute force feature matching. Returns the points in image1 and the points in image 2 that have been matched.
    kps1 - keypoints of first image
    kps2 - keypoints of second image    
    des1 - descriptors of first image
    des2 - descriptors of second image
    norm - cv2.NORM_L1 for SIFT/SURF, cv2.NORM_HAMMING for ORB Features
    '''    
    
    matches = f.BF_matcher(des1,des2,norm)
    

    points1 = np.squeeze(np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2))
    points2 = np.squeeze(np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2))
  
    return points1, points2

def matchpoints_BF_general(kps1,kps2, des1, des2,params):
    '''
    Brute force feature matching. Returns the points in image1 and the points in image 2 that have been matched.
    kps1 - keypoints of first image
    kps2 - keypoints of second image    
    des1 - descriptors of first image
    des2 - descriptors of second image
    norm - cv2.NORM_L1 for SIFT/SURF, cv2.NORM_HAMMING for ORB Features
    '''    
    
    matches = f.BF_matcher_general(des1,des2,params)
    
    points1 = np.squeeze(np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2))
    points2 = np.squeeze(np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2))
  
    return points1, points2

def matchpoints_OpticalFLow(kps1,kps2,im1col,im2col,draw=False,lk_params=dict()):
    '''
    lk_params = dict( winSize  = (21,21),
                         maxLevel = 3,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                         minEigThreshold=0.001))
    '''                     
                     
                    
    '''
    Track Features: Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    Returns p1, p2 , which are the matching points in the two images
    kps1    - features of image 1
    kps2    - features of image 2
    im1col  - image 1
    im2col  - image2
    draw    - indicates that the results are shown
    lk-params - these are the parameters which are used to run the Lucas-Kanade method with pyramids
    '''

                     
                     
    im1      = cv2.cvtColor(im1col,cv2.COLOR_BGR2GRAY)
    im2      = cv2.cvtColor(im2col,cv2.COLOR_BGR2GRAY)

    
    #check if type [[x,y]] or [cv2.KeyPoint]
    if isinstance(kps1[0],np.ndarray): 
            p1           = kps1
            #p_init_guess = kps2
            p_init_guess = kps1
    #else, extract keypoints and convert them to numpy arrays with CV_32F = float32
    else:           
            p1 =  np.asarray([[np.asarray(k.pt)] for k in kps1],dtype=np.float32)   
            p_init_guess =  np.asarray([[np.asarray(k.pt)] for k in kps1],dtype=np.float32)   
        
    #p_init_guess = None
    #calculate optical flow
    p2,status,err = cv2.calcOpticalFlowPyrLK(im1,im2,p1,p_init_guess,**lk_params)
    

    
    #find good points [those who have status=1, o ]
    good_2 = p2[status==1]
    good_1 = p1[status==1]
    

        
    if draw:
        # draw the tracks   26 
         
        mask2 = np.zeros_like(im1col)
        color = np.random.randint(0,255,(len(good_1+5),3))
    
        for i,(new,old) in enumerate(zip(good_2,good_1)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask2 = cv2.line(mask2, (a,b),(c,d), color[i].tolist(), 2)
            im2col = cv2.circle(im2col,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(im2col,mask2)
        
        cv2.imshow('frame',img)
        cv2.waitKey(0)   
    
    return good_1,good_2





#Postprocessing_________________________________________________

def delete_mask(vector, mask):
    '''
    Some alrorithms give a mask back indicating which keypoints have been used.
    This guy takes
    vector  - the points
    maks    - the mask
    
    and returns a vector only containing the points defined by mask
    
    '''
    
    del_mask = []
    for (j,m) in enumerate(mask):
        if m == 0: del_mask.append(j)
            
    goodvector = np.delete(vector,del_mask,axis=0)

    return goodvector



def remove_outliers(matches,kp1,kp2):
    '''
    Removes outlier features using RANSAC
    matches - matches of two images
    kp1 keypoints of first image
    kp2 keypoints of second image
    returns only the good matches
    
    '''
    #Source and destination point coordinates
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M,mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)# treshold should be in [1,10]
    matchesMask = mask.ravel().tolist()
   
    
    goodmatches = delete_mask(matches,matchesMask)
    
    return goodmatches


def getcameramatrix(offcenterCamera=[0,0]):
    '''
    input: offset from image center format [cx,cy]
    
    output: camera matrix
    
    Kamera matrix is usually obtained through calibration and camera specific
    K = [[f,    s,  cx ] \
     [0,    nf, cy]\
     [0,    0,   1]]
     http://www.diva-portal.se/smash/get/diva2:823679/FULLTEXT01.pdf
    '''
     
    #f   = [3408,1945,3463,3599,2165,2727,3870]  
    #cx  = [2104,1344,2080,2104,1296,1632,2064] 
    #cy  = [1560,760,1168,1560,972,1224,1161]     
      
    '''
    One plus one
    htc one
    huawei ascend p7
    mi4
    nexus 6
    samsung galaxy s3
    samsung galaxy s4
    '''
       
    '''
    Oneplus 3:
    IMX 298 Sensor
    f       = 29 mm
    px size = 1.12 um
    d = 6.521 mm
    
    width   = 4720 px <-> 5.28 mm 
    height  = 3600 px <-> 4.03 mm
    
    f in px: = pixelwidth [960 px] * f [29 mm] / chipwidht[5.28 mm]
    
    fx  = 960 *29/5.29 ~ 5260 px
    fy  = 720 *29/4.03 ~
    cx  = 960/2 = 480
    cy  = 720/2 = 360
    
    

    '''  
    cxoff=offcenterCamera[0]
    cyoff=offcenterCamera[1]
    
    fx = 5260         #focal lenght in px
    fy = 5180         #focal lenght in px
    cx = 480+cxoff    #assume center in center. if put to 0, results are weird
    cy = 360+cyoff    #assume center in enter. if pu tto 0, results are weird
    s  = 0            #skew

    #Reference RecoverPose opencv: http://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#ga30ccb52f4e726daa039fd5cb5bf0822b&gsc.tab=0     
       
    K = np.array([[fx,s,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)   
    
    #K_oneplusone  = np.array([[3408, 0, 2104],[0, 3408, 1560], [0,0,1] ])
    
    return K



def get_Rt(keypoints1,keypoints2,image1,image2,descriptors1=None,descriptors2=None,offcenterCamera=[0,0],matchalgo='opticalflow',norm=cv2.NORM_HAMMING,onlyforward=True,maxrot_xyz=[20,40,20]):
    '''
    This guy calculates the Rotation R and the Translation t from image to image
    
    IN:
    
    keypoints1          - keypoints of image 1 (can be only points [x,y] or of type cv2.keyPoints from SURF/SIFT algorithms)
    keypoints2          - keypoints of image 2 (can be only points [x,y] or of type cv2.keyPoints from SURF/SIFT algorithms)
    image1              - image 1
    image2              - image 2
    descriptors1        - descriptors of first keypoints (if you got them from SIFT/SURF) (only needed for BruteForce Matcher) default = None
    descriptors2        - descriptors of second keypoints (if you got them from SIFT/SURF) (only needed for BruteForce Matcher) default = None
    offcenterCamera     - offset from image center format [cx,cy]. Needed to get the right camera Matrix default = [0,0]
    matchalgo           - algorithm used to match features in image1 and image2 possibilities: 'opticalflow'/'bruteforce'. if you use bruteforce you also need descriptors
    norm                - Norm used for matchalgo, only important for Bruteforce. default is cv2.NORM_HAMMING for keypoints from ORB. use cv2.NORM_L2 for SURF 
    onlyforward         - do we accept translation vectors pointing backwards? default = True
    maxrot_xyz          - max xyz rotations allowed in [degrees], if bigger, we do the R_t again with new parameters. default = [20,40,20]

    OUT:
    R                   - This is the output rotation matrix
    t                   - Thsi is the normed output translation vector 
    NRpoints            - This is the number of points that have been used to detect R and t
    points1             - This is the list of points in image1 that were used to recover R and t
    points2             - This is the list of points in image2 that were used to recover R and t
    
    '''    
    
    
    
    tries       = 0    
    maxtries    = 3
    #lk params
    windowsize = (21,21)
    maxlklevel = 3
    
    
    
    while tries <maxtries:
        #TRACK/MATCH FEATURES
        if matchalgo == 'bruteforce':
            image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
            #standard norm = cv2.NORM_HAMMING. use cv2.NORM_L2 for SURF 
            points1,points2 = f.matchpoints_BF(keypoints1,keypoints2,descriptors1,descriptors2,norm)
        
        elif matchalgo == 'opticalflow':
            lk_params = dict( winSize  = windowsize,
                                maxLevel = maxlklevel,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                                minEigThreshold=0.001)
            #calculate matching points with Optical Flow 
            points1,points2 = f.matchpoints_OpticalFLow(keypoints1,keypoints2,image1,image2,False,lk_params)
    
    
          
    
    
        if not points1.any():
            print 'no points found'
            return
            
        #Get the Camera Matrix
        K = f.getcameramatrix(offcenterCamera)
    
        #RECOVER THE POSE
        #METHOD 1:
        '''
        Just to check if result is reasonable.
        #Get Fundamental Matrix
        F,maskF = cv2.findFundamentalMat(points1, points2,cv2.FM_RANSAC)
       
        #Get the points that were actually used according to FM_RANSAC       
        points1_used2 = f.delete_mask(points1,maskF)
        points2_used2 = f.delete_mask(points2,maskF)
        
        #Calc essential matrix ourself E = K.T*F*K     
        E2 = np.dot(np.dot(np.transpose(K),F),K)
        
        #Then recover pose
        (_,R2,t2,_) = cv2.recoverPose(E2, points1_used2, points2_used2)
        #NICE. Both Methods yield roughly same result.
        '''
        
    
        
        #METHOD 2:
        #More direct
        
        #Calculate the Essential Matrix using Ransac to eliminate outliers.
        #Works oly with <60% outliers though
        E,mask_E = cv2.findEssentialMat(points1, points2, K, cv2.FM_RANSAC,0.999,1.0);
    
        
        
        points1_used_E = f.delete_mask(points1,mask_E)
        points2_used_E = f.delete_mask(points2,mask_E)
        
        
        #Recover Pose. Does chirality check itself
        (NRpoints,R,t,maskPose) = cv2.recoverPose(E, points1_used_E, points2_used_E)
        #amount of points
        #R is the Rotation Matrix 
        #t is the translation vector
        # maskPose is i teh points used during recovery
        
        tries = tries + 1
        
        r = f.decompose_Rot(R,degrees=True)        
        if (abs(r[0])>maxrot_xyz[0] or abs(r[1])>maxrot_xyz[1] or abs(r[2])>maxrot_xyz[2]):
            #lk params
            windowsize = (windowsize[0]+3,windowsize[1]+3)
            maxlklevel = maxlklevel+1
            
            #then run again
        else:
            #otherwise don't do the loop again. we've been succcessful
            tries = maxtries
    
    
    maskPose = map(lambda x: 1 if x==255 else 0,maskPose )
    
    #Those are the points used during the recovery
    points1_used_Rt = f.delete_mask(points1_used_E,maskPose)
    points2_used_Rt = f.delete_mask(points2_used_E,maskPose)

    
    
    #Sanity Check
    #The [R|t] equations allow disambiguity in the sign of t. But we know that z must be positive soo
    if (onlyforward and t[2]<=0):     t = np.multiply(-1,t)
        
    return R,t,NRpoints,points1_used_Rt,points2_used_Rt

def get_rot_pos_dir(R_list,t_list,rotmax_xyz=[40,60,10],logfile=None):
    '''
    Gives back the actual Rotation, Position and Direction of the Steps
    IN:
    R_list      - the list of Rotation Matrices in between consecutive images
    t_list      - the list of Translation Vectors in between consecutive images
    rotmax_xyz  - maximal allowed Rotations around x, y and z axis. if the angle is bigger then we assume wrong measurement and use the previous result
    logfile     - if defined, then this is where the output log goes. format ./output/logfile.txt  . default  = None
    
    OUT
    rotation    - actual cumulated rotation of step i compared to beginning [[1,0,0],[0,1,0],[0,0,1]]
    position    - actual cumulated position of step i compared to beginning [0,0,0]
    direction   - actual direction of step i, 0: [0,0,1]
    
    '''
    
    
    
    #Those guys are going to contain the result
    position    = []
    rotation    = []


    #check if we skipped an unreasonable point
    skipped         = 0
    skippedtotal    = 0
       
    if logfile:
        print ' '
        logfile.write('\n')

        
    #loop thourh all points    
    for l in range(0,len(R_list)):
                      
        #if it's the first, load first
        if l == 0:
            position.append(t_list[0].tolist())
            rotation.append(R_list[0].tolist())
            continue

        #if not
        #get rotation in degrees
        r = f.decompose_Rot(R_list[l],degrees=True)

        #decide if the roation is reasonable or wrong
        if (abs(r[0])>rotmax_xyz[0] or abs(r[1])>rotmax_xyz[1] or abs(r[2])>rotmax_xyz[2]):
            #if yes, take the last reasonable translation and rotation               
            thisstep = l-skipped-1
            skipped = skipped+1 
            skippedtotal = skippedtotal+1
            #overwrite Faulty Values
            t_list[l] = t_list[thisstep]
            R_list[l] = R_list[thisstep]
        else:
            thisstep = l
            skipped  = 0    

        if logfile:
            print 'step '+str(l),' : t= ', t_list[l][0], t_list[l][1],t_list[l][2],';\t xyzrot:\t{:2f}\t{:2f}\t{:2f}'.format(r[0],r[1],r[2]),';\t skipped= ',skipped    
            text = '\n step {}: t= {:4f} \t {:4f} \t {:4f} \t xyzrot:\t {:2f} \t {:2f} \t {:2f}; \t skipped= {}'.format(l,float(t_list[l][0]), float(t_list[l][1]),float(t_list[l][2]),r[0],r[1],r[2],skipped)
            logfile.write(text)
               
        
        #NOW HERE actually calculate the next position and orientation             
        t_new = np.add(position[l-1],np.dot(R_list[thisstep],t_list[thisstep]))        
        R_new = np.dot(rotation[l-1],R_list[thisstep])
    
        #Calculate Directions from cumulative Rotation

        position.append(t_new)
        rotation.append(R_new)
    
    #Close the file
    direction = [np.dot(rot,np.array([[0],[0],[1]])) for rot in rotation]
    
    
    return rotation,position,direction


def recover_pose_custom(K,E):
    #Singular Value Decomposition
    # A = u * np.diag(s) * v
    
    # u, v unitary, s 1d array of singular values
    U,S,V = np.linalg.svd(E,full_matrices=True)    
    #s = np.zeros((np.size(U,0),np.size(V,0)),dtype=complex)    
    #s[:len(S),:len(S)] = np.diag(S)
    #np.dot(U,np.dot(s,V))

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Wt = np.array([[0,1,0],[-1,0,0],[0,0,1]])    

    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])

    tx = np.dot(np.dot(U,Z),np.transpose(U))
    
    #t = [TX,TY,TZ], tx = [[0,-TZ, TY],[TZ, 0, -TX],[-TY, TX, 0]]
    t_1 = np.array([[tx[2,1]],[tx[0,2]],[tx[1,0]]])
    t_2 = np.multiply(-1,t_1)
    
    R_1 = np.dot(U,np.dot(Wt,V))
    R_2 = np.dot(U,np.dot(W,V))
    
    return R_1,R_2,t_1,t_2
    
    
    
    
    
def Rt(K1,K2,F):
    '''
    Returns R and t from
    K1 - Kamera Calibration Matrix from Image 1
    K2 - Kamera Calibration Matrix from Image 2 (Can be the same)
    F - Fundamental Matrix of the Transformation
    
    Missing chirality test for now.
    '''
    
    #essential matrix
    #E = K'.T * F * K
    E  = np.dot(np.dot(np.transpose(K1),F),K2)

    #Singular Value Decomposition
    # A = u * np.diag(s) * v
    # u, v unitary, s 1d array of singular values
    U,S,VT = np.linalg.svd(E)
    
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Wt = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    
    #U * W.-1 * V.T
    R  = np.dot(np.dot(U,Wt),np.transpose(V))
    
    
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])

    tx = np.dot(np.dot(U,Z),np.transpose(U))
    
    #t = [TX,TY,TZ], tx = [[0,-TZ, TY],[TZ, 0, -TX],[-TY, TX, 0]]
    t = np.array([[tx[2,1]],[tx[0,2]],[tx[1,0]]])

    #Chireility Check is missing here    
    #Matrices = [E,U,S,V,R,tx]
    #name = ['E','U','S','V','R','t']
    '''
    for i in range(len(Matrices)):
        print name[i]
        print Matrices[i]
        print '  '
    '''
    
    
    return (R,t)


def decompose_Rot(R,degrees=False):
    '''
    Takes 3D rotation matrix R and gives back 3 Euler Angles x,y,z
    return in RAd from - pi to pi except if degrees=True
    R       - rotation Matrix
    degrees - give back rotation in degrees or rad? can be True/False. default = False <-> Rad
    '''
    
    x = math.atan2(R[2,1],R[2,2])
    y = math.atan2(-R[2,0],np.sqrt(np.power(R[2,1],2)+np.power(R[2,2],2)))
    z = math.atan2(R[1,0],R[0,0])
    
    if degrees:
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)
    
    return [x,y,z]
    
def get_Rt_general(keypoints1,keypoints2,image1,image2,K,descriptors1=None,descriptors2=None,matchalgo='opticalflow',params=None):
    '''
    This guy calculates the Rotation R and the Translation t from image to image
    
    IN:
    
    keypoints1          - keypoints of image 1 (can be only points [x,y] or of type cv2.keyPoints from SURF/SIFT algorithms)
    keypoints2          - keypoints of image 2 (can be only points [x,y] or of type cv2.keyPoints from SURF/SIFT algorithms)
    image1              - image 1
    image2              - image 2
    K                   - camera Matrix K = = [[f,    s,  cx ],[0,    nf, cy],[0,    0,   1]]
    descriptors1        - descriptors of first keypoints (if you got them from SIFT/SURF) (only needed for BruteForce Matcher) default = None
    descriptors2        - descriptors of second keypoints (if you got them from SIFT/SURF) (only needed for BruteForce Matcher) default = None
    matchalgo           - algorithm used to match features in image1 and image2 possibilities: 'opticalflow'/'bruteforce'. if you use bruteforce you also need descriptors. default is opticalflow
    params              - params used in the Matching algorithm. for brute force it's the NORM, eg. cv2.NORM_HAMMING

    OUT:
    R                   - This is the output rotation matrix
    t                   - Thsi is the normed output translation vector 
    points1             - This is the list of points in image1 that were used to recover R and t
    points2             - This is the list of points in image2 that were used to recover R and t
    
    '''    
        
    

    #TRACK/MATCH FEATURES
    if matchalgo == 'bruteforce':
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        #standard norm = cv2.NORM_HAMMING. use cv2.NORM_L2 for SURF 
        points1,points2 = f.matchpoints_BF_general(keypoints1,keypoints2,descriptors1,descriptors2,params)
    
    elif matchalgo == 'opticalflow':
        '''
        lk_params = dict( winSize  = windowsize,
                            maxLevel = maxlklevel,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                            minEigThreshold=0.001)
        '''
        #calculate matching points with Optical Flow 
        if params:
            points1,points2 = f.matchpoints_OpticalFLow(keypoints1,keypoints2,image1,image2,False,params)
        else:
            points1,points2 = f.matchpoints_OpticalFLow(keypoints1,keypoints2,image1,image2,False)

    
    
    if not points1.any():
        print 'no points found that match'
        return
    

        
    #Calculate the Essential Matrix using Ransac to eliminate outliers.
    #Works oly with <60% outliers though
    E,mask_E = cv2.findEssentialMat(points1, points2, K, cv2.FM_RANSAC,0.999,1.0);
    
        
    #points that were actually used   
    points1_used_E = f.delete_mask(points1,mask_E)
    points2_used_E = f.delete_mask(points2,mask_E)
        
    if not points1_used_E.any():
        print 'no points found to find Essential Matrix'
        return
        
    #Recover Pose. Does chirality check itself
    (_,R,t,maskPose) = cv2.recoverPose(E, points1_used_E, points2_used_E)
    #amount of points
    #R is the Rotation Matrix 
    #t is the translation vector
    # maskPose is i the points used during recovery
        
    maskPose = map(lambda x: 1 if x==255 else 0,maskPose )
    
    #Those are the points used during the recovery
    points1_used_Rt = f.delete_mask(points1_used_E,maskPose)
    points2_used_Rt = f.delete_mask(points2_used_E,maskPose)
    
    if not points1_used_Rt.any():
        print 'no points found to recover Rt'
        return


        
    return R,t,points1_used_Rt,points2_used_Rt







#_____________________________________________________

#end of file f.py