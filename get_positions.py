# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:58:54 2016

@author: tobias hoelzer

If you execute this file with
python get_positions.py
in a linux terminal it will

1. load 5 images series of 10 images contained in ./pics
2. calculate their keypoints using FAST (you can also use any other and change the matching)
3. match their keypoints using calcOpticalflowLK (you can also use Brute Force if you used e.g. SURf/SIFT)
4. calculate the rotation R and the translation t
5. Visualize the result

OUTPUT
./kps/          - images with their keypoints, field of view, other cars and fluchtpunkt
./matches/      - concatenated images with their matches
./result/       - 2D and 3D visualisation of result
./logfile.txt   - debug logfile
"""

#Dependencies
import numpy as np
import cv2 #using opencv 3
from matplotlib import pyplot as plt
import scipy as sp
import f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os


#Create folders for output
paths1 = r'./output/kps'
paths0 = r'./output/matches'
paths2 = r'./output/result' 
if not os.path.exists(paths0): os.makedirs(paths0)
if not os.path.exists(paths1): os.makedirs(paths1)
if not os.path.exists(paths2): os.makedirs(paths2)   



#Those guys will contain 5 entries which will be lists of the values
keypoints_ALL   = []    #keypoints of the images
descriptors_ALL = []    #descriptors of the images
Rs_ALL          = []    #the Rotation Matrix of each image
ts_ALL          = []    #the Translation Vector of each image
rotation_ALL    = []    #the direction of each image
direction_ALL   = []    #the Cumulated Rotation Matrix of each image
position_ALL    = []    #the cumulated (absoulute) translation vector of each image
nrPoints_ALL    = []    #nr of points used for matching per iimage
othercars_ALL   = []    #list of rectangles around the other cars




#For all cars:
cars = 5
nr_images = 10


#other data sets__________________________________
#I used this to check if i can use the algorithm for other datasets
#pics_w
#offcenterCamera = [[-40,-240]]
#FOV = [[0,478,0,638]]
#pics_w2
#FOV = [[0,1918,0,2558]]
#offcenterCamera = [[980,440]]
#offcenterCamera = [[-172,-140]]
#FOV = [[0,375,0,1240]]
#FAST_hyper = [50]
#_________________________________________________

#CAR PARAMETERS INDIVIDUAL
#car Field of View in Pixels. Begin Row, End Row, Beginn Column, End Column
FOV = [[150,670,0,820],[0,520,0,830],[150,670,30,800],[50,620,0,820],[0,690,45,840]]
#center of image  offset (x,y) [px]
offcenterCamera = [[-66,102],[-56,-7],[-65,114],[-86,61],[-88,102]]
#Car Hyperparemeter for FAST algorithm
FAST_hyper = [40,45,30,45,50]

#Define the reference image to locate the first image of the other cars
image0 = cv2.imread('./pics/0_0.jpg')

#______________________________________________________________________________
#now go!
for i in range(cars):

    #import images of all rides
    images      = [cv2.imread('./pics/'+ str(i) + '_' + str(j) + '.jpg') for j in range(nr_images)]    
    #images = [cv2.imread('./pics_w2/'+str(j)+'.jpg') for j in range(nr_images)]
    #images = [cv2.imread('./01/image_0/'+str(j).zfill(6)+'.png') for j in range(nr_images)]
    
    
    #0 - FEATURE DETECTION#########################################################
    keypoints   = []
    descriptors = []
    
    #Car detection
    othercars        = []    
    
    #For all rides
    for j,im in enumerate(images):

      
        #put the images in grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        #Take only the interesting points. Remove cars and narrow the field of view
        Mask,cars_detected = f.getmask(gray,FOV[i],number=1)#use 255 for SURF algorithm
        
        #detectors: SIFT, SURF, KAZE, AKAZE, BRISK, ORB, FAST, BRISK, 

        #THIS PART NEEDS TO BE CHANGED TO CHANGE FEATURE DETECTOR
        #______________________________________________________________________
        #FAST detector [Because it shows the best results and is FAST. haha. such funny. much FAST. wow]
        detector = cv2.FastFeatureDetector_create(FAST_hyper[i])         
        #detector = cv2.xfeatures2d.SURF_create(700)
        #detector = cv2.ORB_create()
        #detector = cv2.BRISK_create()
        
        #use either A or B, depending on the algorithm
        #A_________________________________
        #Algorithms that only detect points, without descriptors:        
        #FAST, goodFeaturesToTrack
        kps = detector.detect(gray,Mask)
        #kps = cv2.goodFeaturesToTrack(gray,maxCorners=500,qualityLevel=0.01,minDistance=10,mask = Mask)
        #goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
        #quality level ~ 0.01,minDistance ~10 [px], blockSize= 2-4
        des = None
        #B_________________________________
        #Algorithms taht detect keypoints and descriptors
        #kps,des = detector.detectAndCompute(gray,Mask)
        #______________________________________________________________________
            
        
        #Draw Features on Image and show it, including FOV and cars to check if the mask works
        name = './output/kps/car_'+str(i)+'_im_'+str(j)+'.jpg'
        f.DrawFASTKeypointsAndFOV(gray,kps,im,FOV[i],cars_detected,offcenterCamera[i],savename=name)
        
        #Save keypoints and descriptors in the list (if they exist)
        keypoints.append(kps)
        if np.any(des):
            descriptors.append(des)
        #Save the other cars detected in the image
        othercars.append(cars_detected)
        
        
        
    #Save list of Keypoints in the global list
    keypoints_ALL.append (keypoints)
    #SAve list of descriptors in global list
    if np.any(descriptors):
        descriptors_ALL.append(descriptors)
    
    #Save list of other cars in global list
    othercars_ALL.append(othercars)
         
    
    
    
    #UNDISTORT IMAGES
    '''
    not possible, we really need the camera intrinsics to make use of this
    http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
    '''
    
    
    
    
    
    #1 - INITIALIZE THE FIRST POSITION REFERING TO image 0 <-> [0,0,0]#########
    
    #If it's the first car we initialize with the standard direction/vector
    if i ==0:
        Rs      = [np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])]
        ts      = [np.array([[0.0],[0.0],[0.0]])]
        
    #If it's another car then we compare to the first car defined in image0
    else:
        #Check if descriptors exist. if they do, load them
        if np.any(descriptors_ALL):
            des1 = descriptors_ALL[0][0]
            des2 = descriptors_ALL[i][0]
        else:
            des1 = None
            des2 = None
            
        #Find the Rotation and Translation of the first images of the cars 1-4.Details are described below
        R,t,_,p1used,p2used = f.get_Rt(keypoints_ALL[0][0],keypoints_ALL[i][0],image0,images[0],des1,des2,offcenterCamera[i],matchalgo='opticalflow',onlyforward=False,maxrot_xyz = [15,40,15])

        #Draw and save the matched points and how they moved from image to image
        savename = './output/matches/init_car_{}.jpg'.format(i)        
        f.Draw_moved_points(image0,images[0],p1used,p2used,[FOV[0],FOV[i]],[offcenterCamera[0],offcenterCamera[i]],othercars_ALL[0][0],othercars_ALL[i][0],writename=savename)

        
        #Since the cars are about the same position and the t vector is normed, we need to readjust the z part of it by a reasonable, but arbitrary scaling factor
        t[2] = np.multiply(0.2,t[2])
                
        #then save R and t
        Rs      = [np.array(R)]
        ts      = [np.array(t)]
        
    

    #2 - FEATURE (TRACKING/MATCHING) ##########################################
    #3 - RECOVER POSE #########################################################

    #Nr Points used to recover R and t. This is important to see if the recovery had a chance to work {only used for debugging}
    nrPoints    = [0]
    
    #unless it's the last image(we need an n+1 image) and cannot track the last one
    for j in range(len(images)-1):
        
       
        #Check if descriptors exist. If yes, load them
        if np.any(descriptors_ALL):
            des1 = descriptors_ALL[i][j]
            des2 = descriptors_ALL[i][j+1]
        else:
            des1 = None
            des2 = None
        
        
        #Recover the R and t
        R,t,NRpoints,p1_used,p2_used = f.get_Rt(keypoints_ALL[i][j],keypoints_ALL[i][j+1],  #keypoints
                                                images[j],images[j+1],      #images
                                                des1,des2,                  #descriptors (only needed for FLANN and BF algos)
                                                offcenterCamera[i],         #the offsett of the image center
                                                matchalgo='opticalflow',    #algorithm used to match features in image1 and image2 'opticalflow'/'bruteforce'
                                                norm = cv2.NORM_L1,         #Norm used for matchalgo, only important for Bruteforce
                                                onlyforward=True,           #do we accept translation vectors pointing backwards
                                                maxrot_xyz = [15,30,15])    #max xyz rotations allowed [degrees], if bigger, we do the R_t again
        #Outputvalues
        #R is the Rotation Matrix [Radians]
        #t is the translation Vector [Normed]
        #NRpoints is the amoutn of points used to recover the pose
        #p1 and p2 used are the points used in the flow/BF matching
        
        #Draw and save the output images and how the points used to recover R and t move
        name = './output/matches/car_'+str(i)+'_step_'+str(j+1)+'.jpg'
        #name = './pics_w2/debug_step_'+str(j+1)+'.jpg'       
        f.Draw_moved_points(images[j],images[j+1],p1_used,p2_used,FOV[i],offcenterCamera[i],othercars_ALL[i][j],othercars_ALL[i][j+1],writename=name)
        
        
        #Append the Results R and t and the amount of points used
        Rs.append(R)
        ts.append(t)
    
 
    
    #Save all the list in the global list
    nrPoints_ALL.append(nrPoints)
    Rs_ALL.append(Rs)
    ts_ALL.append(ts)


 

    #4 - CALCULATE POSITIONS AND ROTATIONS AND DIRECTIONS( with sanity check)         
    
    
    #write the logfile  with R, t angles and if a step was skipped. put None  
    debugname = './output/logfile.txt'
    
    if (debugname and i == 0):
        logfile = open(debugname,'w')
    elif (debugname and i> 0):
        logfile = open(debugname,'a')
    if logfile:
        logfile.write('Car {}'.format(i))
        print 'Car {}'.format(i)
    

    #Calculate the actual i-th Rotation, Position and Direction
    R_pos,t_pos,d_pos = f.get_rot_pos_dir(Rs_ALL[i],ts_ALL[i],rotmax_xyz=[15,30,15],logfile=logfile)
    
    #Close the logfile
    if logfile: logfile.close()
        
        
    #Save the data in the global lists
    rotation_ALL.append(R_pos)
    position_ALL.append(t_pos)
    direction_ALL.append(d_pos)
    
#5 - VISUALISATION ############################################################


#VISUALISATION


#Filenames
saveas1 = './output/result/2D.png'
saveas2 = './output/result/3D.png'


#Plot the 2D projection
f.visualize_2D(position_ALL,saveas=saveas1)
#Plot the 3D figure including the directions
f.visualize_3D(position_ALL,direction_ALL,saveas=saveas2)




