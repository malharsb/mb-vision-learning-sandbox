import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import *



#Write script for Q2.2.4
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')



# # Debug
# cv_cover = cv2.imread('../data/cv_cover.jpg')
# matches, locs1, locs2 = matchPics(cv_cover, cv_cover, opts)
# num_matches = matches.shape[0]
# m1 = np.zeros((num_matches,2))
# m2 = np.zeros((num_matches,2))
# j=0
# for i in matches:
#     # print(i, " ", i.shape)
#     m1[j] = locs1[i[0]]
#     m2[j] = locs2[i[1]]
#     j+=1	

# # ComputeH works fine!!

# Htest = computeH(m1,m2)
# print("Htest: \n",Htest)
# Htest = computeH_norm(m1,m2)
# print("Htest: \n",Htest)



# Transpose images and calculate matches using matchPics
cv_cover = cv_cover.transpose(1,0,2)
cv_desk = cv_desk.transpose(1,0,2)
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
print("locs1: {} and locs2: {} and matches: {}".format(locs1.shape, locs2.shape, matches.shape))



# create vectors of matched points
num_matches = matches.shape[0]
m1 = np.zeros((num_matches,2))
m2 = np.zeros((num_matches,2))
j=0
for i in matches:
    m1[j] = locs1[i[0]]
    m2[j] = locs2[i[1]]
    j+=1

    	

# Calculate H using cv2
# H,other = cv2.findHomography(m2, m1, method=cv2.RANSAC)
# hei, wid = cv_desk.shape[:2]
# rw,rh = cv_cover.shape[:2]
# hp_cover = cv2.resize(hp_cover, (rw,rh)) 
# hp_warped = cv2.warpPerspective(hp_cover, H, (hei,wid))
# print("H using cv2: \n{}".format(H))
# cv2.imshow('hp_warped', hp_warped)
# cv2.imwrite("../outputs/warped_ransac2.jpg", hp_warped) 



# Calculate bestH2to1 using Ransac
H2 = computeH_norm(m1,m1) # Sanity
print("H using Ransac same image: \n{}".format(H2))
bestH2to1, inliers = computeH_ransac(m1, m2, opts) # Toggel between m1 and m2 to get different images
print("bestH2to1: \n{}".format(bestH2to1))



# Save bestH2to1
# np.save('../outputs/bestH2to1.npy', bestH2to1)
# print("Saved")



# Plot warpPerspective
# bestH2to1 = np.load('../outputs/bestH2to1.npy')
hei, wid = cv_desk.shape[:2]
rw,rh = cv_cover.shape[:2]
hp_cover = cv2.resize(hp_cover, (rw,rh))
hp_warped = cv2.warpPerspective(hp_cover, bestH2to1, (hei,wid))
cv2.imshow('hp_warped', hp_warped)
# cv2.imwrite("../outputs/warped.jpg", hp_warped) 



# Resize hp_cover
rw,rh = cv_cover.shape[:2]
hp_cover = cv2.resize(hp_cover, (rw,rh))
# Composite Image and Print
# bestH2to1 = np.load('../outputs/bestH2to1.npy')
composite_img = compositeH(bestH2to1, cv_desk, hp_cover)
cv2.imshow('composite_img', composite_img)
cv2.imwrite("composite_img.jpg", composite_img) 

