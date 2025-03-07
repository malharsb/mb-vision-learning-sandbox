import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from opts import get_opts
from matchPics import matchPics
from planarH import *
opts = get_opts()



#Write script for Q4.2x

# Import images
# panol = cv2.imread('../data/pano_left.jpg')
# panor = cv2.imread('../data/pano_right.jpg')
panol = cv2.imread('pano_left1.jpg')
panor = cv2.imread('pano_right1.jpg')
print("panol shape: {} and panor shape: {}".format(panol.shape, panor.shape))



# Add a buffer to left image
top = np.zeros((200,1457,3)).astype('uint8')
bottom = np.zeros((200,1457,3)).astype('uint8')
right = np.zeros((1493,1200,3)).astype('uint8')
panobuf = np.vstack((top,panol,bottom))
panobuf = np.hstack((panobuf, right))



# Transpose the images
# print("panobuf shape: {} and panor shape: {}".format(panobuf.shape, panor.shape))
panobuf = panobuf.transpose(1,0,2)
panor = panor.transpose(1,0,2)
# print("panobuf shape: {} and panor shape: {}".format(panobuf.shape, panor.shape)



# Calculate matches
print("\nCalculating matches. takes a while..")
matches, locs1, locs2 = matchPics(panobuf, panor, opts)
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



# Calculate bestH2to1 using Ransac
bestH2to1, inliers = computeH_ransac(m1, m2, opts) # Toggel between m1 and m2 to get different images
print("Calculating Ransac...")
print("bestH2to1: \n{}".format(bestH2to1))
# np.save("H.npy",bestH2to1)
# bestH2to1 = np.load("H.npy")



# Pre-process and compositeH
panor = panor.transpose(1,0,2)
hei, wid = panobuf.shape[:2]
panor_warped = cv2.warpPerspective(panor, bestH2to1, (hei,wid))

composite_img = compositeH(bestH2to1, panobuf, panor)
cv2.imwrite("composite_img.jpg", composite_img) 



