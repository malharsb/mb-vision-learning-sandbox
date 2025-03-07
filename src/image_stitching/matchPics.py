import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	
	#Convert Images to GrayScale	
	I1m = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY) # HxWx3 -> HxW
	I2m = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY) # HxWx3 -> HxW
	
	#Detect Features in Both Images
	locs1 = corner_detection(I1m, sigma) # nx2
	locs2 = corner_detection(I2m, sigma) # nx2
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1m, locs1) # dx2 , p<=n
	desc2, locs2 = computeBrief(I2m, locs2) # dx2 , p<=n

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio) # px2

	return matches, locs1, locs2
