import numpy as np
import cv2
from matchPics import matchPics

# malhar
import scipy
import skimage.feature
import matplotlib.pyplot as plt
from helper import plotMatches
from opts import get_opts
opts = get_opts()

def plotMatches(im1,im2,matches,locs1,locs2,j):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	plt.axis('off')
	skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
	# plt.savefig('../outputs/q2_1_6'+str(j)+'.png') # malhar
	plt.show()
	return


#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
print("img shape: ", img.shape)

# Add condition for gray scale
# Already converting to greyscale in matchPics

hist = []
print("img shape: ", img.shape)

save_list = np.array([1,5,10])

for i in range(36):

	#Rotate Image
	img_rot = scipy.ndimage.rotate(img, i*10)
	
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, img_rot, opts)

	#Update histogram
	hist.append(matches.shape[0])

	# Save
	for z in save_list:
		if z==i:
			# plotMatches(img, img_rot, matches, locs1, locs2, z)
			# plt.savefig('../outputs/rot'+str(z))

	print("i is {} and matches are {}".format(i, matches.shape[0]))



print("hist: \n", hist)
hist = np.array(hist)

#Display histogram
_ = plt.bar(np.arange(len(hist)), hist) 

# plt.savefig('../outputs/Q2_1_6')
plt.show()

