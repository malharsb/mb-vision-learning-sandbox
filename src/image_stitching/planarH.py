import numpy as np
import cv2

# remove 
from matchPics import matchPics
from opts import get_opts


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	assert(x1.shape == x2.shape)
	n = x1.shape[0]

	A = np.zeros((2*n, 9))
	j=0

	# Populate matrix A to size 8x9
	for i in range(4): # CHANGE!!
		A[j,:]   = np.array([-x1[i,0], -x1[i,1], -1, 0, 0, 0, x1[i,0]*x2[i,0], x1[i,1]*x2[i,0], x2[i,0]])
		A[j+1,:] = np.array([0, 0, 0, -x1[i,0], -x1[i,1], -1, x1[i,0]*x2[i,1], x1[i,1]*x2[i,1], x2[i,1]])
		j+=2

	# Use svd instead of eig because eig requires an MxM matrix
	_, _, vh = np.linalg.svd(A)

	# Extract the last 'row' as svd gives v.T
	H2to1 = np.array(vh[-1,:]).reshape((3,3))
	h33=H2to1[2][2]
	H2to1 /= h33
	assert(H2to1.shape ==(3,3))

	return H2to1



def computeH_norm(x1, x2):
	#Q2.2.2

	assert(x1.shape==x2.shape)
	n = x1.shape[0]

	#Compute the centroid of the points
	x1h = np.hstack((x1,np.ones((n,1))))
	x2h = np.hstack((x2,np.ones((n,1))))

	x1_c = np.sum(x1, axis=0) / n 
	x2_c = np.sum(x2, axis=0) / n
	
	#Shift the origin of the points to the centroid
	T1t = np.array([[1,0,-x1_c[0]],[0,1,-x1_c[1]],[0,0,1]])
	T2t = np.array([[1,0,-x2_c[0]],[0,1,-x2_c[1]],[0,0,1]])

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	dist1 = np.linalg.norm(x1 - x1_c, axis=1)
	dist2 = np.linalg.norm(x2 - x2_c, axis=1)

	# print("dist1 shape is {} and dist2 shape is {}\n".format(dist1, dist2))
	max1 = np.max(dist1)
	max2 = np.max(dist2)

	# print(x1h_max, x1h_max, x2h_max, x2h_max)
	T1s = np.array([[ 2**0.5/max1 , 0, 0], [0, 2**0.5/max1, 0], [0,0,1]])
	T2s = np.array([[ 2**0.5/max2 , 0, 0], [0, 2**0.5/max2, 0], [0,0,1]])
	
	#Similarity transform 1
	T1 = np.dot(T1s, T1t)
	x1_n = np.dot(T1, x1h.T).T

	#Similarity transform 2
	T2 = np.dot(T2s, T2t)
	x2_n = np.dot(T2, x2h.T).T

	#Compute homography
	H = computeH(x1_n, x2_n)

	#Denormalization
	H2to1 = np.linalg.multi_dot([np.linalg.inv(T1), H, T2])
	assert(H2to1.shape==(3,3))
	
	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	inliers = 0
	assert(locs1.shape==locs2.shape)
	num_matches = locs1.shape[0]

	x1 = np.zeros((4,2))
	x2 = np.zeros((4,2))

	inliersL =[]

	bestH2to1 = np.zeros((3,3))

	for i in range(max_iters):

		# randomly sample 4 indices
		rand_idx = (np.random.random(4)*num_matches).astype(int)
		# print("rand: ", rand_idx)
		
		# Populate x1, x2 with random
		for j in range(len(rand_idx)):
			x1[j] = locs1[rand_idx[j]]
			x2[j] = locs2[rand_idx[j]]

		H2to1 = computeH_norm(x1, x2)

		# Conver to homogeneous coordinates
		locs1_tilde = np.hstack((locs1, np.ones((num_matches,1))))
		locs2_tilde = np.hstack((locs2, np.ones((num_matches, 1))))

		# Calculate prejection:
		locs1_pred = np.dot(H2to1, locs2_tilde.T).T
		locs1_pred = locs1_pred/ locs1_pred[:,2].reshape((-1,1))
		# locs1 shape is nx3

		dist = pow((locs1_pred - locs1_tilde), 2)
		dist = pow((np.sum(dist, axis=1)), 0.5)
		# dist = np.linalg.norm(locs1_pred - locs1_tilde, axis=1)
		# print("dist is: {}".format(dist))

		# inliers_curr = len(dist[np.where(dist<inlier_tol)])
		inliers_curr = dist[dist<inlier_tol].size
		inliersL.append(inliers)

		if inliers_curr > inliers:
			inliers = inliers_curr
			print(inliers)
			bestH2to1 = H2to1

	return bestH2to1, inliers
	


def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.


	#Create mask of same size as template
	mask = 255*np.ones(img.shape, np.uint8)
	
	#Warp mask by appropriate homography
	template = template.transpose(1,0,2)
	hei, wid = template.shape[:2]
	mask = cv2.warpPerspective(mask, H2to1, (wid,hei))

	# cv2.imwrite("../outputs/mask.jpg", mask) 

	#Warp template by appropriate homography
	img = cv2.warpPerspective(img, H2to1, (wid,hei))	

	#Use mask to combine the warped template and the image
	template[mask>0] = 0
	composite_img = template+img

	# Transpose images and calculate matches using matchPics

	return composite_img

