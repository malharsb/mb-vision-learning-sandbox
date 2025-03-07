import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from opts import get_opts
from matchPics import matchPics
from planarH import *
opts = get_opts()

#Write script for Q3.1

# Load videos and save as .npy for faster loading and debugging
ar_source = loadVid('../data/ar_source.mov')
print(ar_source.shape)
book = loadVid('../data/book.mov')
print("book shape is {}".format(book.shape))

# cv2.imshow('ar1', ar_source[0])
# cv2.imwrite("../outputs/ar1.jpg", ar_source[0]) 
# cv2.imshow('book', book[0])
# cv2.imwrite("../outputs/book.jpg", book[0]) 

# np.save('../outputs/ar_source.npy', ar_source)
# np.save('../outputs/book.npy', book)



# Load up npys and check shape
# ar_source = np.load('../outputs/ar_source.npy')
# book = np.load('../outputs/book.npy')
# print("ar_source shape: {} and book shape: {}".format(ar_source.shape, book.shape))



# Initialize and get cv_cover size to resize
outframes = []
frames = min(ar_source.shape[0], book.shape[0])
print("total frames: {}".format(frames))



# Read cv_cover and 
book_crop = book[0][70:430,130:410, :].transpose(1,0,2) # book_crop shape is 360x280x3
# cv2.imwrite("../outputs/book_crop.jpg", book_crop) 
print("book crop shape: {}".format(book_crop.shape))



# Iterate through each frame
for f in range(1,435):

    # Crop source and get current destination
    ar_crop = ar_source[f][45:315, 177:463, :]
    bookf = book[f].transpose(1,0,2)

    # Create vectors of matched points
    # cv2.imwrite("../outputs/bookf.jpg", bookf) 
    # cv2.imwrite("../outputs/book_crop.jpg", book_crop) 
    # cv2.imwrite("../outputs/ar_crop.jpg", ar_crop) 
    matches, locs1, locs2 = matchPics(bookf, book_crop, opts)

    # Calculate H 
    num_matches = matches.shape[0]
    m1 = np.zeros((num_matches,2))
    m2 = np.zeros((num_matches,2))
    j=0
    for i in matches:
        m1[j] = locs1[i[0]]
        m2[j] = locs2[i[1]]
        j+=1
    bestH2to1, _ = computeH_ransac(m1, m2, opts)

    # Resize ar_crop to fit book_crop
    rw,rh = book_crop.shape[:2]
    ar_crop = cv2.resize(ar_crop, (rw,rh))

    # Calculate composite between source and destination
    print("f {}".format(f)) # Bookf shape is 640x480x3
    composite = compositeH(bestH2to1, bookf, ar_crop)
    outframes.append(composite)
    # cv2.imwrite("../outputs/ar/ar"+str(f)+".jpg", composite) 

 

print("composite size is {}".format(composite.shape))
print("outframes len is {}".format(len(outframes)))
outframes = np.array(outframes)
# np.save('../outputs/outframes.npy', outframes)
# print("Saved")



# Script to load up outframes and create a video
# outframes = np.load('../outputs/outframes.npy')
print("shape of outframes is: {}".format(outframes.shape))
out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640, 480))
for i in range(outframes.shape[0]):
    out.write(outframes[i]) # ouframes[i] shape: 480x640x3
out.release()



# # Script to combine two videos
# outframes1 = np.load('../outputs/ar/outframes1.npy')
# outframes2 = np.load('../outputs/ar/outframes2.npy')
# print("outframes1 shape: ", outframes1.shape, "outrames2 shape: ", outframes2.shape)
# outframes = np.vstack([outframes1, outframes2])
# print("shape of outframes is: {}".format(outframes.shape))
# out = cv2.VideoWriter('../outputs/ar.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640, 480))
# for i in range(outframes.shape[0]):
#     out.write(outframes[i]) # ouframes[i] shape: 480x640x3
# out.release()

# Yo made this change