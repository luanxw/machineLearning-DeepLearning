import cv2 

original = cv2.imread('../machineLearning/data/test.jpeg')
cv2.imshow('original', original)
gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow('cvtColor', gray_im)
gray_im_ = cv2.equalizeHist(gray_im)
cv2.imshow('gray_im_', gray_im_)
cv2.waitKey()