import cv2
#image = cv2.imread()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
#edged = cv2.Canny(gray, 30, 200)
#cv2.imshow('',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



gray = cv2.imread('/home/olya/Documents/thesis/all/alef.598-19.2.png', cv2.IMREAD_GRAYSCALE)

# resize the images and invert it (black background)
gray = cv2.resize(255-gray, (28, 28))/255


cv2.imshow('',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#    # save the processed images
#    cv2.imwrite("pro-img/image_"+str(no)+".png", gray)
"""
all images in the training set have an range from 0-1
and not from 0-255 so we divide our flatten images
(a one dimensional vector with our 784 pixels)
to use the same 0-1 based range
"""
#    flatten = gray.flatten() / 255.0