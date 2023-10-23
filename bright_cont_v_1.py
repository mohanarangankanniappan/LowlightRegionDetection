'''
File Name : Detect Brightness and Contrast of an Image Frame

Description : Identifies the regions of the image is dark and contrast is low.

Author : Mohanarangan Kanniappan

Author email : mohanaranganphd@gmail.com

Date   : 08/10/2013

Version : 0.0

'''

import glob
import sys
import os
from sliding_window import pyramid, sliding_window
import joblib
import argparse
import cv2
import numpy as np
from nms import non_max_suppression_fast

'''
Input Directory

'''

InputDir = "./input/"

'''

Output Directory

'''

OutputDir = "./output/"

'''

Configuration parameters


'''

Resize_Width =  1280
Resize_Height =  720

'''

Contrast Configuration Parameters

'''
Contrast_Thershold = 0.5
Contrast_Stepsize = 16
Contrast_Analytic_Window_Width = 32
Contrast_Analytic_Window_Height = 32
Contrast_Window_Width = 32
Contrast_Window_Height = 32

'''

Brightness Configuration Parameters

'''

Bright_Thershold = 50
Bright_Stepsize = 16
Bright_Analytic_Window_Width =16
Bright_Analytic_Window_Height =16
Bright_Window_Width =16
Bright_Window_Height=16


'''

	To determine the contrast of the region passed.	
	
'''


def iscontrast(img):

    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)
    contrast = 0
    # compute contrast
    if (max+min): 
       contrast = (max-min)/(max+min)
    else:
       contrat = 0 
    #print("Contrast : ", contrast)
    return (contrast > Contrast_Thershold)

'''

	To determine the brightness of the region passed.	
	
'''



def isbright(image, dim=10, thresh=Bright_Thershold):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh


'''

	Detects the dark region of image using sliding window	
	
'''




def detect_dark(image,filename):
		print("Processing for Brightness")
		
		imgblue = np.full((int(Resize_Height), int(Resize_Width), 3),(255,0,0), dtype = np.uint8)
		
		AnalyticWinSize = (Bright_Analytic_Window_Width,Bright_Analytic_Window_Height)
		
	
		#cv2.namedWindow('Sliding Window',cv2.WINDOW_NORMAL)

		print('width: ', int(image.shape[0]))
		print('height:', int(image.shape[1]))
		im = np.full((int(image.shape[0]), int(image.shape[1]), 3),255, dtype = np.uint8)

	
	
		# Image pyramid parameters
		scale = 2.0
		minSize = (500, 500)
		# Sliding window parameters 
		stepSize = Bright_Stepsize
		(winW, winH) = (Bright_Window_Width, Bright_Window_Height)

		bboxes = np.zeros(4,np.int64) # Variable to save the resulting bounding boxes
		# loop over the image pyramid
		for i, resized in enumerate(pyramid(image, scale=scale, minSize=minSize)):
			# loop over the sliding window for each layer of the pyramid
			for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue

				# Draw sliding Window
				clone = resized.copy()
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
				
				# Cropped the resized image using x,y,winW, winH
				cropped_img = resized[y:y + winH, x:x + winW]
				# Resize it so the HOG descriptor can be obtained
				cropped_img_resized = cv2.resize(cropped_img, AnalyticWinSize)
				# Compute the HOG descriptor
				y_pred = isbright(cropped_img_resized)
				# Display both the Sliding window and the 
						
				if y_pred == False:
					if i != 0:
						bboxes = np.vstack((bboxes, np.array([
							int(x*scale*i), int(y*scale*i),
							int((x + winW)*scale*i), int((y + winH)*scale*i)])))
					else:
						bboxes = np.vstack((bboxes, np.array([
							int(x),int(y),int(x + winW), int(y + winH)])))

					
                                        
					if x  < 1920 :
						im[y:y+winH, x:x+winW] = cropped_img
						cv2.putText(image, 'D', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (59, 212, 255), thickness=1)
				else:
					cv2.putText(image, 'B', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
					
	
		writename = os.path.basename(filename)
		writename = OutputDir+"Bright" +writename
		print("writing file name :",writename)
		cv2.imwrite(writename, image)
		cv2.imshow("Brightness", image)
		cv2.waitKey(0)

'''

	Detects the low contrast region of image using sliding window	
	
'''
def detect_contrast(image,filename):
	
	print("Processing for Contrast")	
	imgblue = np.full((int(720), int(1280), 3),(255,0,0), dtype = np.uint8)
	AnalyticWinSize = (Contrast_Analytic_Window_Width,Contrast_Analytic_Window_Height)
        # Image pyramid parameters
	scale = 2.0
	minSize = (500, 500)
	# Sliding window parameters 
	 
	stepSize = Contrast_Stepsize 
	(winW, winH) = (Contrast_Window_Width, Contrast_Window_Height)
	im = np.full((int(image.shape[0]), int(image.shape[1]), 3),255, dtype = np.uint8)

	bboxes = np.zeros(4,np.int64) # Variable to save the resulting bounding boxes
	# loop over the image pyramid
	for i, resized in enumerate(pyramid(image, scale=scale, minSize=minSize)):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# Draw sliding Window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			
			# Cropped the resized image using x,y,winW, winH
			cropped_img = resized[y:y + winH, x:x + winW]
			cropped_img = cv2.bitwise_not(cropped_img)
			# Resize it so the HOG descriptor can be obtained
			cropped_img_resized = cv2.resize(cropped_img, AnalyticWinSize)
			y_pred = iscontrast(cropped_img_resized)
					
			if y_pred == True:
				if i != 0:
					bboxes = np.vstack((bboxes, np.array([
						int(x*scale*i), int(y*scale*i),
						int((x + winW)*scale*i), int((y + winH)*scale*i)])))
				else:
					bboxes = np.vstack((bboxes, np.array([
						int(x),int(y),int(x + winW), int(y + winH)])))

				cv2.putText(image, 'C', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1)
				#cv2.waitKey(0)
			else:
				if x  < 1920 :
						im[y:y+winH, x:x+winW] = cropped_img
						cv2.putText(image, 'P', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (59, 212, 255), thickness=1)
		writename = os.path.basename(filename)
		writename = OutputDir+"Contrast" +writename
		cv2.imwrite(writename, image)

		print("writing file name :",writename)
		bboxes = np.delete(bboxes, (0), axis=0)
	cv2.imshow("Contrast",image)				
	cv2.waitKey(0)
	cv2.destroyAllWindows()




def main():
        
	for filename in glob.glob(InputDir+"*.jpg"):
		print("Processing File :", filename)
		input_image = cv2.imread(filename)
		if input_image is None: 
				result = "Image is empty!!"
				sys.os.exit(0)
		else:
			resized_image = cv2.resize(input_image, (Resize_Width,Resize_Height))
			cv2.imshow("InputImage",resized_image)
			cv2.waitKey(0)
			#detect_dark(resized_image,filename)
			detect_contrast(resized_image,filename)

if __name__ == "__main__":
    main()
