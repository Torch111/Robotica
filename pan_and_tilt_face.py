import cv2
import numpy as np
import os
import sys
from operator import itemgetter

from picamera.array import PiRGBArray
from picamera import PiCamera
import time

import serial


# initialize servo motor
panServoPosition = int(90)
tiltServoPosition = int(90)

usbport = '/dev/ttyUSB0'
ser = serial.Serial(usbport, 9600, timeout=1)		# Set up serial baud rate
ser.write(chr(255))
ser.write(chr(1))		   # servo motor #1 left-right	
ser.write(chr(panServoPosition))
ser.write(chr(255))
ser.write(chr(2))			# servo motor #2 up-down
ser.write(chr(tiltServoPosition))




class Face(object):

	def __init__(self, camera):
		self.res_x = 640
		self.res_y = 480
		self.intXFrameCenter = int(float(self.res_x / 2.0))
		self.intYFrameCenter = int(float(self.res_y / 2.0))

		self.camera = camera
		self.rawCapture = PiRGBArray(camera, size=(640, 480))
		self.camera.resolution = (self.res_x, self.res_y)
		self.framerate = 32
	
	def getImg(self, frame):
		img = frame.array
		img = cv2.flip(img,1)
		return img
	
	def getFaceCascade(self):
		# call Haar Cascade function 
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		return face_cascade
	
	def getEyeCascade(self):
		eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
		return eye_cascade
		
	def getFaces(self, gray, face_cascade):
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)		# face detection
		return faces
		
	def getGray(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return gray

	def dispFrame(self, img):
		cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
		cv2.imshow("Original", img)

	def calcError(self, faces, img, eye_cascade, gray):
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
				
			
			# need to set variables as global to override initial values
			global panServoPosition, tiltServoPosition
		
			
			# move servo motor to center of rectangle
			x_mid = x + (w/2)
			y_mid = y + (h/2)
					
			print "x=" + str(x) + " y=" + str(y) + " w=" + str(w) + " h=" + str(h)
			print "x_mid =" + str(x_mid) + " y_mid =" + str(y_mid)

	def step(self):
		for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
			image = self.getImg(frame)
			self.calcError(self.getFaces(self.getGray(image), self.getFaceCascade()), image, self.getEyeCascade(), self.getGray(image))
			self.dispFrame(image)
			self.rawCapture.truncate(0)
			key = cv2.waitKey(512) & 0xFF
			if key == ord("q"):
				break

face = Face(PiCamera())

while True:
	time.sleep(0.1)
	face.step()









# ###########################################################################
# def main():
# 	# initialize the camera and grab a reference to the raw camera capture
# 	res_x = 640
# 	res_y = 480
# 	camera = PiCamera()
# 	camera.resolution = (res_x, res_y)
# 	camera.framerate = 32
# 	rawCapture = PiRGBArray(camera, size=(640, 480))

# 	# setup frame for tracking
# 	intXFrameCenter = int(float(res_x / 2.0))
# 	intYFrameCenter = int(float(res_y / 2.0))
		
# 	# allow the camera to warm up
# 	time.sleep(0.1)

# 	# capture frames from the camera
# 	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
# 		# grab the raw NumPy array representing the image, then initialize the time stamp
# 		# and occupied/unoccupied text
# 		img = frame.array
# 		img = cv2.flip(img,1)			# flip image (1=horizontal   2=vertical)
	
# 		# call Haar Cascade function 
# 		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 		eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# 		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			# convert to gray scale
# 		faces = face_cascade.detectMultiScale(gray, 1.3, 5)		# face detection
	
		
# 		# coordinates provided by Haar Cascade
# 		#	x,y -------------
# 		#   |				|
# 		#	|				|
# 		#	----------x+w,y+h
# 		#
# 		for (x,y,w,h) in faces:
# 			cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
# 			roi_gray = gray[y:y+h, x:x+w]
# 			roi_color = img[y:y+h, x:x+w]
# 			eyes = eye_cascade.detectMultiScale(roi_gray)
# 			for (ex,ey,ew,eh) in eyes:
# 				cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
				
			
# 			# need to set variables as global to override initial values
# 			global panServoPosition, tiltServoPosition
		
			
# 			# move servo motor to center of rectangle
# 			x_mid = x + (w/2)
# 			y_mid = y + (h/2)
			
# 			if x_mid < intXFrameCenter and panServoPosition >= 2:
# 				panServoPosition = panServoPosition - 3
# 			elif x_mid > intXFrameCenter and panServoPosition <= 178:
# 				panServoPosition = panServoPosition + 3
		
# 			if y_mid < intYFrameCenter and tiltServoPosition >= 2:
# 				tiltServoPosition = tiltServoPosition - 3
# 			elif x_mid > intXFrameCenter and panServoPosition <= 178:
# 				tiltServoPosition = tiltServoPosition + 3
				
# 			ser.write(chr(255))
# 			ser.write(chr(1))		   # servo motor #1 left-right	
# 			ser.write(chr(panServoPosition))
# 			ser.write(chr(255))
# 			ser.write(chr(2))			# servo motor #2 up-down
# 			ser.write(chr(tiltServoPosition))
					
# 			print "x=" + str(x) + " y=" + str(y) + " w=" + str(w) + " h=" + str(h)
# 			print "x_mid =" + str(x_mid) + " y_mid =" + str(y_mid)	
		
		
# 		# show original frame
# 		cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# 		cv2.imshow("Original", img)
	
# 		key = cv2.waitKey(
# 		512) & 0xFF
	
# 		# clear the stream in preparation for the next frame
# 		rawCapture.truncate(0)
			
# 	    # if the 'q' was pressed, break from the loop
# 		if key == ord("q"):
# 			break

# ###########################################################################

# # run code only if script is run directly (not imported)
# if __name__== "__main__":
# 	main()	
