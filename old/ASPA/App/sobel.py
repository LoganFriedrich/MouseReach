import cv2
import numpy as np

# cap    = cv2.VideoCapture('Shared Data/20220525_H22_P1.mp4')
cap    = cv2.VideoCapture('20220721_H36_E2.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Detect horizontal lines using first frame
ret, frame = cap.read()

while(1):
  
    # Take each frame
    _, frame = cap.read()
      
    # Convert to HSV for simpler calculations
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
    # Calculation of Sobelx
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=7)
      
    # Calculation of Sobely
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=7)
      
    # Calculation of Laplacian
    laplacian = cv2.Laplacian(frame,cv2.CV_64F)

    ret, lap_thresh = cv2.threshold(np.absolute(laplacian), 100, 255, cv2.THRESH_BINARY)
      
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('lap_thresh',lap_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
# out.release()
cv2.destroyAllWindows()