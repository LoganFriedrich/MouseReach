import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture('20220721_H36_E2.mp4')
# Capture background 
ret, bg = cap.read()
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Flip frame horizontally
    frame = cv2.flip(frame, 1) 
    
    # Convert captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between background and current frame
    diff = cv2.absdiff(gray, bg)
    
    # Threshold difference image to get foreground mask
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Apply dilation to fill gaps 
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours in thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    # Draw contour of largest area
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
   
    # Segment hand region    
    mask = cv2.bitwise_and(frame, frame, mask=thresh)

    # Display images
    cv2.imshow('Original', frame)
    cv2.imshow('Masked', mask)
    
    # Quit on q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release resources        
cap.release()
cv2.destroyAllWindows()