import cv2
import numpy as np

# Initialize camera
# cap = cv2.VideoCapture('Shared Data/20220525_H22_P1.mp4')
cap = cv2.VideoCapture('20220721_H36_E2.mp4')

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

# 249, 217, 207

print(rgb_to_hsv(249, 217, 207))

while True:
    # Read frame
    ret, frame = cap.read()
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply histogram equalization to enhance contrast
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    # Define wider range of skin color
    lower = np.array([14, 20, 100])
    upper = np.array([255, 255, 255])

    # Threshold image
    mask = cv2.inRange(hsv, lower, upper)
    
    # Apply bilateral filter to reduce noise while preserving edges
    mask = cv2.bilateralFilter(mask, 5, 75, 75) 

    # Find contours and track hand
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bottom_contours = [c for c in contours if (cv2.boundingRect(c)[1] > frame.shape[0]*3/4) & (cv2.contourArea(c) > 100) & (cv2.contourArea(c) < 1000) & (cv2.boundingRect(c)[0] > 100)]
    # Draw bottom contours
    cv2.drawContours(frame, bottom_contours, -1, (0,255,0), 2)
    # contour = max(contours, key=cv2.contourArea)
    # x,y,w,h = cv2.boundingRect(contour)
    # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display results 
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    
    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()