from operator import is_
import time
from turtle import width
import cv2
import numpy as np

def detect_white_blobs(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get just the white dots
    ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY) 

    # Find contours in thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bc = [c for c in contours if cv2.boundingRect(c)[1] > gray.shape[0]*3/4]

    # Draw circles around white dots
    for c in bc:
        if cv2.contourArea(c) > 2:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (255, 255, 0), 2)

        # Draw radius text
        text = str(radius)
        text_pos = (int(x-10), int(y+10))
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

# Set up video capture and output video writer
cap    = cv2.VideoCapture('20220721_H36_E2.mp4')
# cap    = cv2.VideoCapture('Shared Data/20220525_H22_P1.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out    = cv2.VideoWriter('20220721_H36_E2_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# Background subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False) 

pellet_no = 1
swipe_no = 0
frame_no = 0

swipe_seq_frame_count = 0
swipe_seq_count = 0

motion_seq_frame_count = 0
motion_seq_count = 0

is_swipe = False
is_motion = False

# Detect horizontal lines using first frame
ret, frame = cap.read()

# Transform source image to gray if it is not already
if len(frame.shape) != 2:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    gray = frame
    

# [bin]
# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
cv2.THRESH_BINARY, 15, -2)

# Show binary image
# show_wait_destroy("binary", bw)

# Create the images that will use to extract the horizontal and vertical lines
horizontal = np.copy(bw)
# Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = 100
# Create structure element for extracting small(<50 pixels) horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

# Show extracted horizontal lines
# show_wait_destroy("horizontal", horizontal)

while(1):
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    # Detect white blobs
    # detect_white_blobs(frame)
    # Draw horizontal lines on the frame
    for i in range(horizontal.shape[0]):
        if horizontal[i][0] == 255:
            cv2.line(frame, (0, i), (frame.shape[1], i), (0,120,255), 2)
            break

    # cv2.line(frame, (0, 0), (frame.shape[1], 0), (0,120,255), 2)

    
    # Get foreground mask
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # Find contours 
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    # bottom contours
    bottom_contours = [c for c in contours if ((cv2.boundingRect(c)[1] > frame.shape[0]*3/4) & (cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] < frame.shape[0]*3.75/4))]

    # Draw a ROI of bottom contours
    cv2.rectangle(frame, (0, int(frame.shape[0]*3/4)), (frame.shape[1], int(frame.shape[0]*3.75/4)), (0,255,0), 2)

    # If there are more than 300 bottom contours, then there is movement
    if len(bottom_contours) > 200:
        # Calculate total area of bottom contours
        total_area = 0
        for c in bottom_contours:
            total_area += cv2.contourArea(c)
        # If total area is greater than 10000, then there is movement
        if total_area > 10000:
            motion_seq_frame_count += 1
            if not is_motion:
                motion_seq_count += 1
            is_motion = True
            cv2.putText(frame, f'Movement:{len(bottom_contours)}, Area:{total_area}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # time.sleep(0.2)
    else:
        is_motion = False
        if motion_seq_frame_count > 0:
            print(f'Motion sequence {motion_seq_count}({frame_no}) frame count: {motion_seq_frame_count}')
        motion_seq_frame_count = 0
        # cv2.putText(frame, f'Movement{len(bottom_contours)}, Area:{total_area}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # time.sleep(0.2)
    
    swipe_contours = [c for c in bottom_contours if (cv2.contourArea(c) > 100) 
                      & (cv2.contourArea(c) < 700) 
                      & ((cv2.boundingRect(c)[0] > 140) & (cv2.boundingRect(c)[0] < 180))
                      & ((cv2.boundingRect(c)[1] > 400) & (cv2.boundingRect(c)[1] < 500))
                      ]
    
    # Draw ROI of swipe contours
    cv2.rectangle(frame, (140, 400), (180, 500), (0,0,255), 2)


    if len(swipe_contours) == 1 and not is_motion:
        swipe_seq_frame_count += 1
        if not is_swipe:
            swipe_seq_count += 1
        is_swipe = True
        swipe_area = 0
        swipe_loc = ''
        for c in swipe_contours:
            swipe_area += cv2.contourArea(c)
            swipe_loc += f'({cv2.boundingRect(c)[0]}, {cv2.boundingRect(c)[1]})'
        # Draw swipe contours
        cv2.drawContours(frame, swipe_contours, -1, (0,255,0), 2)
        cv2.putText(frame, f'Swipe:{len(swipe_contours)}{swipe_loc}, Area:{swipe_area}', (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        time.sleep(0.05)
    else:
        is_swipe = False
        if swipe_seq_frame_count > 0:
            print(f'Swipe sequence {swipe_seq_count}({frame_no}) frame count: {swipe_seq_frame_count}')
        swipe_seq_frame_count = 0

    # Draw number of bottom contours on top right of the frame
    cv2.putText(frame, f'{len(bottom_contours)}', (width-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Draw bottom contours 
    # for c in bottom_contours:
    #     cv2.drawContours(frame, [c], -1, (0,255,0), 2)
        
    # Write annotated frame to output video 
    # out.write(frame)
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
# out.release()
cv2.destroyAllWindows()