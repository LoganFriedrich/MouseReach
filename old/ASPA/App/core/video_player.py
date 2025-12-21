import os
import threading
import time
import cv2

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtWidgets import QApplication, QMainWindow, QWidget

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.widgets import Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from sklearn.cluster import DBSCAN

class VideoPlayer:
    def __init__(self, tn_count) -> None:
        self.thumbnails = []
        self.highlight = None
        self.seek_changed = False
        self.seek_value = 0
        self.play = False
        self.playback_speed = 16 # 16ms
        self.tn_count = tn_count
        self.play_swipes_only = False
        self.swipe_frames = None
        self.current_frame_idx = 0
        self.num_frames = 0
        self.has_video = False
        self.df = None
        self.mark_frames = True
        self.ref_x = 0
        self.ref_y = 0
        self.vdo_play_timer = QtCore.QTimer()
        self.vdo_play_timer.timeout.connect(self.on_vdo_play_timer)

        self.init_video_player(tn_count)

    #region Video Player

    def set_df(self, df):
        self.df = df
        self.ref_x = df['Reference_x'].mean()
        self.ref_y = df['Reference_y'].mean()

    def update_main_view(self, tn_idx=0):
        # Check if frames are available
        if self.num_frames == 0:
            return

        # Display the current frame
        self.ax_main_view.imshow(self.frames[tn_idx])
        self.ax_main_view.set_title(str(self.current_frame_idx)) 

    def init_video_player(self, tn_count=6):

        gs_rows = 3  # Gridspec rows
        gs_cols = tn_count * 2  # Gridspec columns

        self.tn_size = 100  # Change the size of the thumbnails as desired
        self.tn_count = tn_count  # 2 columns are reserved for navigation buttons
        # create a figure
        self.fig_vp = plt.figure()
        self.fig_vp.patch.set_facecolor('lightgrey')
        self.can_vp = FigureCanvas(self.fig_vp)

        self.fig_vp.subplots_adjust(wspace=0.0, hspace=0.05,
                                    top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # create grid for different subplots
        self.gs = gridspec.GridSpec(ncols=gs_cols, nrows=gs_rows, height_ratios=[0.8, 0.05, 0.15],
                                    hspace=0.1, wspace=0.05)
        
        self.ax_main_view = self.fig_vp.add_subplot(self.gs[:-2, :])
        self.ax_main_view.set_xticks([])
        self.ax_main_view.set_yticks([])
        # Add a blank image
        self.im_main_view = self.ax_main_view.imshow(np.ones((540, 480, 3)))
        # self.im_main_view = self.ax_main_view.imshow(np.ones((540, 480)))

        # Create the navigation buttons
        self.ax_prev = self.fig_vp.add_subplot(self.gs[-2, -3])
        self.ax_next = self.fig_vp.add_subplot(self.gs[-2, -2])
        self.ax_play = self.fig_vp.add_subplot(self.gs[-2, -1])
        
        # Create video progress bar
        self.pb_ax   = self.fig_vp.add_subplot(self.gs[-2, :-3])
        self.pb_ax.set_axis_off()
        self.pb_ax._pb_ax = 'pb_ax'
        self.pb_status = ''
        self.btn_prev = Button(self.ax_prev, '<')
        self.btn_next = Button(self.ax_next, '>')
        self.btn_play = Button(self.ax_play, 'Play')

        # Initialize the thumbnails
        self.init_thumbnails()

        self.vdo_thread = threading.Thread(target=self.play_vdo)

    def read_video_thread(self, video_path):
        thread = threading.Thread(target=self.read_video, args=(video_path,))
        thread.start()

    # Function to read video and extract frames
    def read_video(self, video_path=''):

        if not os.path.isfile(video_path):
            print('Invalid video path')
            return
        else:
            self.vdo_path = video_path
        
        self.cap = cv2.VideoCapture(video_path)

        self.swipe_seq_frame_count = 0
        self.swipe_seq_count = 0
        self.is_swipe = False

        self.motion_seq_frame_count = 0
        self.motion_seq_count = 0
        self.is_motion = False

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        # Background subtractor 
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        success, frame = self.cap.read()
        if not success:
            print('Failed to read video')
            self.has_video = False
            # self.cap.release()
            return
        # Get the video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.vdo_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vdo_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print(f'FPS: {fps}, Width: {self.vdo_width}, Height: {self.vdo_height}')

        self.frames = []
        while len(self.frames) < self.tn_count:
            ret, frame = self.cap.read()
            if not ret:
                self.has_video = False
                break
            # Convert to grayscale and append to frames list
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # self.cap.release()

        self.has_video = True

        self.tot_secs = self.num_frames / fps

        self.current_frame_idx = 0

        self.update_vdo_display()

        self.can_vp.draw()

    def on_vdo_play_timer(self):
        if self.seek_changed:
            self.seek_changed = False
            self.current_frame_idx = self.seek_value

        if self.play_swipes_only and self.swipe_idx < self.swipe_r:
            self.current_frame_idx = int(self.swipe_frames.iloc[self.swipe_idx]['bodyparts_coords'])
            self.swipe_idx += 1
        elif self.play_swipes_only and self.swipe_idx >= self.swipe_r:
            self.vdo_play_timer.stop()
            self.play = False
            self.btn_play.label.set_text('Play')
        else:
            self.current_frame_idx += 1
            if self.current_frame_idx >= self.num_frames:
                self.current_frame_idx = 0
                self.vdo_play_timer.stop()
                self.play = False
                self.btn_play.label.set_text('Play')

        # self.update_main_view()

        self.update_vdo_display()
        self.can_vp.draw()

    def play_on_click(self, event):
        self.play = not self.play
        if self.play:
            # Change button text to 'Pause'
            self.btn_play.label.set_text('Pause')
            # self.vdo_thread.start()
            self.swipe_idx, self.swipe_r = self.init_swipe_frames_playback()
            self.vdo_play_timer.start(self.playback_speed)
        else:
            # Change button text to 'Play'
            self.btn_play.label.set_text('Play')
            # self.vdo_thread.join()
            # self.vdo_thread = threading.Thread(target=self.play_vdo)
            self.vdo_play_timer.stop()

    def init_swipe_frames_playback(self):
        if self.swipe_frames is not None:
            swipe_idx = 0
            swipe_r, swipe_c = self.swipe_frames.shape
        else:
            swipe_idx = 0
            swipe_r = 0

        return swipe_idx, swipe_r

    def play_vdo(self):
        swipe_idx, swipe_r = self.init_swipe_frames_playback()
            
        while self.play:
            if self.seek_changed:
                self.seek_changed = False
                self.current_frame_idx = self.seek_value

            if self.play_swipes_only and swipe_idx < swipe_r:
                self.current_frame_idx = int(self.swipe_frames.iloc[swipe_idx]['bodyparts_coords'])
                swipe_idx += 1
            elif self.play_swipes_only and swipe_idx >= swipe_r:
                break
            else:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.num_frames:
                    self.current_frame_idx = 0
                    break

            # self.update_main_view()

            self.update_vdo_display()
            time.sleep(self.playback_speed)
            self.can_vp.draw()

    def set_playback_speed(self, value):
        # increase video playback speed based on slider value
        self.playback_speed = 0.1 / value

    def clear_vdo(self):
        self.num_frames = 0
        self.current_frame_idx = 0
        self.ax_main_view.clear()
        self.ax_main_view.set_axis_off()
        self.im_main_view = self.ax_main_view.imshow(np.ones((540, 480, 3)))
        # self.pb_ax.clear()
        # self.pb_ax.set_axis_off()
        self.update_progressbar()
        # Clear thumbnails
        for ax in self.thumbnails:
            ax.clear()
            ax.set_axis_off()
            ax.imshow(np.ones((100, 100, 3)))

        self.can_vp.draw()

    #endregion Video Player

    #region Mark frames

    def mark_frame(self):

        self.ax_main_view.clear()
        if not self.mark_frames:
            return  

        frame_info = self.df.iloc[self.current_frame_idx]


        # Colors for different levels of likelihood (<0.5, 0.5-0.9, >0.9)
        colors = ['red', 'yellow', 'green']

        # Mark right hand 
        color_idx = frame_info['RightHand_likelihood'] * 2
        color_idx = int(color_idx) if color_idx < 1.8 else 2
        cset = self.ax_main_view.scatter(frame_info['RightHand_x'], 
                                         frame_info['RightHand_y'], marker='x',
                                         c=colors[color_idx], s=100)
        
        # Mark the pellet 
        color_idx = frame_info['Pellet_likelihood'] * 2
        color_idx = int(color_idx) if color_idx < 1.8 else 2
        self.ax_main_view.scatter(frame_info['Pellet_x'], frame_info['Pellet_y'], marker='*', 
                                  c=colors[color_idx], s=100)

        # Mark the pillar
        color_idx = frame_info['Pillar_likelihood'] * 2
        color_idx = int(color_idx) if color_idx < 1.8 else 2
        self.ax_main_view.scatter(frame_info['Pillar_x'], frame_info['Pillar_y'], marker='o', 
                                  c=colors[color_idx], s=100)
        
        # Mark the nose
        color_idx = frame_info['Nose_likelihood'] * 2
        color_idx = int(color_idx) if color_idx < 1.8 else 2
        self.ax_main_view.scatter(frame_info['Nose_x'], frame_info['Nose_y'], marker='+', 
                                  c=colors[color_idx], s=100)
        
        # self.draw_swipe_contours(self.ax_main_view, 0)
        self.draw_pellet_contours(self.ax_main_view, 0)

    #endregion Mark frames

    #region Thumbnail

    def init_thumbnails(self):
        for i in range(0, self.tn_count * 2, 2):
            ax = self.fig_vp.add_subplot(self.gs[-1, i:i+2])
            ax.set_axis_off()
            self.thumbnails.append(ax)
            # set an empty image
            ax.imshow(np.ones((540, 480, 3)))

    def read_frames(self):
        # self.cap = cv2.VideoCapture(self.vdo_path)
        success, frame = self.cap.read()
        
        if not success:
            print('Failed to read video')
            self.has_video = False
            # self.cap.release()
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.frames = []
        while len(self.frames) < self.tn_count:
            success, frame = self.cap.read()
            if not success:
                # self.cap.release()
                self.has_video = False
                break
            # Convert to grayscale and append to frames list
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        # self.cap.release()

    def update_vdo_display(self):
        self.read_frames()
        self.mark_frame()
        self.update_progressbar()
        # Check if frames are available
        if not self.has_video:
            self.ax_main_view.imshow(np.ones((540, 480, 3)))
            return

        # Clear main view
        # self.ax_main_view.clear()
        # self.ax_main_view.set_xticks([])
        # self.ax_main_view.set_yticks([])
        # self.ax_main_view.set_frame_on(False)
        self.ax_main_view.set_axis_off()

        # Display the current frame
        self.ax_main_view.imshow(self.frames[0])
        self.ax_main_view.set_title(str(self.current_frame_idx))  
        # self.im_main_view.set_data(self.frames[self.current_frame_idx])

        if self.current_frame_idx+self.tn_count <= self.num_frames:

            # Add the new thumbnails
            idx = 0
            for i in range(self.current_frame_idx, self.current_frame_idx+self.tn_count):
                ax = self.thumbnails[idx]
                ax.clear()
                ax._id  = idx
                ax._fid = i
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                ax.set_axis_off()
                ax.imshow(self.frames[idx])

                idx += 1
                # Display the frame number on right top corner
                ax.text(0.95, 0.90, str(i), transform=ax.transAxes, color='blue',
                        fontsize=10, ha='right', weight='bold')
                
                # Mark a small circle on the top left corner if swipe is detected
                if hasattr(self, 'swipes') and i in self.swipes:
                    ax.plot(50, 50, 'D', color='red', markersize=6)
                    
                    # ax.text(0.05, 0.90, u'\N{check mark}', transform=ax.transAxes, color='m',
                    #     fontsize=20, ha='left', weight='bold')
                elif hasattr(self, 'valleys') and i in self.valleys:
                    ax.plot(50, 50, 'D', color='green', markersize=6)
            
        self.highlight_thumbnail()

    def draw_swipe_contours_hsv(self, ax, frame_idx):

        frame = self.frames[frame_idx]

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
        bottom_contours = [c for c in contours if ((cv2.boundingRect(c)[1] > self.ref_y-75) & (cv2.boundingRect(c)[1] < self.ref_y+75))
                                                & ((cv2.boundingRect(c)[0] > self.ref_x+60) & (cv2.boundingRect(c)[0] < self.ref_x+150))
                                                & ((cv2.contourArea(c) > 50) & (cv2.contourArea(c) < 1000))]

        # Draw bottom contours
        for c in bottom_contours:
            ax.plot(c[:,0,0], c[:,0,1], color='green', linewidth=2)

    def draw_pellet_contours(self, ax, frame_idx):
        frame = self.frames[frame_idx]
            
        # Get foreground mask
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours 
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        xmin = int(self.ref_x+60)
        xmax = int(self.ref_x+120)
        ymin = int(self.ref_y-50)
        ymax = int(self.ref_y+60)

        # Draw region of interest for the swipe contours
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)

        # self.draw_pellet_box(xmin-20, xmax-30, ymin, ymax, frame)
        # self.detect_bottom_corners(frame, xmin-30, xmax, ymin+30, ymax)
        # self.detect_pellet_box(frame, xmin, xmax, ymin, ymax)
        # Draw number of bottom contours on top right of the frame
        # cv2.putText(frame, f'{len(bottom_contours)}', (width-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        # Find pellet movmt. contours 
        # contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pellet_contours = [c for c in contours if ((cv2.boundingRect(c)[1] > self.ref_y-50) & (cv2.boundingRect(c)[1] < self.ref_y+25))
                                                & ((cv2.boundingRect(c)[0] > self.ref_x+60) & (cv2.boundingRect(c)[0] < self.ref_x+110))
                                                & ((cv2.contourArea(c) < 25) & (cv2.contourArea(c) > 5))]

        # Draw pellet contours
        for c in pellet_contours:
            ax.plot(c[:,0,0], c[:,0,1], color='red', linewidth=2)

        # Find tray movmt. contours
        bottom_contours = [c for c in contours if (cv2.boundingRect(c)[1] > ymin)]
        
        # If there are more than 300 bottom contours, then there is movement
        if len(bottom_contours) > 200:
            # Calculate total area of bottom contours
            total_area = 0
            for c in bottom_contours:
                total_area += cv2.contourArea(c)
            # If total area is greater than 10000, then there is movement
            if total_area > 10000:
                self.motion_seq_frame_count += 1
                if not self.is_motion:
                    self.motion_seq_count += 1
                self.is_motion = True
                cv2.putText(frame, f'Movement:{len(bottom_contours)}, Area:{total_area}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # time.sleep(0.2)
        else:
            self.is_motion = False
            if self.motion_seq_frame_count > 0:
                print(f'Motion sequence {self.motion_seq_count}({self.current_frame_idx}) frame count: {self.motion_seq_frame_count}')
            self.motion_seq_frame_count = 0

        swipe_contours = [c for c in bottom_contours if ((cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] < ymax))
                                                & ((cv2.boundingRect(c)[0] > xmin) & (cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2] < xmax))
                                                & ((cv2.contourArea(c) > 100) & (cv2.contourArea(c) < 1000))]
        
        if len(swipe_contours) == 1 and not self.is_motion:
            self.swipe_seq_frame_count += 1
            if not self.is_swipe:
                self.swipe_seq_count += 1
            self.is_swipe = True
            swipe_area = 0
            swipe_loc = ''
            for c in swipe_contours:
                swipe_area += cv2.contourArea(c)
                swipe_loc += f'({cv2.boundingRect(c)[0]}, {cv2.boundingRect(c)[1]})'
            # Draw swipe contours
            cv2.drawContours(frame, swipe_contours, -1, (0,255,0), 1)
            # cv2.putText(frame, f'Swipe:{swipe_loc}, Area:{swipe_area}', (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            time.sleep(0.05)
        else:
            self.is_swipe = False
            if self.swipe_seq_frame_count > 0:
                print(f'Swipe sequence {self.swipe_seq_count}({self.current_frame_idx}) frame count: {self.swipe_seq_frame_count}')
            self.swipe_seq_frame_count = 0

    def draw_pellet_box(self, xmin, xmax, ymin, ymax, frame):
        # Extract the ROI from the frame
        roi = frame[ymin:ymax, xmin:xmax]
        
        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Find contours in the edge-detected ROI
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect all contour points
        all_points = np.vstack([cont.reshape(-1, 2) for cont in contours])
        
        # Use DBSCAN to cluster nearby points
        clustering = DBSCAN(eps=20, min_samples=5).fit(all_points)
        
        # Find the largest cluster
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts[unique_labels != -1])]
        
        # Get points of the largest cluster
        cluster_points = all_points[labels == largest_cluster_label]
        
        if len(cluster_points) > 0:
            # Fit a rectangle to the cluster points
            rect = cv2.minAreaRect(cluster_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Find the bottom right corner (point with largest x + y value)
            bottom_right = tuple(box[np.argmax(box.sum(axis=1))])
            
            # Adjust coordinates relative to the full frame
            bottom_right = (bottom_right[0] + xmin, bottom_right[1] + ymin)
            
            # Draw the rectangle and the bottom right corner on the original frame
            cv2.drawContours(frame, [box + (xmin, ymin)], 0, (0, 255, 0), 2)
            cv2.circle(frame, bottom_right, 5, (0, 0, 255), -1)
            
            return bottom_right
        
        return None

    def detect_bottom_corners(self, frame, xmin, xmax, ymin, ymax):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)

        # Filter corners based on ROI
        filtered_corners = [corner for corner in corners if xmin <= corner[0][0] <= xmax and ymin <= corner[0][1] <= ymax]
        print(f"Number of corners detected within the ROI: {len(filtered_corners)}")
        # Sort corners by y-coordinate (descending) to focus on bottom corners
        filtered_corners.sort(key=lambda c: c[0][1], reverse=True)
        # # Draw circles around the filtered corners
        # frame_with_corners = frame.copy()
        # for corner in filtered_corners:
        #     x, y = corner.ravel()
        #     cv2.circle(frame_with_corners, (x, y), 5, (0, 0, 255), -1)

        # # Display the image using matplotlib
        # plt.figure(figsize=(12, 8))
        # plt.imshow(cv2.cvtColor(frame_with_corners, cv2.COLOR_BGR2RGB))

        # Parameters for filtering
        y_tolerance = 1  # Maximum y-difference for corners to be considered on the same line
        min_x_distance = 20  # Minimum x-distance between corners

        bottom_corners = []
        for i, corner1 in enumerate(filtered_corners):
            x1, y1 = corner1[0]
            for corner2 in filtered_corners[i+1:]:
                x2, y2 = corner2[0]
                
                # Check if corners have similar y-coordinates
                if abs(y1 - y2) <= y_tolerance:
                    # Check if corners are far enough apart in x-direction
                    if abs(x1 - x2) >= min_x_distance:
                        bottom_corners.extend([corner1, corner2])
                        break
            if len(bottom_corners) == 2:
                break

        # Draw circles around the detected corners
        # result_frame = frame.copy()
        for corner in bottom_corners:
            x, y = corner.ravel()
            # cv2.circle(result_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return result_frame, bottom_corners

    def detect_pellet_box(self, frame, xmin, xmax, ymin, ymax):
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size and shape
        for contour in contours:
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            if (xmin < x < xmax and 
                xmin < x + w < xmax and
                ymin < y < ymax and 
                ymin < y + h < ymax and
                10 < w < 100 and 
                10 < h < 100):
                # Draw bounding box around the detected box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw the contour
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)

    def detect_white_blobs(self, img):
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

    # Function to handle the click event on the navigation buttons
    def nav_on_click(self, event):
        if event.inaxes == self.ax_prev:
            self.seek_value = self.current_frame_idx - 1
        else:
            self.seek_value = self.current_frame_idx + 1

        if self.current_frame_idx < 0:
            self.seek_value = 0
        elif self.current_frame_idx >= self.num_frames:
            self.seek_value = self.num_frames - 1

        self.seek_changed = True

        if not self.play:
            self.current_frame_idx = self.seek_value
            self.update_vdo_display()
            self.can_vp.draw()

    # Function to update the main view
    def thumbnail_on_click(self, event):

        if hasattr(event.inaxes, '_id'):
            self.seek_value = int(event.inaxes._fid)
            self.mark_frame()
            self.highlight_thumbnail()
            self.update_main_view(event.inaxes._id)
            self.update_progressbar()
        elif hasattr(event.inaxes, '_pb_ax'):
            x_click = event.xdata  # Get the x-coordinate of the mouse click
            x_min, x_max = self.pb_ax.get_xlim()  # Get the limits of the x-axis
            percentage = (x_click - x_min) / (x_max - x_min) * 100  # Calculate the percentage
            # print(f"Clicked at {x_click:.2f} (Percentage: {percentage:.2f}%)")

            self.seek_value = int(percentage * self.num_frames / 100)
            self.seek_changed = True
            if not self.play:
                self.current_frame_idx = self.seek_value
                self.update_vdo_display()

        else:
            if not self.play:
                self.update_vdo_display()

        self.can_vp.draw()

    # Function to highlight the selected thumbnail
    def highlight_thumbnail(self):
        if self.num_frames == 0:
            return

        # Set red frame on the selected thumbnail
        ax = next((ax for ax in self.thumbnails if ax._id == self.current_frame_idx), None)
        if ax is None:
            return
        bbox = ax.get_tightbbox(self.fig_vp.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(self.fig_vp.transFigure.inverted()).bounds
        # slightly increase the very tight bounds:
        xpad = 0.0 * width
        ypad = 0.0 * height

        if self.highlight is not None:
            self.highlight.remove()
                
        self.highlight = self.fig_vp.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor='red', linewidth=1, fill=False))

    #endregion Thumbnail

    #region Progressbar

    def display_trackbar(self, num_frames=0):
        self.has_video = False
        self.num_frames = num_frames
        self.tot_secs = self.num_frames / 60

        self.current_frame_idx = 0

        self.update_progressbar()
        self.clear_vdo()

    # Function to update the progress bar
    def update_progressbar(self, pb_status=''):

        if self.num_frames == 0:
            return
        
        if '' == pb_status:
            self.pb_status = f'RH[{self.df.iloc[self.current_frame_idx]["RightHand_likelihood"]:.2f}]: ' + \
                             f'({self.df.iloc[self.current_frame_idx]["RightHand_x"]:.2f}, {self.df.iloc[self.current_frame_idx]["RightHand_y"]:.2f})  ' + \
                             f'Pellet[{self.df.iloc[self.current_frame_idx]["Pellet_likelihood"]:.2f}]: ' + \
                             f'({self.df.iloc[self.current_frame_idx]["Pellet_x"]:.2f}, {self.df.iloc[self.current_frame_idx]["Pellet_y"]:.2f})  ' + \
                             f'Pillar[{self.df.iloc[self.current_frame_idx]["Pillar_likelihood"]:.2f}]:' + \
                             f'({self.df.iloc[self.current_frame_idx]["Pillar_x"]:.2f}, {self.df.iloc[self.current_frame_idx]["Pillar_y"]:.2f})'

        # Clear the progress bar
        self.pb_ax.clear()

        pb_perc = self.current_frame_idx / self.num_frames

        curr_time_secs = self.tot_secs * pb_perc 

        # Convert seconds to hours, minutes and seconds
        curr_m, curr_s = divmod(int(curr_time_secs), 60)
        tot_m, tot_s = divmod(int(self.tot_secs), 60)

        # percentage text
        self.pb_ax.text(0.5, 0.1, f'{self.current_frame_idx}/{self.num_frames}    {pb_perc*100:.2f} %', fontdict={'family':'Arial', 'color':'black', 
                                                    'weight': 'bold','size': '10', 'ha':'center', 
                                                    'va':'bottom'}, transform=self.pb_ax.transAxes)
        self.pb_ax.text(0.5, 0.5, f'{curr_m}:{curr_s}/{tot_m}:{tot_s} ', fontdict={'family':'Arial', 'color':'black', 
                                                    'weight': 'bold','size': '10', 'ha':'center', 
                                                    'va':'bottom'}, transform=self.pb_ax.transAxes)
        # Draw a rectangle in figure coordinates ((0, 0) is bottom left and (1, 1) is 
        # # upper right).
        bg   = Rectangle((0, 0), width=1, height=1, transform=self.pb_ax.transAxes, facecolor='white', linewidth=0)
        pbar = Rectangle((0, 0), width=pb_perc, 
                          height=1, transform=self.pb_ax.transAxes, facecolor='green', linewidth=0)
        self.pb_ax.add_patch(bg)
        self.pb_ax.add_patch(pbar)
        self.pb_ax.set_title(self.pb_status, loc='left', fontdict={'family':'Arial', 'color':'black', 
                                                    'weight': 'bold','size': '10'})
        self.pb_ax.set_axis_off()

    #endregion Progressbar
