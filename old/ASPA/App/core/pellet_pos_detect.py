import cv2
import numpy as np
import csv
import pytesseract
from collections import deque
import os
import glob
import sys
import pandas as pd
from scipy import stats
 
def find_tesseract_executable():
    if sys.platform.startswith('win'):
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in common_paths:
            if os.path.isfile(path):
                return path
    else:
        from shutil import which
        return which('tesseract')
    return None
 
tesseract_path = find_tesseract_executable()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("Tesseract executable not found. OCR functionality will be disabled.")
 
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
input_videos = []
for ext in VIDEO_EXTENSIONS:
    input_videos.extend(glob.glob(f'*{ext}'))
 
if not input_videos:
    print("No video files found in the current directory.")
    exit()
 
def preprocess_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
def find_pink_square(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    return None
 
def find_circles(frame):
    gray = preprocess_frame(frame)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=3, maxRadius=30)
    return circles[0] if circles is not None else []
 
def detect_numbers(frame):
    if not tesseract_path:
        return []
    gray = preprocess_frame(frame)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    try:
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        numbers = [(int(text), (data['left'][i], data['top'][i], data['width'][i], data['height'][i])) 
                   for i, text in enumerate(data['text']) if text.isdigit()]
        return sorted(numbers, key=lambda x: x[1][0])
    except Exception as e:
        print(f"An error occurred during OCR: {str(e)}")
        return []
 
def classify_circles(circles, pink_square, number_regions):
    x, y, w, h = pink_square
    reference_circles, pillars, pellets = [], [], []
    for circle in circles:
        cx, cy, r = circle
        if x < cx < x+w and y < cy < y+h:
            reference_circles.append(circle)
        elif any(nx < cx < nx+nw and ny < cy < ny+nh for _, (nx, ny, nw, nh) in number_regions):
            continue
        elif r > 10:
            pillars.append(circle)
        else:
            pellets.append(circle)
    return reference_circles, pillars, pellets
 
def find_pellet_lines(pillars, pink_square):
    x, y, w, h = pink_square
    line1 = [(px, py) for px, py, _ in pillars if abs(px - (x + w/2)) < abs(px - (x + w/2 + w))]
    line2 = [(px, py) for px, py, _ in pillars if (px, py) not in line1]
    line1.sort(key=lambda p: p[0])
    line2.sort(key=lambda p: p[0])
    return (line1, line2) if abs(line1[0][0] - (x + w/2)) < abs(line2[0][0] - (x + w/2)) else (line2, line1)
 
def calculate_pillar_line(pillars):
    """Calculate the line of pillars using linear regression."""
    x = np.array([p[0] for p in pillars])
    y = np.array([p[1] for p in pillars])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value**2
 
def estimate_pillar_positions(pillars, slope, intercept):
    """Estimate ideal positions of pillars based on the calculated line."""
    x_coords = [p[0] for p in pillars]
    y_coords = [slope * x + intercept for x in x_coords]
    return list(zip(x_coords, y_coords))
 
def calculate_pillar_spacing(pillars):
    """Calculate the average spacing between pillars."""
    if len(pillars) < 2:
        return None
    spacings = [pillars[i+1][0] - pillars[i][0] for i in range(len(pillars)-1)]
    return np.mean(spacings)
 
def find_active_pillar_position(main_line, pink_square, avg_spacing):
    """Find the position of the active pillar (second from the left of the pink square)."""
    x, y, w, h = pink_square
    center_x = x + w/2
    
    # Find the two pillars to the right of the pink square
    right_pillars = [p for p in main_line if p[0] > center_x]
    if len(right_pillars) < 2:
        return None
    
    # The active pillar is the second one to the right
    active_x = right_pillars[1][0]
    active_y = right_pillars[1][1]
    
    return active_x, active_y
 
def maintain_pillar_history(current_pillars, history, max_history=30):
    """Maintain a history of pillar positions for better estimation."""
    history.append(current_pillars)
    if len(history) > max_history:
        history.popleft()
    return history
 
def estimate_missing_pillars(current_pillars, history, avg_spacing):
    """Estimate positions of missing pillars based on history and spacing."""
    if not history:
        return current_pillars
 
    all_x_coords = set()
    for past_pillars in history:
        all_x_coords.update(p[0] for p in past_pillars)
    
    current_x_coords = set(p[0] for p in current_pillars)
    missing_x_coords = all_x_coords - current_x_coords
    
    estimated_pillars = list(current_pillars)
    for x in missing_x_coords:
        # Find the closest existing pillar
        closest_pillar = min(current_pillars, key=lambda p: abs(p[0] - x))
        # Estimate y-coordinate based on the closest pillar and average spacing
        estimated_y = closest_pillar[1] + (x - closest_pillar[0]) * (avg_spacing / abs(x - closest_pillar[0]))
        estimated_pillars.append((x, estimated_y))
    
    return sorted(estimated_pillars, key=lambda p: p[0])
 
def calculate_confidence(frame, position, window_size=5):
    x, y = map(int, position)
    window = frame[max(0, y-window_size):min(frame.shape[0], y+window_size+1),
                   max(0, x-window_size):min(frame.shape[1], x+window_size+1)]
    return np.mean(window) / 255.0
 
def find_matching_csv(video_path):
    video_name = os.path.basename(video_path)[:15]
    csv_files = glob.glob(f"{video_name}*.csv")
    return csv_files[0] if csv_files else None
 
def process_video(video_path):
    dlc_csv_path = find_matching_csv(video_path)
    if not dlc_csv_path:
        print(f"Matching DeepLabCut CSV file not found for {video_path}. Skipping this video.")
        return
 
    df = pd.read_csv(dlc_csv_path, header=[0, 1, 2], index_col=0)
 
    new_columns = pd.MultiIndex.from_product([['CustomPelletDetection'], 
                                              ['x', 'y', 'confidence_x', 'confidence_y', 'pillar_number', 'active_pellet_number'],
                                              ['detected_pellet']])
 
    for col in new_columns:
        df[col] = np.nan
 
    cap = cv2.VideoCapture(video_path)
    pillar_history = deque(maxlen=30)
    active_pellet_number = 1  # Initialize the active pellet number
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
 
        if frame_index < len(df.index):
            frame_number = df.index[frame_index]
        else:
            print(f"Warning: Video frame {frame_index} exceeds DeepLabCut CSV data. Skipping remaining frames.")
            break
 
        pink_square = find_pink_square(frame)
        
        if pink_square is None:
            continue
 
        numbers = detect_numbers(frame)
        circles = find_circles(frame)
        _, pillars, pellets = classify_circles(circles, pink_square, numbers)
 
        if not pillars:
            continue
 
        main_line, _ = find_pellet_lines(pillars, pink_square)
        
        # Maintain history and estimate missing pillars
        pillar_history = maintain_pillar_history(main_line, pillar_history)
        avg_spacing = calculate_pillar_spacing(main_line)
        estimated_pillars = estimate_missing_pillars(main_line, pillar_history, avg_spacing)
        
        # Calculate the pillar line
        slope, intercept, r_squared = calculate_pillar_line(estimated_pillars)
        
        # Estimate ideal positions of pillars
        ideal_positions = estimate_pillar_positions(estimated_pillars, slope, intercept)
        
        # Find the position of the active pillar
        active_pillar_position = find_active_pillar_position(ideal_positions, pink_square, avg_spacing)
        
        if active_pillar_position:
            x, y = active_pillar_position
            confidence = calculate_confidence(preprocess_frame(frame), active_pillar_position)
            
            # Determine pillar number (you may need to adjust this based on your specific requirements)
            pillar_number = next((i+1 for i, p in enumerate(ideal_positions) if p[0] == x), None)
            
            # Write to DataFrame
            df.loc[frame_number, ('CustomPelletDetection', 'x', 'detected_pellet')] = x
            df.loc[frame_number, ('CustomPelletDetection', 'y', 'detected_pellet')] = y
            df.loc[frame_number, ('CustomPelletDetection', 'confidence_x', 'detected_pellet')] = confidence
            df.loc[frame_number, ('CustomPelletDetection', 'confidence_y', 'detected_pellet')] = confidence
            df.loc[frame_number, ('CustomPelletDetection', 'pillar_number', 'detected_pellet')] = pillar_number
            df.loc[frame_number, ('CustomPelletDetection', 'active_pellet_number', 'detected_pellet')] = active_pellet_number
 
            print(f"Frame {frame_number}: Active pillar at ({x}, {y}), number {pillar_number}, active pellet number {active_pellet_number}, confidence {confidence}")
 
            # Check if the pellet has been knocked off (you may need to adjust this logic)
            if not any(abs(p[0] - x) < avg_spacing/2 and abs(p[1] - y) < avg_spacing/2 for p in pellets):
                active_pellet_number += 1
        else:
            # If no active pillar is found, still record the current active pellet number
            df.loc[frame_number, ('CustomPelletDetection', 'active_pellet_number', 'detected_pellet')] = active_pellet_number
 
        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{total_frames} frames from {video_path}")
 
    cap.release()
 
    output_csv_path = os.path.splitext(dlc_csv_path)[0] + '_with_custom_detection.csv'
    df.to_csv(output_csv_path)
    print(f"Updated CSV with custom detection saved to {output_csv_path}")
 
if __name__ == "__main__":
    for video in input_videos:
        print(f"Processing video: {video}")
        process_video(video)
    print("All videos processed.")
 