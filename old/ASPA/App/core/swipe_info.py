import sys
import threading
import time
import traceback
import numpy as np
import pandas as pd

from qtpy.QtWidgets import QTableWidgetItem

class SwipeInfo:
    def __init__(self, tbl=None, ui=True, reachFilter=None, df=None) -> None:
        self.tblSwipeInfo = tbl
        self.df = df
        self.df_full = None
        self.has_ui = ui
        self.y_min = 0
        self.y_max = 0
        self.attention_frames = 0
        self.tot_frames = 0
        self.pellet_x_median = 0
        self.pellet_y_median = 0
        self.pellet_pos_th = 4
        self.pellet_loc_df = None

        if df is not None:
            self.display_frame_info()

        if reachFilter is not None:
            self.reachOutcomeFilter = reachFilter
            self.reachOutcomeFilter.textChanged.connect(self.filter_data)
    
    #region Read DLC output

    def read_csv_thread(self, file_path):
        thread = threading.Thread(target=self.read_csv, args=(file_path,))
        thread.start()

    def read_csv(self, file_path):
        # Read excel file with name same as video file
        csv_file = file_path

        comb = '_'

        # Create headers from first 3 rows, 1st is unused, 2nd is part name, 3rd is x/y/likelihood
        cols = [x+comb+y for x, y in list(zip(pd.read_csv(csv_file, nrows=2).values.tolist()[0], 
                                            pd.read_csv(csv_file, nrows=2).values.tolist()[1]))]

        # get actual data
        self.df = pd.read_csv(csv_file, skiprows=3, header=None)

        # add headers
        self.df.columns = cols

    #endregion Read DLC output

    #region Swipe Info Viewer

    def display_range(self, start):
        # Clear table
        self.tblSwipeInfo.clear()
        # Display dataframe in pyqt table widget
        # Get the number of rows and columns
        num_rows, num_cols = self.df.shape

        # Set the table widget dimensions
        self.tblSwipeInfo.setRowCount(200)
        self.tblSwipeInfo.setColumnCount(num_cols)

        # Set the table headers
        self.tblSwipeInfo.setHorizontalHeaderLabels(self.df.columns)

        # Populate the table widget with data
        for row in range(0, 200):
            for col in range(num_cols):
                if col == 0:
                    str_item = f'{self.df.iloc[row+start, col]}'
                else:
                    str_item = f'{self.df.iloc[row+start, col]:.2f}'
                item = QTableWidgetItem(str_item)
                self.tblSwipeInfo.setItem(row, col, item)

    def display_frame_info(self):
        try:
            # Clear table
            self.tblSwipeInfo.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = self.df.shape

            # Set the table widget dimensions
            self.tblSwipeInfo.setRowCount(num_rows)
            self.tblSwipeInfo.setColumnCount(num_cols)

            # Set the table headers
            self.tblSwipeInfo.setHorizontalHeaderLabels(self.df.columns)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    str_item = f'{self.df.iloc[row, col]:.2f}'
                    item = QTableWidgetItem(str_item)
                    self.tblSwipeInfo.setItem(row, col, item)
        except:
            print(traceback.format_exc())

    def select_row(self, row):
        if row - 10 < 0:
            self.display_range(0)
            sel_row = row
        elif row + 190 > self.df.shape[0]:
            self.display_range(self.df.shape[0] - 200)
            sel_row = self.df.shape[0] - 200
        else:
            self.display_range(row-10)
            sel_row = 10

        # Scroll to selected row
        self.tblSwipeInfo.scrollToItem(self.tblSwipeInfo.item(0, 0))
        
        # Highlight selected row
        self.tblSwipeInfo.selectRow(sel_row)

        # Refresh table
        self.tblSwipeInfo.viewport().update()

        right_hand_x = self.tblSwipeInfo.item(sel_row, 10).text()
        right_hand_y = self.tblSwipeInfo.item(sel_row, 11).text()
        right_hand_likelihood = self.tblSwipeInfo.item(sel_row, 12).text()

        pellet_x = self.tblSwipeInfo.item(sel_row, 4).text()
        pellet_y = self.tblSwipeInfo.item(sel_row, 5).text()
        pellet_likelihood = self.tblSwipeInfo.item(sel_row, 6).text()

        info = f'Right hand: ({right_hand_x}, {right_hand_y}), {right_hand_likelihood}. Pellet: ({pellet_x}, {pellet_y}), {pellet_likelihood}'

        return info

    def display_df(self, df):
        try:
            # Clear table
            self.tblSwipeInfo.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            # Set the table widget dimensions
            self.tblSwipeInfo.setRowCount(num_rows)
            self.tblSwipeInfo.setColumnCount(num_cols)

            # Set the table headers
            self.tblSwipeInfo.setHorizontalHeaderLabels(df.columns)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    # Check if item is string or float
                    if df.iloc[row, col] is None or df.iloc[row, col] == 'nan':
                        str_item = ''
                    else:
                        str_item = f'{df.iloc[row, col]:.2f}' if isinstance(df.iloc[row, col], float) else f'{df.iloc[row, col]}'
                    item = QTableWidgetItem(str_item)
                    self.tblSwipeInfo.setItem(row, col, item)
            
            # Resize columns to fit contents
            self.tblSwipeInfo.resizeColumnsToContents()
        except:
            print(traceback.format_exc())

    #endregion Swipe Info Viewer

    #region Analysis
            
    def save_swipe_analysis(self):
        self.df.to_excel('swipe_analysis.xlsx')

    def analyse_swipe_sequences_thread(self, swipe_sequences):
        thread = threading.Thread(target=self.analyse_swipe_sequences, args=(swipe_sequences,))
        thread.start()

    def analyse_swipe_sequences(self, swipe_sequences):

        try:
            reach_outcomes = self.determine_reach_outcomes(swipe_sequences)
            swipe_lengths  = self.determine_swipe_lengths(swipe_sequences)
            swipe_breadths = self.determine_swipe_breadth(swipe_sequences)
            swipe_areas    = self.determine_swipe_areas(swipe_sequences)
            swipe_speeds, swipe_distmm   = self.determine_swipe_speeds(swipe_sequences)
            swipe_ranges   = self.determine_swipe_range(swipe_sequences)
            tray_motion    = self.determine_tray_motion(swipe_sequences)

            self.df.insert(2, 'Reach outcome', reach_outcomes.values())
            self.df.insert(3, 'Swipe breadth', swipe_breadths.values())
            self.df.insert(4, 'Swipe length', swipe_lengths.values())
            self.df.insert(5, 'Swipe area', swipe_areas.values())
            self.df.insert(6, 'Swipe speed', swipe_speeds.values())
            self.df.insert(7, 'Tray motion', tray_motion.values())
            self.df.insert(8, 'Swipe range', swipe_ranges.values())
            self.df.insert(9, 'Swipe distance', swipe_distmm.values())

            if self.has_ui:
                self.display_df(self.df)
                #self.save_swipe_analysis()
                # return self.generate_summary()
            # else:
                # self.df.to_csv('swipe_analysis.csv', index=False)
                # Generate summary
                # return self.generate_summary()
            return self.generate_summary(), self.df
        except:
            print(traceback.format_exc())
        
    def generate_summary(self):

        summary = {}
        summary['Total swipes'] = self.df.shape[0]

        summary['Attention'] = f"{self.attention_frames / self.tot_frames * 100:.2f} %"
        
        if not self.df.empty:
            summary['Eaten'] = self.df['Reach outcome'].str.contains('swipe successful').sum()
        
            summary['Displaced'] = self.df['Reach outcome'].str.contains('pellet displaced').sum()
        
            summary['Missed'] = self.df['Reach outcome'].str.contains('swipe missed').sum()
            summary['Missed(on pillar)'] = self.df['Reach outcome'].str.contains("swipe missed (on pillar)",regex=False).sum()
            summary['Missed(off pillar)'] = summary['Missed'] - summary['Missed(on pillar)']

            df_S = self.df[self.df['Reach outcome'].str.contains('swipe successful')]
            df_D = self.df[self.df['Reach outcome'].str.contains('pellet displaced')]
            df_M = self.df[self.df['Reach outcome'].str.contains('swipe missed')]
            df_M_on_pillar = self.df[self.df['Reach outcome'].str.contains("swipe missed (on pillar)",regex=False)]
            df_M_off_pillar = df_M[~df_M.index.isin(df_M_on_pillar.index)]

            summary['Swipe length']  = self.get_mean(col='Swipe length')
            summary['Swipe length S'] = self.get_mean(df=df_S, col='Swipe length')
            summary['Swipe length D'] = self.get_mean(df=df_D, col='Swipe length')
            summary['Swipe length M'] = self.get_mean(df=df_M, col='Swipe length')
            summary['Swipe length M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe length')
            summary['Swipe length M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe length')

            summary['Swipe breadth'] = self.get_mean(col='Swipe breadth')
            summary['Swipe breadth S'] = self.get_mean(df=df_S, col='Swipe breadth')
            summary['Swipe breadth D'] = self.get_mean(df=df_D, col='Swipe breadth')
            summary['Swipe breadth M'] = self.get_mean(df=df_M, col='Swipe breadth')
            summary['Swipe breadth M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe breadth')
            summary['Swipe breadth M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe breadth')
            
            summary['Swipe area'] = self.get_mean(col='Swipe area')
            summary['Swipe area S'] = self.get_mean(df=df_S, col='Swipe area')
            summary['Swipe area D'] = self.get_mean(df=df_D, col='Swipe area')
            summary['Swipe area M'] = self.get_mean(df=df_M, col='Swipe area')
            summary['Swipe area M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe area')
            summary['Swipe area M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe area')

            summary['Swipe speed'] = self.get_mean(col='Swipe speed', unit='mm/s')
            summary['Swipe speed S'] = self.get_mean(df=df_S, col='Swipe speed', unit='mm/s')
            summary['Swipe speed D'] = self.get_mean(df=df_D, col='Swipe speed', unit='mm/s')
            summary['Swipe speed M'] = self.get_mean(df=df_M, col='Swipe speed', unit='mm/s')
            summary['Swipe speed M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe speed', unit='mm/s')
            summary['Swipe speed M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe speed', unit='mm/s')

            summary['Swipe duration'] = self.get_mean(col='Swipe range', unit='seconds')
            summary['Swipe duration S'] = self.get_mean(df=df_S, col='Swipe range', unit='seconds')
            summary['Swipe duration D'] = self.get_mean(df=df_D, col='Swipe range', unit='seconds')
            summary['Swipe duration M'] = self.get_mean(df=df_M, col='Swipe range', unit='seconds')
            summary['Swipe duration M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe range', unit='seconds')
            summary['Swipe duration M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe range', unit='seconds')

            summary['Swipe distance'] = self.get_mean(col='Swipe distance', unit='mm')
            summary['Swipe distance S'] = self.get_mean(df=df_S, col='Swipe distance', unit='mm')
            summary['Swipe distance D'] = self.get_mean(df=df_D, col='Swipe distance', unit='mm')
            summary['Swipe distance M'] = self.get_mean(df=df_M, col='Swipe distance', unit='mm')
            summary['Swipe distance M(on pillar)'] = self.get_mean(df=df_M_on_pillar, col='Swipe distance', unit='mm')
            summary['Swipe distance M(off pillar)'] = self.get_mean(df=df_M_off_pillar, col='Swipe distance', unit='mm')
        else:
            summary['Eaten'] = 0
            summary['Displaced'] = 0
            summary['Missed'] = 0
            summary['Missed(on pillar)'] = 0
            summary['Missed(off pillar)'] = 0

            summary['Swipe length'] = 0
            summary['Swipe length S'] = 0
            summary['Swipe length D'] = 0
            summary['Swipe length M'] = 0
            summary['Swipe length M(on pillar)'] = 0
            summary['Swipe length M(off pillar)'] = 0

            summary['Swipe breadth'] = 0
            summary['Swipe breadth S'] = 0
            summary['Swipe breadth D'] = 0
            summary['Swipe breadth M'] = 0
            summary['Swipe breadth M(on pillar)'] = 0
            summary['Swipe breadth M(off pillar)'] = 0

            summary['Swipe area'] = 0
            summary['Swipe area S'] = 0
            summary['Swipe area D'] = 0
            summary['Swipe area M'] = 0
            summary['Swipe area M(on pillar)'] = 0
            summary['Swipe area M(off pillar)'] = 0

            summary['Swipe speed'] = 0
            summary['Swipe speed S'] = 0
            summary['Swipe speed D'] = 0
            summary['Swipe speed M'] = 0
            summary['Swipe speed M(on pillar)'] = 0
            summary['Swipe speed M(off pillar)'] = 0

            summary['Swipe duration'] = 0
            summary['Swipe duration S'] = 0
            summary['Swipe duration D'] = 0
            summary['Swipe duration M'] = 0
            summary['Swipe duration M(on pillar)'] = 0
            summary['Swipe duration M(off pillar)'] = 0

            summary['Swipe distance'] = 0
            summary['Swipe distance S'] = 0
            summary['Swipe distance D'] = 0
            summary['Swipe distance M'] = 0
            summary['Swipe distance M(on pillar)'] = 0
            summary['Swipe distance M(off pillar)'] = 0
        summary = self.get_pellet_summary(summary)

        return summary
    
    def get_pellet_summary(self, summary):
        try: 
            # Add for every pellet how many swipes and whether it missed (0), displaced (1) or eaten (2)
            for pellet in range(1, 21):
                pellet_swipes = self.df[self.df['Pellet #'] == pellet]
                if len(pellet_swipes) == 0:
                    summary[f'{pellet}'] = f'[S0|D0|M0](N)({-1})'
                    continue
                
                # Check if pellet was missed, displaced, eaten
                eaten = pellet_swipes['Reach outcome'].str.contains('swipe successful').sum()
            
                displaced = pellet_swipes['Reach outcome'].str.contains('pellet displaced').sum()
                
                
                missed = pellet_swipes['Reach outcome'].str.contains('missed').sum()

                if 0 < eaten:
                    outcome = 'S'
                    frame_no = pellet_swipes[pellet_swipes['Reach outcome'].str.contains('swipe successful')]['bodyparts_coords'].values[0]
                elif 0 < displaced:
                    outcome = 'D'
                    frame_no = pellet_swipes[pellet_swipes['Reach outcome'].str.contains('pellet displaced')]['bodyparts_coords'].values[0]
                elif 0 < missed:
                    outcome = 'M'
                    frame_no = pellet_swipes[pellet_swipes['Reach outcome'].str.contains('swipe missed')]['bodyparts_coords'].values[0]
                else:
                    outcome = 'N'
                    frame_no = -1
                
                summary[f'{pellet}'] = f'[S{eaten}|D{displaced}|M{missed}]({outcome})({frame_no})'
        except:
            print(traceback.format_exc())

        return summary
     
    def get_mean(self, df=None, col='', unit='mm'):
        if df is None:
            df = self.df
        mean = np.mean(df[col].str.extract(f'([\d.]+)(?={unit})').astype(float))
        # return f'{mean:.2f} {unit}'
        return f'{mean:.2f}'

    #endregion Analysis

    #region Tray motion

    def determine_tray_motion(self, swipe_sequences):
        tray_motion = {}
        for key, value in swipe_sequences.items():
            tray_motion[key] = self.determine_tray_motion_for_sequence(value)

        return tray_motion
    
    def determine_tray_motion_for_sequence(self, swipe_sequence):
        # Get frames where pellet is moving
        std_dev = swipe_sequence['Pellet_x'].rolling(4).std()
        motion = ((std_dev > 2) & 
                        # ((swipe_sequence['Pellet_likelihood'] > 0.8) & (swipe_sequence['Pellet_likelihood'] < 1.0)) & 
                        (swipe_sequence['Pillar_likelihood'] < 0.5) & 
                        (swipe_sequence['RightHand_likelihood'] < 0.5)).astype(int)
        
        return motion.sum()
    
    #endregion Tray motion

    #region Range

    def determine_swipe_range(self, swipe_sequences):
        swipe_ranges = {}
        for key, value in swipe_sequences.items():
            swipe_ranges[key] = self.determine_swipe_range_for_sequence(value)

        return swipe_ranges
    
    def determine_swipe_range_for_sequence(self, swipe_sequence):
        beg = swipe_sequence['bodyparts_coords'].iloc[0]
        end = swipe_sequence['bodyparts_coords'].iloc[-1]
        num_frames = len(swipe_sequence)
        time = num_frames / 60
        swipe_range = f'{beg:.0f}-{end:.0f} ({num_frames:.0f} frames, {time:.2f}seconds)'

        return swipe_range

    #endregion Range

    #region Swipe length

    def determine_swipe_lengths(self, swipe_sequences):
        swipe_lengths = {}
        for key, value in swipe_sequences.items():
            swipe_lengths[key] = self.determine_swipe_length_for_sequence(value)

        return swipe_lengths
    
    def determine_swipe_length_for_sequence(self, swipe_sequence):
        # Get min x of right hand and max x of right hand where likelihood is high
        # swipe_sequence = swipe_sequence[(swipe_sequence['RightHand_likelihood'] > 0.9)]
        if swipe_sequence.shape[0] <= 1:
            return 'NA'
        max_y = swipe_sequence['RightHand_y'].max()
        # Get nose y for max y
        nose_y = swipe_sequence[swipe_sequence['RightHand_y'] == max_y]['Nose_y'].values[0]

        if nose_y > max_y:
            return 'NA'

        swipe_length = max_y - nose_y
        swipe_length = f'{swipe_length:.2f} ({swipe_length/4:.2f}mm)'

        return swipe_length
    
    #endregion Swipe length

    #region Swipe breadth

    def determine_swipe_breadth(self, swipe_sequences):
        swipe_breadths = {}
        for key, value in swipe_sequences.items():
            swipe_breadths[key] = self.determine_swipe_breadth_for_sequence(value)

        return swipe_breadths
    
    def determine_swipe_breadth_for_sequence(self, swipe_sequence):
        # max_y = swipe_sequence['RightHand_y'].max()
        # swipe_sequence = swipe_sequence[(swipe_sequence['RightHand_y'] >= max_y-10)]
        if len(swipe_sequence) <= 1:
            return 'NA'
        min_x = swipe_sequence['RightHand_x'].min()
        max_x = swipe_sequence['RightHand_x'].max()

        swipe_breadth = max_x - min_x
        swipe_breadth = f'{swipe_breadth:.2f} ({swipe_breadth/4:.2f}mm)'

        return swipe_breadth
    
    #endregion Swipe breadth

    #region Area

    def determine_swipe_areas(self, swipe_sequences):
        swipe_areas = {}
        for key, value in swipe_sequences.items():
            swipe_areas[key] = self.determine_swipe_area_for_sequence(value)

        return swipe_areas
    
    def determine_swipe_area_for_sequence(self, swipe_sequence):
        # Use shoelace formula to calculate area of polygon
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # Get x and y coordinates of right hand where likelihood is high
        # swipe_sequence = swipe_sequence[(swipe_sequence['RightHand_likelihood'] > 0.9)]
        if swipe_sequence.shape[0] <= 1:
            return 'NA'
        x = swipe_sequence['RightHand_x'].values
        y = swipe_sequence['RightHand_y'].values

        # Calculate area
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        area = f'{area:.2f} ({area/16:.2f}mm^2)'
        
        return area

    #endregion Area

    #region Swipe speed

    def determine_swipe_speeds(self, swipe_sequences):
        swipe_speeds = {}
        swipe_distancesmm = {}
        for key, value in swipe_sequences.items():
            swipe_speeds[key], swipe_distancesmm[key] = self.determine_swipe_speed_for_sequence(value)

        return swipe_speeds, swipe_distancesmm
    
    def determine_swipe_speed_for_sequence(self, swipe_sequence):
        # Calculate swipe speed using right hand x,y coordinates and number of frames in the sequence
        # swipe_seq = swipe_sequence[(swipe_sequence['RightHand_likelihood'] > 0.9)]

        if len(swipe_sequence) <= 1:
            return 'NA'

        # Calculate distance traveled
        x = swipe_sequence['RightHand_x'].values
        y = swipe_sequence['RightHand_y'].values

        # Calculate distance traveled
        distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        distance = np.sum(distance)
        distancemm = distance/4

        # Calculate time taken
        time = swipe_sequence['bodyparts_coords'].iloc[-1] - swipe_sequence['bodyparts_coords'].iloc[0]
        time_sec = time/60

        # Calculate speed
        speed = distancemm/time_sec
        speed = f'{speed:.2f}mm/s'
        distancemm = f'{distance:.2f}({distancemm:.2f}mm)'

        return speed, distancemm
    
    #endregion Swipe speed

    #region Reach outcome         

    def distance(self, x1, y1, x2, y2):
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance

    def determine_reach_outcomes(self, swipe_sequences):
        reach_outcomes = {}
        for key, value in swipe_sequences.items():
            reach_outcomes[key] = self.determine_reach_outcome(value)

        return reach_outcomes
    
    def determine_reach_outcome(self, swipe_sequence):
        group = swipe_sequence['bodyparts_coords'].values.tolist()
        # expand group by 3 frames at the beginning and end
        group = list(range(group[0]-6, group[0])) + group + list(range(group[-1]+1, group[-1]+7))
        pellet_df = self.df_full[self.df_full['bodyparts_coords'].isin(group)]

        # Check if mouse hand is not obtructing in the last 3 frames
        df = pellet_df.iloc[-3:]
        df = df[(df['Pellet_y']-df['RightHand_y'] > 8)|(df['Pellet_likelihood'] > 0.5)]
        if len(df) < 3:
            # extend group by 6 frames at the end
            group = group + list(range(group[-1]+1, group[-1]+10))
            pellet_df = self.df_full[self.df_full['bodyparts_coords'].isin(group)]

        # Check if pellet is detected with high likelihood
        begin_pellet_likelihood = pellet_df.iloc[0:3]['Pellet_likelihood'].mean()
        end_pellet_likelihood   = pellet_df.iloc[-3:]['Pellet_likelihood'].mean()

        b_pellet_on_pillar_begin = False
        b_pellet_on_pillar_end = False

        avg_pellet_likelihood = pellet_df['Pellet_likelihood'].mean()

        # Get begin pellet location by getting the location from any of the first 6 frames with high likelihood
        begin_pellet = pellet_df.iloc[0:6][(pellet_df.iloc[0:6]['Pellet_likelihood'] > 0.95)]
        beg_pellet_x_median = -1
        beg_pellet_y_median = -1
        if len(begin_pellet) > 0:
            beg_pellet_x_median = begin_pellet['Pellet_x'].median()
            beg_pellet_y_median = begin_pellet['Pellet_y'].median()

        # Get end pellet location by getting the location from any of the last 6 frames with high likelihood
        end_pellet = pellet_df.iloc[-6:][(pellet_df.iloc[-6:]['Pellet_likelihood'] > 0.95)]
        end_pellet_x_median = -1
        end_pellet_y_median = -1
        if len(end_pellet) > 0:
            end_pellet_x_median = end_pellet['Pellet_x'].median()
            end_pellet_y_median = end_pellet['Pellet_y'].median()

        pellet_displacement = -1
        if len(begin_pellet) > 0 and len(end_pellet) > 0:
            # Calculate distance between begin and end pellet locations
            pellet_displacement = self.distance(beg_pellet_x_median, beg_pellet_y_median, end_pellet_x_median, end_pellet_y_median)

        # Pellet on pillar
        try:
            pell_on_pill_x = self.pellet_loc_df[(self.pellet_loc_df['start'] <= group[0]) & 
                                                (self.pellet_loc_df['end'] >= group[-1])]['pell_x'].values[0]
            pell_on_pill_y = self.pellet_loc_df[(self.pellet_loc_df['start'] <= group[0]) & 
                                                (self.pellet_loc_df['end'] >= group[-1])]['pell_y'].values[0]
        except:
            pell_on_pill_x = self.pellet_x_median
            pell_on_pill_y = self.pellet_y_median

        # Check if pellet is on pillar by checking in any of the first 6 frames
        beg_distance = self.distance(pell_on_pill_x, pell_on_pill_y, beg_pellet_x_median, beg_pellet_y_median)
        if beg_distance < self.pellet_pos_th:
            b_pellet_on_pillar_begin = True

        # Check if pellet is on pillar by checking in any of the last 6 frames
        distance = self.distance(pell_on_pill_x, pell_on_pill_y, end_pellet_x_median, end_pellet_y_median)
        if distance < self.pellet_pos_th:
            b_pellet_on_pillar_end = True

        # Check if pellet moved
        pellet_y_std = pellet_df['Pellet_y'].rolling(4).std()
        pellet_x_std = pellet_df['Pellet_x'].rolling(4).std()
        b_pellet_moved = (pellet_y_std.mean() > 1) or (pellet_x_std.mean() > 1) or (pellet_displacement > 0.6)

        if begin_pellet_likelihood < 0.5 and avg_pellet_likelihood < 0.5:
            return "no pellet"
        elif (b_pellet_moved and 
              end_pellet_likelihood > 0.5 and 
              b_pellet_on_pillar_begin and 
               not b_pellet_on_pillar_end):
            return "pellet displaced"
        elif ((begin_pellet_likelihood > 0.5 and 
               end_pellet_likelihood < 0.5 and 
               b_pellet_on_pillar_begin) or
             (pellet_displacement > 60)):
            return "swipe successful"
        elif b_pellet_on_pillar_begin and b_pellet_on_pillar_end:
            return f"swipe missed (on pillar) {pellet_displacement:.2f} {beg_distance:.2f} {distance:.2f}"
        else:
            return f"swipe missed {pellet_displacement:.2f} {beg_distance:.2f} {distance:.2f}"


        
        if   (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood < 0.9 and 
              end_pillar_likelihood > 0.9 and
              end_pellet_likelihood < 0.5):
            return "swipe successful (on pillar)"
        elif (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood > 0.9 and 
              end_pillar_likelihood > 0.9 and
              end_pellet_likelihood < 0.5):
            return "swipe successful (~ on pillar)"
        elif (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood < 0.9 and 
              end_pillar_likelihood > 0.8 and
              end_pellet_likelihood > 0.9):
            return "pellet displaced"
        elif (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood < 0.9 and 
              end_pillar_likelihood < 0.9 and
              end_pellet_likelihood > 0.9):
            return "swipe missed (on pillar)"
        elif (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood > 0.9 and 
              end_pillar_likelihood > 0.9 and
              end_pellet_likelihood > 0.5):
            return "swipe missed (~ on pillar)"
        elif (begin_pellet_likelihood > 0.9 and 
              begin_pillar_likelihood > 0.9 and 
              end_pillar_likelihood < 0.9 and
              end_pellet_likelihood > 0.9):
            return "pellet displaced to pillar"
        elif (begin_pellet_likelihood < 0.9 and 
              begin_pillar_likelihood < 0.9 and 
              end_pillar_likelihood > 0.9 and
              end_pellet_likelihood > 0.9):
            return "swipe missed (~ on pillar)"
        elif (begin_pellet_likelihood < 0.9 and 
              begin_pillar_likelihood > 0.9 and 
              end_pillar_likelihood > 0.9 and
              end_pellet_likelihood > 0.9):
            return "swipe missed (~ on pillar)"
        elif avg_pillar_likelihood > 0.5 and avg_pellet_likelihood < 0.5:
            return "no pellet"
        elif avg_pellet_likelihood < 0.5 and avg_pillar_likelihood < 0.5:
            return "no pellet no pillar"
        elif avg_pellet_likelihood > 0.5 and avg_pillar_likelihood > 0.5:
            return "swipe missed (~ on pillar)"
        elif avg_pellet_likelihood > 0.5 and avg_pillar_likelihood < 0.5:
            return "swipe missed (on pillar)"
        elif avg_pellet_likelihood < 0.5 and avg_pillar_likelihood > 0.5:
            return "swipe successful"
        else:
            return "NA"

    #endregion Reach outcome

    #region Filter

    def filter_data(self):
        reach_txt = self.reachOutcomeFilter.text()
        filtered_df = self.df[self.df['Reach outcome'].apply(lambda x: reach_txt.lower() in str(x).lower())]

        self.display_df(filtered_df)

    #endregion Filter

if __name__ == "__main__":
    show_gui = True
    swipe_info = SwipeInfo(ui=False)
    swipe_info.read_csv_thread(file_path='20220721_H36_E2DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')
    # swipe_info.analyse_swipe_sequences_thread(swipe_sequences=swipe_info.swipe_sequences)
