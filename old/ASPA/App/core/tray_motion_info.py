import sys
import threading
import time
import traceback
import numpy as np
import pandas as pd

from qtpy.QtWidgets import QTableWidgetItem

class TrayMotionInfo:
    def __init__(self, tbl=None, ui=True, reachFilter=None, df=None) -> None:
        self.tblTrayMotion = tbl
        self.df = df
        self.has_ui = ui

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

    #region Tray Motion Info Viewer

    def display_range(self, start):
        # Clear table
        self.tblTrayMotion.clear()
        # Display dataframe in pyqt table widget
        # Get the number of rows and columns
        num_rows, num_cols = self.df.shape

        # Set the table widget dimensions
        self.tblTrayMotion.setRowCount(200)
        self.tblTrayMotion.setColumnCount(num_cols)

        # Set the table headers
        self.tblTrayMotion.setHorizontalHeaderLabels(self.df.columns)

        # Populate the table widget with data
        for row in range(0, 200):
            for col in range(num_cols):
                if col == 0:
                    str_item = f'{self.df.iloc[row+start, col]}'
                else:
                    str_item = f'{self.df.iloc[row+start, col]:.2f}'
                item = QTableWidgetItem(str_item)
                self.tblTrayMotion.setItem(row, col, item)

    def display_frame_info(self):
        try:
            # Clear table
            self.tblTrayMotion.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = self.df.shape

            # Set the table widget dimensions
            self.tblTrayMotion.setRowCount(num_rows)
            self.tblTrayMotion.setColumnCount(num_cols)

            # Set the table headers
            self.tblTrayMotion.setHorizontalHeaderLabels(self.df.columns)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    str_item = f'{self.df.iloc[row, col]:.2f}'
                    item = QTableWidgetItem(str_item)
                    self.tblTrayMotion.setItem(row, col, item)
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
        self.tblTrayMotion.scrollToItem(self.tblTrayMotion.item(0, 0))
        
        # Highlight selected row
        self.tblTrayMotion.selectRow(sel_row)

        # Refresh table
        self.tblTrayMotion.viewport().update()

        right_hand_x = self.tblTrayMotion.item(sel_row, 10).text()
        right_hand_y = self.tblTrayMotion.item(sel_row, 11).text()
        right_hand_likelihood = self.tblTrayMotion.item(sel_row, 12).text()

        pellet_x = self.tblTrayMotion.item(sel_row, 4).text()
        pellet_y = self.tblTrayMotion.item(sel_row, 5).text()
        pellet_likelihood = self.tblTrayMotion.item(sel_row, 6).text()

        info = f'Right hand: ({right_hand_x}, {right_hand_y}), {right_hand_likelihood}. Pellet: ({pellet_x}, {pellet_y}), {pellet_likelihood}'

        return info

    def display_df(self, df):
        try:
            # Clear table
            self.tblTrayMotion.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            # Set the table widget dimensions
            self.tblTrayMotion.setRowCount(num_rows)
            self.tblTrayMotion.setColumnCount(num_cols)

            # Set the table headers
            self.tblTrayMotion.setHorizontalHeaderLabels(df.columns)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    if (col == 1) or (col == 2) or (col == 3) or (col == 4) or (col == 5) or (col == 6):
                        str_item = f'{df.iloc[row, col]}'
                    else:
                        str_item = f'{df.iloc[row, col]:.2f}'
                    item = QTableWidgetItem(str_item)
                    self.tblTrayMotion.setItem(row, col, item)
            
            # Resize columns to fit contents
            self.tblTrayMotion.resizeColumnsToContents()
        except:
            print(traceback.format_exc())

    #endregion Tray Motion Info Viewer

    #region Analysis

    def analyse_tray_motion_sequences_thread(self, tray_motion_sequences):
        thread = threading.Thread(target=self.analyse_tray_motion_sequences, args=(tray_motion_sequences,))
        thread.start()

    def analyse_tray_motion_sequences(self, tray_motion_sequences):

        reach_outcomes = self.determine_reach_outcomes(tray_motion_sequences)
        tray_motion_lengths  = self.determine_tray_motion_lengths(tray_motion_sequences)
        tray_motion_breadths = self.determine_tray_motion_breadth(tray_motion_sequences)
        tray_motion_areas    = self.determine_tray_motion_areas(tray_motion_sequences)
        tray_motion_speeds   = self.determine_tray_motion_speeds(tray_motion_sequences)
        tray_motion_ranges   = self.determine_tray_motion_range(tray_motion_sequences)

        self.df.insert(1, 'Reach outcome', reach_outcomes.values())
        self.df.insert(2, 'Tray Motion breadth', tray_motion_breadths.values())
        self.df.insert(3, 'Tray Motion length', tray_motion_lengths.values())
        self.df.insert(4, 'Tray Motion area', tray_motion_areas.values())
        self.df.insert(5, 'Tray Motion speed', tray_motion_speeds.values())
        self.df.insert(6, 'Tray Motion range', tray_motion_ranges.values())

        if self.has_ui:
            self.display_df(self.df)
        else:
            self.df.to_csv('tray_motion_analysis.csv', index=False)


    #endregion Analysis

    #region Range

    def determine_tray_motion_range(self, tray_motion_sequences):
        tray_motion_ranges = {}
        for key, value in tray_motion_sequences.items():
            tray_motion_ranges[key] = self.determine_tray_motion_range_for_sequence(value)

        return tray_motion_ranges
    
    def determine_tray_motion_range_for_sequence(self, tray_motion_sequence):
        beg = tray_motion_sequence['bodyparts_coords'].iloc[0]
        end = tray_motion_sequence['bodyparts_coords'].iloc[-1]
        num_frames = end - beg
        time = num_frames / 60
        tray_motion_range = f'{beg:.0f}-{end:.0f} ({num_frames:.0f} frames, {time:.2f} seconds))'

        return tray_motion_range

    #endregion Range

    #region Tray Motion length

    def determine_tray_motion_lengths(self, tray_motion_sequences):
        tray_motion_lengths = {}
        for key, value in tray_motion_sequences.items():
            tray_motion_lengths[key] = self.determine_tray_motion_length_for_sequence(value)

        return tray_motion_lengths
    
    def determine_tray_motion_length_for_sequence(self, tray_motion_sequence):
        # Get min x of right hand and max x of right hand where likelihood is high
        tray_motion_sequence = tray_motion_sequence[(tray_motion_sequence['RightHand_likelihood'] > 0.9)]
        min_x = tray_motion_sequence['RightHand_y'].min()
        max_x = tray_motion_sequence['RightHand_y'].max()

        tray_motion_length = max_x - min_x
        tray_motion_length = f'{tray_motion_length:.2f} ({tray_motion_length/4:.2f}mm)'

        return tray_motion_length
    
    #endregion Tray Motion length

    #region Tray Motion breadth

    def determine_tray_motion_breadth(self, tray_motion_sequences):
        tray_motion_breadths = {}
        for key, value in tray_motion_sequences.items():
            tray_motion_breadths[key] = self.determine_tray_motion_breadth_for_sequence(value)

        return tray_motion_breadths
    
    def determine_tray_motion_breadth_for_sequence(self, tray_motion_sequence):
        # Get min x of right hand and max x of right hand where likelihood is high
        tray_motion_sequence = tray_motion_sequence[(tray_motion_sequence['RightHand_likelihood'] > 0.9) & (tray_motion_sequence['RightHand_y'] >= 440)]
        min_x = tray_motion_sequence['RightHand_x'].min()
        max_x = tray_motion_sequence['RightHand_x'].max()

        tray_motion_breadth = max_x - min_x
        tray_motion_breadth = f'{tray_motion_breadth:.2f} ({tray_motion_breadth/4:.2f}mm)'

        return tray_motion_breadth
    
    #endregion Tray Motion breadth

    #region Area

    def determine_tray_motion_areas(self, tray_motion_sequences):
        tray_motion_areas = {}
        for key, value in tray_motion_sequences.items():
            tray_motion_areas[key] = self.determine_tray_motion_area_for_sequence(value)

        return tray_motion_areas
    
    def determine_tray_motion_area_for_sequence(self, tray_motion_sequence):
        # Use shoelace formula to calculate area of polygon
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # Get x and y coordinates of right hand where likelihood is high
        tray_motion_sequence = tray_motion_sequence[(tray_motion_sequence['RightHand_likelihood'] > 0.9)]
        x = tray_motion_sequence['RightHand_x'].values
        y = tray_motion_sequence['RightHand_y'].values

        # Calculate area
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        area = f'{area:.2f} ({area/16:.2f} mm^2)'
        return area

    #endregion Area

    #region Tray Motion speed

    def determine_tray_motion_speeds(self, tray_motion_sequences):
        tray_motion_speeds = {}
        for key, value in tray_motion_sequences.items():
            tray_motion_speeds[key] = self.determine_tray_motion_speed_for_sequence(value)

        return tray_motion_speeds
    
    def determine_tray_motion_speed_for_sequence(self, tray_motion_sequence):
        # Calculate tray_motion speed using right hand x,y coordinates and number of frames in the sequence
        tray_motion_seq = tray_motion_sequence[(tray_motion_sequence['RightHand_likelihood'] > 0.9)]

        if len(tray_motion_seq) == 0:
            return 'N/A'

        # Calculate distance traveled
        x = tray_motion_seq['RightHand_x'].values
        y = tray_motion_seq['RightHand_y'].values

        # Calculate distance traveled
        distance = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        distance = np.sum(distance)
        distancemm = distance/4

        # Calculate time taken
        time = tray_motion_seq['bodyparts_coords'].iloc[-1] - tray_motion_seq['bodyparts_coords'].iloc[0]
        time_sec = time/60

        # Calculate speed
        speed = distancemm/time_sec
        speed = f'{speed:.2f} mm/s'

        return speed
    
    #endregion Tray Motion speed

    #region Reach outcome            

    def determine_reach_outcomes(self, tray_motion_sequences):
        reach_outcomes = {}
        for key, value in tray_motion_sequences.items():
            reach_outcomes[key] = self.determine_reach_outcome(value)

        return reach_outcomes
    
    def determine_reach_outcome(self, tray_motion_sequence):
        # Check if pellet is detected with high likelihood
        if tray_motion_sequence.iloc[0]['Pellet_likelihood'] < 0.9:
            return "no pellet"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] < 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] > 0.9 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] < 0.9):
            return "tray_motion successful (on pillar)"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] > 0.9 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] < 0.9):
            return "tray_motion successful (~ on pillar)"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] < 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] > 0.8 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] > 0.9):
            return "pellet displaced"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] < 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] < 0.9 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] > 0.9):
            return "tray_motion missed (on pillar)"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] > 0.9 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] > 0.9):
            return "tray_motion missed (~ on pillar)"
        elif (tray_motion_sequence.iloc[0]['Pellet_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[0]['Pillar_likelihood'] > 0.9 and 
              tray_motion_sequence.iloc[-1]['Pillar_likelihood'] < 0.9 and
              tray_motion_sequence.iloc[-1]['Pellet_likelihood'] > 0.9):
            return "pellet displaced to pillar"

    #endregion Reach outcome

    #region Filter

    def filter_data(self):
        reach_txt = self.reachOutcomeFilter.text()
        filtered_df = self.df[self.df['Reach outcome'].apply(lambda x: reach_txt.lower() in str(x).lower())]

        self.display_df(filtered_df)

    #endregion Filter

if __name__ == "__main__":
    show_gui = True
    tray_motion_info = TrayMotionInfo(ui=False)
    tray_motion_info.read_csv_thread(file_path='20220721_H36_E2DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')
    # tray_motion_info.analyse_tray_motion_sequences_thread(tray_motion_sequences=tray_motion_info.tray_motion_sequences)
