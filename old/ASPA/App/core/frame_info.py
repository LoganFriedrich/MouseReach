import threading
import traceback
import pandas as pd
from qtpy.QtWidgets import QTableWidgetItem

class FrameInfo:

    def __init__(self, tbl, ui=True) -> None:
        self.tblFrameInfo = tbl
        self.has_ui = ui
        self.num_rows = 0

    #region Frame Info Viewer

    def read_csv_thread(self, file_path):
        thread = threading.Thread(target=self.read_csv, args=(file_path,))
        thread.start()

    def read_csv(self, csv_file):
        if csv_file == '':
            return

        comb = '_'

        # Create headers from first 3 rows, 1st is unused, 2nd is part name, 3rd is x/y/likelihood
        cols = [x+comb+y for x, y in list(zip(pd.read_csv(csv_file, nrows=2).values.tolist()[0], 
                                              pd.read_csv(csv_file, nrows=2).values.tolist()[1]))]

        # get actual data
        self.df = pd.read_csv(csv_file, skiprows=3, header=None)

        # add headers
        self.df.columns = cols
        
        # Add a column for pellet number with values as zeros
        self.df['Pellet #'] = 0


        self.num_rows = self.df.shape[0]

        if self.has_ui:
            self.display_frame_info()

    def display_range(self, start):
        try:
            # Clear table
            self.tblFrameInfo.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = self.df.shape

            # Set the table widget dimensions
            self.tblFrameInfo.setRowCount(200)
            self.tblFrameInfo.setColumnCount(num_cols)

            # Set the table headers
            self.tblFrameInfo.setHorizontalHeaderLabels(self.df.columns)

            # Populate the table widget with data
            for row in range(0, 200):
                for col in range(num_cols):
                    if col == 0:
                        str_item = f'{self.df.iloc[row+start, col]}'
                    else:
                        str_item = f'{self.df.iloc[row+start, col]:.2f}'
                    item = QTableWidgetItem(str_item)
                    self.tblFrameInfo.setItem(row, col, item)
        except:
            print(traceback.format_exc())

    def display_frame_info(self):
        try:
            # Clear table
            self.tblFrameInfo.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = self.df.shape

            # Set the table widget dimensions
            self.tblFrameInfo.setRowCount(200)
            self.tblFrameInfo.setColumnCount(num_cols)

            # Set the table headers
            self.tblFrameInfo.setHorizontalHeaderLabels(self.df.columns)

            # Populate the table widget with data
            for row in range(0, 200):
                for col in range(num_cols):
                    str_item = f'{self.df.iloc[row, col]:.2f}'
                    item = QTableWidgetItem(str_item)
                    self.tblFrameInfo.setItem(row, col, item)
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
        self.tblFrameInfo.scrollToItem(self.tblFrameInfo.item(0, 0))
        
        # Highlight selected row
        self.tblFrameInfo.selectRow(sel_row)

        # Refresh table
        self.tblFrameInfo.viewport().update()

        right_hand_x = self.tblFrameInfo.item(sel_row, 10).text()
        right_hand_y = self.tblFrameInfo.item(sel_row, 11).text()
        right_hand_likelihood = self.tblFrameInfo.item(sel_row, 12).text()

        pellet_x = self.tblFrameInfo.item(sel_row, 4).text()
        pellet_y = self.tblFrameInfo.item(sel_row, 5).text()
        pellet_likelihood = self.tblFrameInfo.item(sel_row, 6).text()

        info = f'Right hand: ({right_hand_x}, {right_hand_y}), {right_hand_likelihood}. Pellet: ({pellet_x}, {pellet_y}), {pellet_likelihood}'

        return info

    def display_df_thread(self, df):
        thread = threading.Thread(target=self.display_df, args=(df,))
        thread.start()
        
    def display_df(self, df):
        try:
            # Clear table
            self.tblFrameInfo.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            # Set the table widget dimensions
            self.tblFrameInfo.setRowCount(num_rows)
            self.tblFrameInfo.setColumnCount(num_cols)

            # Set the table headers
            self.tblFrameInfo.setHorizontalHeaderLabels(df.columns)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    # Check if item is string or float
                    if df.iloc[row, col] is None or df.iloc[row, col] == 'nan':
                        str_item = ''
                    else:
                        str_item = f'{df.iloc[row, col]:.2f}' if isinstance(df.iloc[row, col], float) else f'{df.iloc[row, col]}'
                    item = QTableWidgetItem(str_item)
                    self.tblFrameInfo.setItem(row, col, item)
        except:
            print(traceback.format_exc())

    #endregion Frame Info Viewer