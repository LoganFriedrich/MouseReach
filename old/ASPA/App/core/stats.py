from datetime import datetime
import re
import os
import threading
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from qtpy.QtWidgets import QTableWidgetItem
import numpy as np

class Stats:
    def __init__(self, animal_group='H', ai=True, ui=True) -> None:
        self.df = None # DeepLabCut data for a video
        self.has_ui = ui
        self.animal_group = animal_group
        self.ai = ai
        self.init_stat_viewer()

    def init_stat_viewer(self):
        if self.has_ui:
            # create a figure
            self.fig_stats = plt.figure()
            self.fig_stats.patch.set_facecolor('lightgrey')
            self.can_stats = FigureCanvas(self.fig_stats)
            self.ax = self.fig_stats.add_subplot(111)
            self.fig_stats.subplots_adjust(wspace=0.15, hspace=0.25,
                                        top=0.82, bottom=0.13, left=0.05, right=0.97)
        
            self.read_folder_data()

    def show_stats_thread(self, df):
        thread = threading.Thread(target=self.show_stats, args=(df,))
        thread.start()

    def show_stats(self, df=None):
        pass
        
    def show_stats_dep(self, df=None):

        if df is not None:
            self.df = df
        
        num_rows, num_cols = self.df.shape

        # Create a histogram for each column
        cols = 6
        col_idx = 1
        for i in range(num_cols//cols):
            for j in range(cols):
                ax = self.fig_stats.add_subplot(num_cols//cols, cols, col_idx)
                ax.hist(self.df.iloc[:, col_idx], bins=50)
                ax.set_title(self.df.columns[col_idx])
                col_idx += 1

    def read_folder_data(self):
        self.info_df = pd.read_excel(os.path.join('Human observed', 'Others', 'Test Type Date Ranges.xlsx'))
        self.dates = self.info_df['Test Date'].dt.strftime('%d-%b-%Y').unique()
        # for file in os.listdir('Human observed'):
        #     if file.endswith('.xlsx'):
        #         try:
        #             info_df = pd.read_excel(os.path.join('Human observed', file), sheet_name='Data') # usecols='A:G'
        #             if hasattr(self, 'info_df'):
        #                 self.info_df = pd.concat([self.info_df, info_df])
        #             else:
        #                 self.info_df = info_df
        #         except:
        #             print(traceback.format_exc())
        self.plot_df = None
        for file in os.listdir('AI'):
            if file.endswith('.xlsx'):
                plot_df = pd.read_excel(os.path.join('AI', file))
                if hasattr(self, 'plot_df'):
                    self.plot_df = pd.concat([self.plot_df, plot_df])
                else:
                    self.plot_df = plot_df
        self.update_plot_df()
        #filter = {'Group':'H', 'Animal ID': ''}
        #self.plot_swipe_feature('Attention', filter=filter)

    def parse_values(self, col, filter=None):
        df = self.plot_df
        if filter['Animal ID'] != '':
            animal_id = filter['Animal Group'] + filter['Animal ID']
            df = df[df['Animal ID'] == animal_id]
        if filter['Test Date'] != '':
            df = df[df['Date'] == filter['Test Date']]
        if filter['Test Type'] != '':
            df = df[df['Test Type'] == filter['Test Type']]
            
        valid = df[col].dropna()  
        
        if not valid.empty:
            total  = valid.str.split(' ').str[-1].str.strip('()')
            values = valid.str.split(' ').str[0].str.strip('[]').str.split('|')
            return [values.str[i].astype(int).mean() for i in range(3)] + [total.str[0].astype(int).mean()]
        else: 
            return [0, 0, 0, 0] 

    def plot_pellet_stats(self, feature, plot_type, plot_group, filter=None):
        # blank df
        missed = []
        displaced = [] 
        retrieved = []
        total = []

        for i in range(1, 21):
            parsed = self.parse_values('P' + str(i), filter)
            missed.append(parsed[0])
            displaced.append(parsed[1])
            retrieved.append(parsed[2])
            total.append(parsed[3])

        index = np.arange(len(total))
        bar_width = 0.25

        self.ax.bar(index, missed, bar_width, label='Missed')
        self.ax.bar(index + bar_width, displaced, bar_width, label='Displaced')
        self.ax.bar(index + 2*bar_width, retrieved, bar_width, label='Retrieved')

        self.ax.set_xlabel('Pellets')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Pellet statistics')

        self.ax.set_xticks(index + 1.5*bar_width)
        self.ax.set_xticklabels(['P' + str(i+1) for i in range(len(total))])
        self.ax.tick_params(axis='x')

        self.ax.legend()
        self.fig_stats.tight_layout()

    def plot_swipe_feature(self, feature, plot_type, plot_group, filter=None):
        self.ax.clear()

        # if self.animal_group != filter['Group']:
        #     self.animal_group = filter['Group']
        #     self.read_folder_data()

        if plot_group == 'Pellets':
            self.plot_pellet_stats(feature, plot_type, plot_group, filter)
            self.can_stats.draw()
            return
        
        plot_df = self.plot_df

        if not feature == 'Total swipes':
            # Check if column is string
            if self.plot_df[feature].dtype == 'O':
                plot_df[feature] = plot_df[feature].str.split(' ').str[0]
                # Convert column to numeric
                plot_df[feature] = pd.to_numeric(plot_df[feature])

        if len(filter['Animal Group']) and 'All' not in filter['Animal Group']:
            # Get rows where animal ID is in group
            plot_df = plot_df[plot_df['Animal Group'].isin(filter['Animal Group'])]

        if len(filter['Test Date']) and 'All' not in filter['Test Date']:
            plot_df = plot_df[plot_df['Date'].isin(filter['Test Date'])]

        if len(filter['Test Phase']) and 'All' not in filter['Test Phase']:
            plot_df = plot_df[plot_df['Test Phase'].isin(filter['Test Phase'])]

        if len(filter['Tray Type']) and 'All' not in filter['Tray Type']:
            plot_df = plot_df[plot_df['Tray Type'].isin(filter['Tray Type'])]
        # Add Group to plot_df

        # plot_df['Group'] = plot_df['Animal ID'].str[0]

        # if len(filter['Test Type']) and 'All' not in filter['Test Type']:
        #     df = self.info_df[self.info_df['Test Type'].str.contains('|'.join(map(re.escape, filter['Test Type'])), na=False)]
        #     # df = self.info_df[self.info_df['Test Type'].str.contains(filter['Test Type'], na=False)]
        # else:
        #     df = self.info_df

        # if len(filter['Tray Type']) and 'All' not in filter['Tray Type']:
        #     df = df[df['Tray Type'].str.contains('|'.join(map(re.escape, filter['Tray Type'])), na=False)]

        # # Replace values in 'Test Type' column
        # df['Test Type'] = df['Test Type'].str.replace(r'.*Train.*', 'Train', regex=True)
        # df['Test Type'] = df['Test Type'].str.replace(r'.*Post-Injury.*', 'Post-injury', regex=True)
        # df['Test Type'] = df['Test Type'].str.replace(r'.*Rehab.*', 'Rehab', regex=True)
        # # Filter plot_df where 'Date' is in the corrected 'Test Date'
        # plot_df = plot_df[plot_df['Date'].isin(df['Test Date'])]

        # # plot_df['Date'] = pd.to_datetime(plot_df['Date'], format='%d-%b-%Y')
        # # df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        # plot_df = plot_df.set_index('Date') 
        # df = df.set_index('Test Date')

        # # Add Test Type to plot_df
        # plot_df = plot_df.join(df['Test Type'], on='Date', how='left')

        # df.reset_index(inplace=True)

        # Groupby Animal ID and Date, take mean Total Swipes
        df_swipe = plot_df.groupby(['Animal ID', 'Date', 'Test Phase', 'Animal Group', 'Tray Type', 'Injury Type'])[feature].mean().reset_index()

        # colors for plot_group
        colors = []
        for i in range(len(df_swipe[plot_group])):
            colors.append(plt.cm.tab20(i))

        if len(filter['Animal ID']) == 0:
            if plot_type == 'Box':
                # box plot of all animals
                df_swipe.boxplot(column=feature, by=plot_group, ax=self.ax)
            # elif plot_type == 'Violin':
            #     # violin plot of all animals
            #     self.ax.violinplot(df_swipe[plot_group], df_swipe[feature], 
            #               showmeans=True, showmedians=True)
            elif plot_type == 'Scatter':
                # scatter plot of all animals
                df_swipe.plot.scatter(x=plot_group, y=feature, c=colors, ax=self.ax)
            elif plot_type == 'Line':
                # line plot of all animals
                df_swipe.plot(x=plot_group, y=feature, c=colors, ax=self.ax)
            elif plot_type == 'GroupedBar' and (plot_group == 'Test Phase' or 
                                                plot_group == 'Animal Group' or 
                                                plot_group == 'Tray Type' or 
                                                plot_group == 'Injury Type'):
                # Grouped bar plot of all animals
                df_swipe.groupby(['Animal ID', plot_group])[feature].mean().unstack().plot(kind='bar', ax=self.ax)

            # for animal_id in df_swipe['Animal ID'].unique():
            #     df_swipe[df_swipe['Animal ID'] == animal_id].plot(x='Date', y=feature, label=animal_id, marker='o', ax=self.ax)
        elif len(filter['Group']) > 0 and len(filter['Animal ID']) > 0:
            for group in filter['Group']:
                for animal_id in filter['Animal ID']:
                    combined_id = group + animal_id
                    df_swipe[df_swipe['Animal ID'] == combined_id].plot(x='Date', y=feature, label=combined_id, marker='o', ax=self.ax)

        # self.ax.set_title(feature)
        # self.ax.set_xlabel('Date')
        # self.ax.set_ylabel(feature)
        if plot_group == 'Date' and (plot_type == 'Box' or plot_type == 'Violin'):
            # Display all dates on x-axis
            self.ax.set_xticks(ticks=np.arange(1,len(self.dates)+1), labels=self.dates, rotation=45, ha='right', rotation_mode='anchor')
        elif plot_group == 'Date' and (plot_type == 'Scatter' or plot_type == 'Line'):
            self.ax.set_xticks(ticks=np.arange(len(self.dates)), labels=self.dates, rotation=45, ha='right', rotation_mode='anchor')
        # self.ax.legend(bbox_to_anchor=(0., 1.06, 1., .105), loc='lower left',
        #             ncols=8, mode="expand", borderaxespad=0.)
        
        # Tight layout
        self.fig_stats.tight_layout()
        
        self.can_stats.draw()

    def update_plot_df(self):
        # Convert column to datetime
        self.plot_df['Date'] = pd.to_datetime(self.plot_df['Date'], format='%d-%b-%Y')
        self.info_df['Test Date'] = pd.to_datetime(self.info_df['Test Date'], format='%d-%b-%Y')

        # Sort by Animal ID, Date, Tray
        self.plot_df = self.plot_df.sort_values(by=['Date', 'Animal ID', 'Tray'])

        # Format datetimes to new format
        date_str = self.plot_df['Date'].dt.strftime('%d-%b-%Y')

        # Get list of unique dates
        self.dates = date_str.unique()

        # Create dictionary to map dates to day numbers
        date_dict = {date:i+1 for i, date in enumerate(self.dates)}
        # print(date_dict)

        # Apply mapping 
        self.plot_df['Day'] = date_str.map(date_dict)

        def extract_animal_group(animal_id):
            # Use regex to match one or more letters at the start of the string
            match = re.match(r'^([A-Za-z]+)', animal_id)
            if match:
                return match.group(1)
            else:
                return None  # or some default value

        # Apply the function to create the Animal Group column
        self.plot_df['Animal Group'] = self.plot_df['Animal ID'].apply(extract_animal_group)

        # Add Test Type to plot_df using info_df dates
        self.plot_df = self.plot_df.merge(
            self.info_df[['Test Date', 'Animal Group', 'Test Type', 'Test Phase', 'Tray Type', 'Injury Type']],
            left_on=['Date', 'Animal Group'],
            right_on=['Test Date', 'Animal Group'],
            how='left',
            suffixes=('_x', '_y')
        )

        # Drop the redundant 'Test Date' column from the merge
        # self.plot_df.drop(columns=['Tray Type_x'], inplace=True)

        self.plot_df = self.plot_df.rename(columns={'Test Type_y': 'Test Type', 
                                                    'Test Phase_y': 'Test Phase', 
                                                    'Tray Type_y': 'Tray Type',
                                                    'Injury Type_y': 'Injury Type'})

        # self.plot_df['Test Type'] = self.plot_df['Test Type'].str.replace(r'.*Train.*', 'Train', regex=True)    

        # self.plot_df = self.plot_df.set_index('Date')
        # self.info_df = self.info_df.set_index('Test Date')
        # self.plot_df = self.plot_df.join(self.info_df['Test Type'], how='left')

        # self.plot_df = self.plot_df.reset_index(drop=True)
        # self.info_df = self.info_df.reset_index(drop=True)

        # print(self.info_df.head(2))
        # print(self.plot_df.head(2))

    def display_df(self, df=None):
        try:
            if df is None:
                df = self.info_df
            # Clear table
            self.tblSummary.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            # Set the table widget dimensions
            self.tblSummary.setRowCount(num_rows)
            self.tblSummary.setColumnCount(num_cols)

            # Set the table headers by converting non-string columns to string
            col_names = [str(col) for col in df.columns.values]
            self.tblSummary.setHorizontalHeaderLabels(col_names)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    # Check if item is string or float
                    if df.iloc[row, col] is None or df.iloc[row, col] == 'nan':
                        str_item = ''
                    else:
                        str_item = f'{df.iloc[row, col]:.2f}' if isinstance(df.iloc[row, col], float) else f'{df.iloc[row, col]}'
                    item = QTableWidgetItem(str_item)
                    self.tblSummary.setItem(row, col, item)
            
            self.tblSummary.resizeColumnsToContents()
        except:
            print(traceback.format_exc())

    def filter_data(self, filter):
        for row in range(self.tblSummary.rowCount()):
            
            match = False
            
            for column in range(self.tblSummary.columnCount()):
            
                header = self.tblSummary.horizontalHeaderItem(column).text()
                item = self.tblSummary.item(row, column)
                
                if header in filter:
                    val_list = filter[header]
                    for val in val_list:
                        if val in item.text():
                            match = True
                            break
                    
            self.tblSummary.setRowHidden(row, not match)

    def get_info(self, file):
        info = file.split('_')
        # parse date and compare with info_df
        date = self.parse_date(info[0])
        # Check if date is in info_df, then get animal group, tray type, animal id, tray id
        # row = self.info_df[(self.info_df['Test Date'] == date) & (self.info_df['Animal #'] == int(info[1][1:]))]
        # print(info[1][1:])
        # print(row)

        # Regular expression to match the animal group
        pattern = r'_(\w+)\d{2}_'

        # Search for the pattern in the filename
        match = re.search(pattern, file)

        if match:
            animal_group = match.group(1)
            print(f"Animal group: {animal_group}")
        else:
            animal_group = ""
            print("Animal group not found")

        date = date.strftime("%B %d, %Y")

        return f'{date}, {animal_group}, Day:{info[1]}, Tray: {info[2]}'


        # return f'{row.iloc[0, 0].strftime("%B %d, %Y")}, {info[1]}, Day:{row.iloc[0, 1]}, Tray type: {row.iloc[0, 2]}, Tray #:{info[2][1:2]}'

    def parse_date(self, date_str):
        formats = ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d", "%d/%m/%Y") # list of formats to try

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                pass
        
        raise ValueError("No valid date format found")