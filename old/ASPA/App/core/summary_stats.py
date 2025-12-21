import os
import pandas as pd
from qtpy.QtWidgets import QApplication
import sys
from df_viewer import DataFrameViewer
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # Check if summary.xlsx exists
    if os.path.exists('summary.xlsx'):
        df = pd.read_excel('summary.xlsx')
        df.insert(0, 'Animal ID', df['Video'].str[9:12])
        df.insert(1, 'Date', df['Video'].str[:8])
        df.insert(3, 'Tray', df['Video'].str[-2:])

        # Remove Video column
        df = df.drop(columns=['Video'])

        # Sort by Animal ID, Date, Tray
        df = df.sort_values(by=['Animal ID', 'Date', 'Tray'])

        # Insert columns 'Eaten', 'Displaced' from info_df where Animal ID, Date, Tray match
        for i in range(20):
            animal_id = df['Animal ID'].iloc[i]
            date = df['Date'].iloc[i]
            tray = df['Tray'].iloc[i]
            tray_no_str = f'.{int(tray[-1])-1}' if int(tray[-1])-1 != 0 else ''

        # Reorder columns so that H Eaten and H Displaced are next to Eaten and Displaced
        cols = list(df.columns)
        cols = cols[:7] + cols[-2:] + cols[7:-2]
        df = df[cols]

        print(df.head())

        # app = QApplication(sys.argv)
        # window = DataFrameViewer(df)
        # window.show()
        # sys.exit(app.exec_())

        # # Line plot of total swipes over time
        # plt.figure()
        # plt.plot(df['Date'], df['Total swipes'])
        # plt.xlabel('Date')
        # plt.ylabel('Total swipes')
        # plt.title('Total Swipes Over Time')

        # # Bar plot of attention by tray  
        # plt.figure()
        # plt.bar(df['Tray'], df['Attention'])
        # plt.xlabel('Tray')
        # plt.ylabel('Attention (%)')
        # plt.title('Attention by Tray')

        # # Scatter plot of swipe speed vs swipe duration
        # plt.figure()
        # plt.scatter(df['Swipe speed'], df['Swipe duration'])
        # plt.xlabel('Swipe speed (mm/s)')  
        # plt.ylabel('Swipe duration (sec)')
        # plt.title('Swipe Speed vs Duration')

        # # Histogram of swipe area
        # plt.figure()
        # plt.hist(df['Swipe area'], bins=20)
        # plt.xlabel('Swipe area (mm^2)')
        # plt.ylabel('Frequency')
        # plt.title('Swipe Area Distribution')

        #plt.show()

        # Get list of unique dates
        dates = df['Date'].unique()

        # Sort dates
        dates.sort() 

        # Create dictionary to map dates to day numbers
        date_dict = {date:i+1 for i, date in enumerate(dates)}

        # Apply mapping 
        df['Day'] = df['Date'].map(date_dict)

        # Groupby Animal ID and Date, take mean Total Swipes
        df_swipe = df.groupby(['Animal ID', 'Date'])['Total swipes'].mean().reset_index()

        g = sns.FacetGrid(df_swipe, col='Animal ID', col_wrap=7)
        g.map(plt.plot, 'Date', 'Total swipes')

        plt.show()