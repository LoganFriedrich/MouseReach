
from datetime import timedelta, datetime
import os
import sys
import threading
import time
import traceback
import pandas as pd

from video_player import VideoPlayer
from frame_info import FrameInfo
from swipe_info import SwipeInfo
from stats import Stats
from action_detector import ActionDetector
from df_viewer import DataFrameViewer

from qtpy.QtWidgets import QApplication

import concurrent.futures



class FrameInfoProcessor:
    def __init__(self, tblFrameInfo, tblSwipeInfo, reachFilter, tn_count=6, ui=True) -> None:        
        self.thumbnails = []
        self.highlight = None
        self.play = False
        self.playback_speed = 0.1
        self.init = False
        self.play_swipes_only = False
        self.has_ui = ui

        self.init_frame_info(tbl=tblFrameInfo)
        self.init_swipe_info(tbl=tblSwipeInfo, reachFilter=reachFilter)
        self.init_vdo_player(tn_count=tn_count)
        self.init_stats()
        self.init_swipe_detector()

        self.init = True

    #region Stats

    def init_stats(self):
        self.stats = Stats(ui=self.has_ui)

    def get_info(self, vdo_filename):
        return self.stats.get_info(vdo_filename)

    #endregion Stats

    #region FrameInfo

    def display_trackbar(self):
        self.vdo_player.display_trackbar(num_frames=self.frame_info.num_rows)

    def init_frame_info(self, tbl):
        self.frame_info = FrameInfo(tbl=tbl, ui=self.has_ui)

        if self.has_ui:
            self.frame_info.tblFrameInfo.itemSelectionChanged.connect(self.tblFrameInfo_itemSelectionChanged)

    def tblFrameInfo_itemSelectionChanged(self):
        try:
            if not self.init:
                return
            # Get the current selected row
            row = self.frame_info.tblFrameInfo.currentRow()

            self.vdo_player.current_frame_idx = int(float(self.frame_info.tblFrameInfo.item(row, 0).text()))
            
            self.vdo_player.update_vdo_display()
            self.vdo_player.can_vp.draw()
        except:
            print(traceback.format_exc())

    #endregion FrameInfo

    #region SwipeInfo

    def init_swipe_info(self, tbl, reachFilter):
        self.swipe_info = SwipeInfo(tbl=tbl, reachFilter=reachFilter, ui=self.has_ui)

        if self.has_ui:
            self.swipe_info.tblSwipeInfo.itemSelectionChanged.connect(self.tblSwipeInfo_itemSelectionChanged)

    def tblSwipeInfo_itemSelectionChanged(self):
        try:
            if not self.init:
                return
            # Get the current selected row
            row = self.swipe_info.tblSwipeInfo.currentRow()
            self.vdo_player.current_frame_idx = int(float(self.swipe_info.tblSwipeInfo.item(row, 0).text()))
            self.vdo_player.update_vdo_display()
            self.vdo_player.can_vp.draw()

            # Highlight swipe in swipe detection viewer
            self.action_detector.highlight_swipe(x=self.vdo_player.current_frame_idx, y=int(float(self.swipe_info.tblSwipeInfo.item(row, 20).text())))
        except:
            print(traceback.format_exc())

    #endregion SwipeInfo

    #region Video Player

    def init_vdo_player(self, tn_count=6):
        if not self.has_ui:
            return
        self.vdo_player = VideoPlayer(tn_count=tn_count)

        # Connect the click event to the update_main_view function
        self.vdo_player.fig_vp.canvas.mpl_connect('button_press_event', self.thumbnail_on_click)

        # Connect the click event on the navigation buttons to the on_click function
        self.vdo_player.btn_prev.on_clicked(self.nav_on_click)
        self.vdo_player.btn_next.on_clicked(self.nav_on_click)
        self.vdo_player.btn_play.on_clicked(self.play_on_click)

    def play_on_click(self, event):
        self.vdo_player.play = not self.vdo_player.play
        if self.vdo_player.play:
            self.vdo_player.btn_play.label.set_text('Pause')
            self.vdo_player.vdo_thread.start()
        else:
            self.vdo_player.vdo_thread.join()
            self.vdo_player.btn_play.label.set_text('Play')
            self.vdo_player.vdo_thread = threading.Thread(target=self.vdo_player.play_vdo)

    #region Thumbnail

    def nav_on_click(self, event):

        self.vdo_player.nav_on_click(event)

        self.frame_info.tblFrameInfo.itemSelectionChanged.disconnect(self.tblFrameInfo_itemSelectionChanged)
        self.vdo_player.pb_status = self.frame_info.select_row(self.vdo_player.current_frame_idx)
        self.frame_info.tblFrameInfo.itemSelectionChanged.connect(self.tblFrameInfo_itemSelectionChanged)

    # Function to update the main view
    def thumbnail_on_click(self, event):

        self.vdo_player.thumbnail_on_click(event)

        self.frame_info.tblFrameInfo.itemSelectionChanged.disconnect(self.tblFrameInfo_itemSelectionChanged)
        self.vdo_player.pb_status = self.frame_info.select_row(self.vdo_player.current_frame_idx)
        self.frame_info.tblFrameInfo.itemSelectionChanged.connect(self.tblFrameInfo_itemSelectionChanged)

    #endregion Thumbnail

    #endregion Video Player

    #region Tray Motion

    def detect_tray_motion(self, start, end):
        # Get rows that have body_coords between start and end
        self.tray_frames = self.frame_info.df[(self.frame_info.df['bodyparts_coords'] >= start) &
                                              (self.frame_info.df['bodyparts_coords'] <= end)]
        
        self.action_detector.detect_tray_movmt(df=self.tray_frames)

    #endregion Tray Motion

    #region Swipe Detector

    def init_swipe_detector(self):
        self.action_detector = ActionDetector(ui=self.has_ui)

        if self.has_ui:
            self.action_detector.can_feat.mpl_connect('button_press_event', self.detection_on_click)
            self.action_detector.can_feat.mpl_connect('pick_event', self.detection_on_pick)
            self.action_detector.can_feat.mpl_connect('pick_event', self.detection_on_unpick)

    def detection_on_pick(self, event):
        line = event.artist
        ind = event.ind[0]
        data = line.get_data()
        xdata, ydata = data
        x = xdata[ind]
        y = ydata[ind]

        # Increase size and make semitransparent
        line.set_data(xdata, ydata) 
        line.set_markersize(20)
        line.set_alpha(0.5)
        self.action_detector.fig_feat.canvas.draw_idle()

    def detection_on_unpick(self, event):
        line = event.artist
        data = line.get_data()
        line.set_data(data)
        line.set_markersize(8) 
        line.set_alpha(1)
        self.action_detector.fig_feat.canvas.draw_idle()

    def detection_on_click(self, event):
        x = event.xdata # x data coordinate of click
        y = event.ydata # y data coordinate of click
        print(f'Click at ({x}, {y})')
        if x is None:
            return
        self.vdo_player.current_frame_idx = int(x)

        self.thumbnail_on_click(event)

    # Function to filter the frames where the mouse is swiping
    def detect_swipes(self, start, end, threshold=50, rh_y_conf=0.8, std_motion=1, title=''):
        # Get reference y
        ref_y = int(self.frame_info.df['Reference_y'].iloc[start])
        ref_x = int(self.frame_info.df['Reference_x'].iloc[start])

        # Get rows that have body_coords between start and end
        self.swipe_frames = self.frame_info.df[(self.frame_info.df['bodyparts_coords'] >= start) &
                                              (self.frame_info.df['bodyparts_coords'] <= end)]
        
        self.action_detector.detect_tray_movmt(df=self.swipe_frames, title=title)
        
        # xlim for swipe detector
        start = self.swipe_frames['bodyparts_coords'].iloc[0]
        end = self.swipe_frames['bodyparts_coords'].iloc[-1]
        
        # Print average right hand likelihood
        print(f"Average RH likelihood({self.swipe_frames.shape[0]}): {self.swipe_frames['RightHand_likelihood'].mean()}")
        print(f'No. of frames with RH likelihood > 0.9: {self.swipe_frames[self.swipe_frames["RightHand_likelihood"] > 0.9].shape[0]}/{self.swipe_frames.shape[0]}')

        # Get rows where right hand likelihood is greater than 0.9 and right hand y is greater than nose y
        self.swipe_frames = self.swipe_frames[(self.swipe_frames['Nose_y'] > ref_y-60) &
                                              (self.swipe_frames['Nose_x'] > ref_x+70) &
                                              (self.swipe_frames['Nose_x'] < ref_x+100) &
                                              (self.swipe_frames['Nose_likelihood'] > 0.5) &
                                              (self.swipe_frames['RightHand_likelihood'] > 0.1)&
                                              (self.swipe_frames['RightHand_y'] - self.swipe_frames['Nose_y'] > 8)]
        
        # Print average right hand likelihood
        print(f"Average RH likelihood after filter({self.swipe_frames.shape[0]}): {self.swipe_frames['RightHand_likelihood'].mean()}")
        print(f'No. of frames with RH likelihood > 0.9: {self.swipe_frames[self.swipe_frames["RightHand_likelihood"] > 0.9].shape[0]}/{self.swipe_frames.shape[0]}')
        
        self.swipe_seq_dict, y_min, y_max = self.action_detector.detect_swipes_new(df=self.swipe_frames, threshold=threshold, rh_y_conf=rh_y_conf, std_motion=std_motion, title=title)
                
        if len(self.swipe_seq_dict) == 0:
            print('No swipes detected')
            return None
        
        # Display all swipes in swipe info tab
        swipe_df = self.swipe_frames[self.swipe_frames['bodyparts_coords'].isin(self.swipe_seq_dict.keys())]
        swipe_df = self.action_detector.add_pellet_no(swipe_df)
        self.swipe_info.df = swipe_df
        self.swipe_info.y_min = y_min
        self.swipe_info.y_max = y_max
        self.swipe_info.tot_frames = end - start
        self.swipe_info.attention_frames = self.frame_info.df[(self.frame_info.df['Nose_likelihood'] > 0.9)& (self.frame_info.df['Nose_y'] > y_min - 50)].shape[0]

        summary = self.analyse_swipes_new()

        if self.has_ui:
            self.action_detector.ax.set_xlim(start-100, end+100)
            self.action_detector.ax.set_ylim(bottom=y_min, top=y_max+10)
            self.action_detector.update_title_info(summary)

        return summary

    def analyse_swipes_new(self):
        # Get all rows that have a swipe between two valleys
        row_indices = []
        for seq in self.swipe_seq_dict.values():
            row_indices.extend(seq.index)

        print(f"Number of rows: {len(row_indices)}")

        # Get missed swipes
        swipes_idx_list = list(self.swipe_seq_dict)
        missed_swipes = [item for item in swipes_idx_list if item not in self.swipe_seq_dict.keys()]
        print(f"Missed swipes: {missed_swipes}")

        self.swipe_frames = self.swipe_frames.loc[row_indices]

        self.swipe_frames = self.action_detector.add_pellet_no(self.swipe_frames)

        if self.has_ui:
            self.vdo_player.swipe_frames = self.swipe_frames
            self.frame_info.display_df(self.swipe_frames)

        if len(missed_swipes) > 0:
            print("Skipping analysis")
            # summary = self.swipe_info.analyse_swipe_sequences(swipe_seq_list)
            summary = None
            return summary

        # if self.has_ui:
        #     self.swipe_info.analyse_swipe_sequences_thread(swipe_seq_list)
        # else:
        if len(self.swipe_seq_dict) == 0:
            return None
        summary = self.swipe_info.analyse_swipe_sequences(self.swipe_seq_dict)
            
        return summary

    #endregion Swipe Detector


def convert_timedelta(tdelta):
    days, seconds = tdelta.days, tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return days, hours, minutes, seconds

def process_video(file_path):
    title = file_path.split('/')[-1][:15]
    csv_file = file_path.replace('.mp4', 'DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')

    try:
    
        frame_proc = FrameInfoProcessor(tblFrameInfo=None,
                                        tblSwipeInfo=None,
                                        reachFilter=None,
                                        tn_count=10, ui=False)
        frame_proc.frame_info.read_csv(csv_file)
        frame_proc.stats.df = frame_proc.frame_info.df

        summary = {'Date': datetime.strptime(title[:8], '%Y%m%d').strftime('%d-%b-%Y'),
                'Animal ID': title[9:12],
                'Tray': title[-2:]}
        
        swipe_summary = frame_proc.detect_swipes(start=0, end=380000,  
                                        threshold=50, 
                                        rh_y_conf=0.8,
                                        std_motion=1,
                                        title=title)
                                        
        if swipe_summary is not None:
            summary.update(swipe_summary)
        else:
            summary = None
            print(f'No swipes detected for {title}')
    except:
        # write to log file
        log_file.write(f'Error processing {title}\n')
        log_file.write(traceback.format_exc())
        print(traceback.format_exc())
        summary = None

    return summary

def update_summary_with_manual_recording(summary_file, recorded_file):
    # Check if summary.xlsx exists
    if os.path.exists(summary_file):
        df = pd.read_excel(summary_file)
        info_df = pd.pd.read_excel(recorded_file, sheet_name='1 - ENTER DATA HERE') # usecols='A:G'

        # Sort by Animal ID, Date, Tray
        df = df.sort_values(by=['Animal ID', 'Date', 'Tray'])

        # Insert columns 'Eaten', 'Displaced' from info_df where Animal ID, Date, Tray match
        df['H Eaten'] = None
        df['H Displaced'] = None
        for i in range(len(df)):
            animal_id = df['Animal ID'].iloc[i]
            date = df['Date'].iloc[i]
            tray = df['Tray'].iloc[i]
            tray_no_str = f'.{int(tray[-1])-1}' if int(tray[-1])-1 != 0 else ''
            eaten = info_df[(info_df['Animal #'] == int(animal_id[-2:])) & (info_df['Test Date'] == stats.parse_date(date))][f'Eaten{tray_no_str}'].iloc[0]
            displaced = info_df[(info_df['Animal #'] == int(animal_id[-2:])) & (info_df['Test Date'] == date)][f'Displaced{tray_no_str}'].iloc[0]
            df['H Eaten'].iloc[i] = eaten
            df['H Displaced'].iloc[i] = displaced

        # Reorder columns so that H Eaten and H Displaced are next to Eaten and Displaced
        cols = list(df.columns)
        cols = cols[:7] + cols[-2:] + cols[7:-2]
        df = df[cols]

        app = QApplication(sys.argv)
        window = DataFrameViewer(df)
        window.show()
        sys.exit(app.exec_())

def generate_group_dlc_summary(folder_path):
    # Dataframe to store summary of all videos 
    df_summary = pd.DataFrame()

    # Get list of all video files
    files = [os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) if f.endswith('DLC_resnet50_MSA - V1Mar3shuffle1_1030000.csv')]
    
    print(f'Number of files: {len(files)}')

    # Run using concurrent.futures
    start_time = time.time()
    print(f'Running all experiments with {os.cpu_count()-1} processes', flush=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_video, files)
        
    for result in results:
        if result is not None:
            df_summary = df_summary._append(result, ignore_index=True)

    d, h, m, s = convert_timedelta(timedelta(seconds=(time.time() - start_time)))
    print(f'Finished all experiments in {h}h, {m}m, {s}s', flush=True)

    # Write summary dataframe to excel, use parent folder name as filename
    group = folder_path.split('\\')[-1]
    df_summary.to_excel(os.path.join(folder_path, f'summary_{group}.xlsx'), index=False)

if __name__ == '__main__':
    folder_path = r'D:\! DLC Output\Analyzed'
    log_file = open(f'log_{time.asctime()}.txt', 'w')

    # Go to Single Animal folder for every group using os.walk
    for root, dirs, _ in os.walk(folder_path):
        # Check if Single_Animal folder exists
        if 'Single_Animal' in dirs:
            generate_group_dlc_summary(folder_path=os.path.join(root, 'Single_Animal'))

    log_file.close()


    
        
    




