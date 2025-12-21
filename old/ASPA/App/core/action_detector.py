from cProfile import label
from itertools import count
from tracemalloc import start
from turtle import back, color
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from pyparsing import col
from matplotlib.ticker import FuncFormatter

from scipy.signal import find_peaks

import pandas as pd

def format_fn(tick_val, tick_pos):
    if int(tick_val) >=1000:
        return '%1.fk' % int(tick_val/1000)
    else:
        return str(int(tick_val))

class ActionDetector:
    def __init__(self, ui=True) -> None:
        self.title = ''
        self.df_full = None
        self.has_ui = ui
        self.pellet_x_median = 0
        self.pellet_y_median = 0
        self.pellet_pos_th = 4
        if ui:
            self.init_swipe_detection_viewer()

    def init_swipe_detection_viewer(self):
        # create a figure
        self.fig_feat = plt.figure()
        self.fig_feat.patch.set_facecolor('lightgrey')
        self.can_feat = FigureCanvas(self.fig_feat)
        self.ax = self.fig_feat.add_subplot(111)
        self.fig_feat.subplots_adjust(wspace=0.15, hspace=0.25,
                                       top=0.8, bottom=0.10, left=0.03, right=0.99)
        
        self.mid = 420
        
    def update_title_info(self, summary):
        if summary is None:
            print('Summary is None')
            return
        title = ''
        for key in summary.keys():
            if key == '1':# or key == 'P11':
                title += '\n'
            if summary[key] is None or summary[key] == 'nan':
                val = ''
            else:
                val = f'{summary[key]:.2f}' if isinstance(summary[key], float) else f'{summary[key]}'
            title += f'{key}:{val} '
        self.fig_feat.suptitle(title, fontsize=10)
        self.can_feat.draw()
     
    def distance(self, x1, y1, x2, y2):
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance

    def detect_tray_movmt(self, df, title):
        # Plot scatter plot of pellet y where likelihood is greater than 0.9
        x = df['bodyparts_coords'].values
        self.mid = df[df['RightHand_likelihood'] > 0.9]['RightHand_y'].max() - 20
        # p_y = df[df['Pellet_likelihood'] > 0.5]['Pellet_y'].values
        # py_baseline = np.median(p_y)
        p_x = df[df['Pellet_likelihood'] > 0.5]['bodyparts_coords'].values
        p_y = df[df['Pellet_likelihood'] > 0.5]['Pellet_x'].values
        # pillar_likelihood = df[df['Pellet_likelihood'] > 0.95]['Pillar_likelihood'].values
        # if pillar_likelihood.mean() > 0.3:
        #     pellet_df = df[(df['Pellet_likelihood'] > 0.95) & (df['Pellet_y']-df['RightHand_y'] > 8)]
        # else:
        pellet_df = df[(df['Pellet_likelihood'] > 0.95) & (df['Pellet_y']-df['RightHand_y'] > 8) & (df['Pillar_likelihood'] < 0.1)]
        x = pellet_df['bodyparts_coords'].values
        # y = pellet_df['Pellet_x'].values
        # Check pellet motion in x using rolling std
        motion = pellet_df['Pellet_x'].rolling(20).std()
        # Get frames there is no motion
        no_motion_frames = np.where(motion < 1)[0]
        no_motion_in_x_df = pellet_df.iloc[no_motion_frames]
        y = no_motion_in_x_df['Pellet_x'].values
        baseline = np.median(y)
        # Set low likelihood values to baseline
        y = np.where((pellet_df['Pellet_likelihood'] > 0.5) & (pellet_df['Pillar_likelihood'] < 0.1), pellet_df['Pellet_x'], baseline)
        # y = np.where((df['Pellet_likelihood'] > 0.5) & (df['Pillar_likelihood'] < 0.1), df['Pellet_x'], baseline)
        
        # Plot curve
        if self.has_ui:
            self.ax.clear()
            self.ax.plot(x, y-baseline+self.mid, label='Pellet motion', color='red')
            self.ax.plot(x, motion+self.mid, label='Pellet motion', color='green')

        pellet_x = df[(df['Pellet_likelihood'] > 0.9) & (df['Pillar_likelihood'] < 0.5)]['Pellet_x'].values
        if len(pellet_x) == 0:
            print(f'No pellet detected in {title}')
            return
        max_y = pellet_x.max()        

        peaks, _ = find_peaks(motion, height=2)
        # Add first frame to peaks
        peaks = np.insert(peaks, 0, 0)
        
        # Check x[peaks] and x[peaks+1] for distance between peaks
        valid_peaks = []
        for i in range(len(peaks) - 1):
            if x[peaks[i + 1]] - x[peaks[i]] >= 1000:
                valid_peaks.append(peaks[i])
        # Add the last peak if it wasn't added
        if len(peaks) > 0:
            valid_peaks.append(peaks[-1])

        peaks = np.array(valid_peaks)
        print(f'Pellet count: {len(peaks)} in {title}')
        # peaks, _   = find_peaks(y,  height=(baseline-5, max_y),  distance=1800)
        # print(f'Pellet count: {len(peaks)} in {title}')

        if self.has_ui:
            y = np.repeat(self.mid, len(x))
            # self.ax.clear()
            self.ax.scatter(x[peaks], y[peaks], label='Tray motion', color='gray', marker='v')
        
        pellet_num = 1
        self.pellet_loc_dict = {}
        self.pellet_eaten_prob = {}
        self.pellet_loc_df = pd.DataFrame()
        pell_df = pd.DataFrame()
        for peak in peaks:
            if self.has_ui:
                self.ax.text(x[peak], self.mid+5, f'P{pellet_num}', fontsize=10, color='b', ha='center', va='bottom')
            if pellet_num > 1:
                pellet_df = df[(df['bodyparts_coords'] > prev_frame) 
                                     & (df['bodyparts_coords'] < x[peak]) 
                                     & (df['Pellet_y']-df['RightHand_y'] > 8)]
                pellet_likelihood = pellet_df['Pellet_likelihood'].values
                pillar_likelihood = pellet_df['Pillar_likelihood'].values
                
                # Get frames where pellet likelihood is low
                # if pellet_likelihood.mean() < 0.5:
                #     low_likelihood_frames = np.where(pellet_likelihood < 0.5)[0]
                #     print(f'Low likelihood frames: {low_likelihood_frames}')
                pell_df = no_motion_in_x_df[(no_motion_in_x_df['bodyparts_coords'] > prev_frame) &
                                            (no_motion_in_x_df['bodyparts_coords'] < prev_frame+75)]
                if len(pell_df) > 0 and pellet_likelihood.mean() > 0.5:
                    med_x = pell_df['Pellet_x'].median()
                    med_y = pell_df['Pellet_y'].median()
                    self.pellet_loc_dict[pellet_num-1] = f'({pell_df.iloc[0]["Pellet_x"]:.2f},{pell_df.iloc[0]["Pellet_y"]:.2f}), med:({med_x:.2f}, {med_y:.2f})'                    
                    row_data = pd.DataFrame({'pell_num':[pellet_num-1], 'start':prev_frame, 'end':x[peak], 'frames':len(pell_df),
                                                    'pell_x': med_x, 'pell_y': med_y})
                    
                    self.pellet_loc_df = pd.concat([self.pellet_loc_df, row_data])
                elif len(pellet_df) > 0 and pillar_likelihood.mean() > 0.8:
                    med_x = pellet_df['Pillar_x'].median()
                    med_y = pellet_df['Pillar_y'].median()
                    self.pellet_loc_dict[pellet_num-1] = f'({pellet_df.iloc[0]["Pellet_x"]:.2f},{pellet_df.iloc[0]["Pellet_y"]:.2f}), med:({med_x:.2f}, {med_y:.2f})'
                    row_data = pd.DataFrame({'pell_num':[pellet_num-1], 'start':prev_frame, 'end':x[peak], 'frames':len(pellet_df),
                                                    'pell_x': med_x, 'pell_y': med_y})
                    self.pellet_loc_df = pd.concat([self.pellet_loc_df, row_data])
                
                pellet_eaten_prob = pellet_likelihood.mean()
                # self.pellet_eaten_prob[pellet_num-1] = f'frames:{prev_frame}-{x[peak]}({len(pellet_likelihood)}), prob:{pellet_eaten_prob}'
                self.pellet_eaten_prob[pellet_num-1] = dict(start=prev_frame, end=x[peak], frames=len(pellet_likelihood), prob=pellet_eaten_prob)
                if self.has_ui:
                    self.ax.text((prev_frame + x[peak])//2, self.mid-5, f'{pellet_eaten_prob:.2f}', fontsize=10, color='r', ha='center', va='top')
                    if len(pell_df) > 0:
                        # Median pellet location
                        self.ax.text((prev_frame + x[peak])//2, self.mid-10, f'({med_x:.0f}, {med_y:.0f})', fontsize=7, color='k', ha='center', va='top')
            prev_frame = x[peak]
            pellet_num += 1

        if self.pellet_loc_df.empty:
            print(f'No pellet detected in {title}')
            with open('pellet_loc.txt', 'a') as f:
                f.write(f'{title}:\n')
                f.write('No pellet detected\n\n')
            # if not pell_df.empty:
            #     self.pellet_x_median = pell_df['Pellet_x'].median()
            #     self.pellet_y_median = pell_df['Pellet_y'].median()
            # else:
            #     self.pellet_x_median = 0
            #     self.pellet_y_median = 0
            return
        self.pellet_x_median = self.pellet_loc_df['pell_x'].median()
        self.pellet_y_median = self.pellet_loc_df['pell_y'].median()

        # Use pellet_x_median and pellet_y_median for outliers in self.pellet_loc_df
        self.pellet_loc_df = self.pellet_loc_df[(np.abs(self.pellet_loc_df['pell_x']-self.pellet_x_median) < 15) &
                                                (np.abs(self.pellet_loc_df['pell_y']-self.pellet_y_median) < 8)]
        self.pellet_loc_df = self.pellet_loc_df.reset_index(drop=True)

        self.pellet_x_median = self.pellet_loc_df['pell_x'].median()
        self.pellet_y_median = self.pellet_loc_df['pell_y'].median()

        # Write pellet loc dict to file
        with open('pellet_loc.txt', 'w') as f:
            f.write(f'{title}:\n')
            for key, value in self.pellet_loc_dict.items():
                f.write(f'P{key}:{value}\n')
            f.write('\n')
        
        if self.has_ui:
            self.ax.legend()
            self.can_feat.draw()

    def detect_swipes_new(self, df, threshold, rh_y_conf, min_frames=4, min_frames_std=8, th_adjacent=2, std_motion=0.8, title=''):
        self.title = title
        y_min = df[df['RightHand_likelihood'] > 0.95]['RightHand_y'].min()
        y_max = df[df['RightHand_likelihood'] > 0.95]['RightHand_y'].max() 
        y_thr = y_max - threshold
        # print(f'RightHand Y min: {y_min}, Y max: {y_max}, Range: {y_max-y_min}, Threshold: {threshold}')

        # Draw a horizontal line at y_thr
        if self.has_ui:
            self.ax.hlines(y_thr, df['bodyparts_coords'].min(), df['bodyparts_coords'].max(), alpha=0.2, color='r', label='Threshold')
            self.ax.hlines(y_max, df['bodyparts_coords'].min(), df['bodyparts_coords'].max(), alpha=0.2, color='g', label='Max RH y')
            # self.ax.hlines(y_min, df['bodyparts_coords'].min(), df['bodyparts_coords'].max(), alpha=0.2, color='b', label='Min')
        df_all = df
        df = df[df['RightHand_y'] > y_thr]
        df = df[df['RightHand_likelihood'] > rh_y_conf]

        swipe_idxs = df['bodyparts_coords'].values.tolist()

        swipe_seq_dict = {}

        if len(swipe_idxs) == 0:
            return swipe_seq_dict, y_min, y_max
        
        group = [swipe_idxs[0]] 
        b_motion = True

        for i in range(1, len(swipe_idxs)):
            if swipe_idxs[i] - swipe_idxs[i-1] < th_adjacent and b_motion:
                group.append(swipe_idxs[i])
                # Check if there 3 or more consecutive frames & if there is motion
                if len(group) >= min_frames_std:
                    std_frames = df[df['bodyparts_coords'].isin(group[-min_frames_std:])]
                    # check std deviation of y to see if there is motion
                    if std_frames['RightHand_y'].std() < std_motion or \
                       std_frames['RightHand_x'].std() < std_motion:
                        b_motion = False
                    else:
                        b_motion = True
            else:
                b_motion = True
                if len(group) < min_frames:
                    group = [swipe_idxs[i]]
                    continue
                df_gp = df[df['bodyparts_coords'].isin(group)]
                if df_gp['RightHand_y'].std() > std_motion or \
                   df_gp['RightHand_x'].std() > std_motion:
                    self.add_gp_to_dict(df_all, group, swipe_seq_dict, y_max, threshold)
                group = [swipe_idxs[i]]

        if len(group) >= min_frames :
            self.add_gp_to_dict(df_all, group, swipe_seq_dict, y_max, threshold)

        if self.has_ui:
            # self.ax.clear() Already cleared in detect_tray_movmt

            # colors for each group
            colors = [color for color in plt.cm.tab20(np.linspace(0, 1, len(swipe_seq_dict)))]
            color_idx = 0
            for key, value in swipe_seq_dict.items():
                g_x = value['bodyparts_coords'].values.tolist()
                g_y = value['RightHand_y'].values.tolist()

                # Plot the group with a different color
                self.ax.vlines(g_x, y_min, g_y, alpha=0.2, color=colors[color_idx])
                val = value[value['bodyparts_coords'] == key]['RightHand_y']
                self.ax.scatter(key, val, marker='x', color=colors[color_idx])
                # Get reach outcome
                outcome = self.determine_reach_outcome(value)
                if outcome:
                    # Add text above the x marker for swipe outcome
                    self.ax.text(key, val+5, f'{outcome}', fontsize=10, color='g', ha='center', va='bottom')
                color_idx += 1

            self.ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
            self.ax.legend()
            self.ax.set_title(f'{title}', fontsize=10)
            self.can_feat.draw()

        return swipe_seq_dict, y_min, y_max

    def add_gp_to_dict(self, df, group, dict, y_max, threshold):
        # Add three frames at the beginning and end in the group
        # group = self.modify_list(group, df)
        g_df = df[df['bodyparts_coords'].isin(group)]
        # Get index of max y value
        g_max = g_df['RightHand_y'].max()
        if y_max - g_max < threshold:
            idx = g_df[g_df['RightHand_y'] == g_max]['bodyparts_coords'].values[0]
            
            dict[idx] = g_df

    def modify_list(self, group, df):
        # Check if hand is close to pellet or pillar
        # df_begin = df[df['bodyparts_coords'].isin(group[0:3])]
        # begin_pellet_df = df_begin[(np.abs(df_begin['Pellet_y']-df_begin['RightHand_y']) > 4) | (np.abs(df_begin['Pillar_y']-df_begin['RightHand_y']) > 4)]
        # if len(begin_pellet_df) < 3:
            # Add three elements at the beginning
        start = group[0] - 6
        group = list(range(start, group[0])) + group

        # df_end = df[df['bodyparts_coords'].isin(group[-3:])]
        # end_pellet_df = df_end[(np.abs(df_end['Pellet_y']-df_end['RightHand_y']) > 4) | (np.abs(df_end['Pillar_y']-df_end['RightHand_y']) > 4)]
        # if len(end_pellet_df) < 3:
            # Add three elements at the end
        end = group[-1] + 7
        group.extend(range(group[-1]+1, end))

        return group

    def add_pellet_no(self,df):
        try:
            for key, value in self.pellet_eaten_prob.items():
                df.loc[(df['bodyparts_coords'] > value['start']) & (df['bodyparts_coords'] < value['end']), 'Pellet #'] = key
            
            # Move the column to the front of the dataframe
            cols = df.columns.tolist()
            cols.insert(1, cols.pop(cols.index('Pellet #')))
            df = df.reindex(columns=cols)
        except:
            print(self.title)

        return df

    def highlight_swipe(self, x, y):
        if hasattr(self, 'highlight'):
            self.highlight.remove()
        self.highlight = patches.Rectangle((x, y), 20, 10, fill=False, edgecolor='green', linewidth=2)
        self.ax.add_artist(self.highlight)
        
        self.can_feat.draw()

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
            group = group + list(range(group[-1]+1, group[-1]+7))
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
            pell_on_pill_x = self.pellet_loc_df[(self.pellet_loc_df['start'] <= group[0]) & (self.pellet_loc_df['end'] >= group[-1])]['pell_x'].values[0]
            pell_on_pill_y = self.pellet_loc_df[(self.pellet_loc_df['start'] <= group[0]) & (self.pellet_loc_df['end'] >= group[-1])]['pell_y'].values[0]
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
        b_pellet_moved = (pellet_y_std.mean() > 1.5) or (pellet_x_std.mean() > 1.5) or (pellet_displacement > 1.5)

        if begin_pellet_likelihood < 0.5 and avg_pellet_likelihood < 0.5:
            return 0 # "no pellet"
        elif b_pellet_moved and end_pellet_likelihood > 0.5 and b_pellet_on_pillar_begin:
            return 1 # "pellet displaced"
        elif begin_pellet_likelihood > 0.5 and end_pellet_likelihood < 0.5 and b_pellet_on_pillar_begin:
            return 2 # "swipe successful"
        elif b_pellet_on_pillar_begin and b_pellet_on_pillar_end:
            return 3 # f"swipe missed (on pillar) {pellet_displacement:.2f} {beg_distance:.2f} {distance:.2f}"
        else:
            return 4 # f"swipe missed {pellet_displacement:.2f} {beg_distance:.2f} {distance:.2f}"
