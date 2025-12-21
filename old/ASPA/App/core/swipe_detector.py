from itertools import count
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from scipy.signal import find_peaks

class SwipeDetector:
    def __init__(self) -> None:
        self.init_swipe_detection_viewer()


    def init_swipe_detection_viewer(self):
        # create a figure
        self.fig_feat = plt.figure()
        self.fig_feat.patch.set_facecolor('lightgrey')
        self.can_feat = FigureCanvas(self.fig_feat)
        self.fig_feat.subplots_adjust(wspace=0.15, hspace=0.25,
                                       top=0.8, bottom=0.10, left=0.03, right=0.99)
        
    def update_title_info(self, summary):
        title = ''
        for key in summary.keys():
            if summary[key] is None or summary[key] == 'nan':
                val = ''
            else:
                val = f'{summary[key]:.2f}' if isinstance(summary[key], float) else f'{summary[key]}'
            title += f'{key}:{val}  '
        self.fig_feat.suptitle(title)
        self.can_feat.draw()
        
    def detect_swipes(self, df, threshold, interval, val_th, val_int, title=''):
        self.fig_feat.clear()

        # Max right hand y where right hand likelihood is greater than 0.9
        max_y = df[df['RightHand_likelihood'] > 0.5]['RightHand_y'].max()
        min_y = df[df['RightHand_likelihood'] > 0.5]['RightHand_y'].min()
        mid = (max_y + min_y) / 2

        print(f'Swipe threshold: ({max_y-threshold}, {max_y}), interval: {interval}')
        print(f'Valley threshold: ({-(max_y-val_th+5)}, {-(max_y-val_th-threshold)}), interval: {val_int}')

        x = df['bodyparts_coords'].values
        y = df['RightHand_y'].values
        
        peaks, _   = find_peaks(y,  height=(max_y-threshold, max_y),  distance=interval)
        valleys, _ = find_peaks(-y, height=(-(max_y-val_th+5), -(max_y-val_th-threshold)), distance=val_int)

        print(f'Swipe count: {len(peaks)}, valleys count: {len(valleys)}')
        # print(f'Swipes: {x[peaks]}')
        # print(f'Valleys: {x[valleys]}')

        # create an axis
        ax = self.fig_feat.add_subplot(111)
        
        ax.plot(x, y, label='Right Hand Y')
        ax.plot(x[peaks], y[peaks], 'x', label='Swipes', color='r')
        ax.plot(x[valleys], y[valleys], 'o', color='g')
        ax.legend()
        ax.set_title(f'Swipe detection (swipes:{len(peaks)}) {title}')
        self.can_feat.draw()

        self.detect_tray_movmt(df, mid=mid, ax=ax)

        return x[peaks], x[valleys]
    
    def detect_tray_movmt(self, df, mid=420, ax=None, std_frames=4, std_thresh=1):
        

        # Get frames where pellet is moving
        # df['rolling_std'] = df['Pellet_x'].rolling(std_frames).std()
        # df['motion'] = ((df['rolling_std'] > std_thresh) & 
        #                 (((df['Pellet_likelihood'] > 0.8) & (df['Pillar_likelihood'] < 0.5)) | 
        #                 ((df['Pillar_likelihood'] > 0.8) & (df['Pellet_likelihood'] < 0.5))) & 
        #                 (df['RightHand_likelihood'] < 0.2)).astype(int)

        # print(df['rolling_std'])

        # print(df['motion'].value_counts())

        # Plot motion markers
        x = df['bodyparts_coords'].values
        y = df['motion'].values
        mask = y == 1
        x = x[mask]
        y = y[mask] * mid

        peaks, _ = find_peaks(y,  height=0.99,  distance=3)
        # valleys, _ = find_peaks(-y, height=-.99, distance=3)

        # Quantization
        # Set all values between 0.9 and 0.99 to 1 and rest to zero
        # yq = np.where((y > 0.98) | (y < 0.9), 0, 1)

        # y = np.array([0.1, 0.5, 0.99]) 
        # yq= np.where((y > 0.98) | (y < 0.9), 0, 1)

        if ax is None:
            self.fig_feat.clear()
            ax = self.fig_feat.add_subplot(111)
            ax.set_title(f'Tray motion detection')
        ax.scatter(x, y, label='Tray motion', marker='v', color='k')
        
        # ax.plot(x, y, label='Pellet likelihood')
        # ax.plot(x, yq, label='Pellet likelihood Quantized', color='g')
        # ax.plot(x[peaks], y[peaks], 'x', label='Detection', color='gray')
        # ax.plot(x[valleys], y[valleys], 'o', color='g', label='Motion')
        ax.legend()
        
        self.can_feat.draw()
