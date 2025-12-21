from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt



class Progressbar():
    def __init__(self):

        self.pb_left, self.pb_width    = 0, 1
        self.pb_bottom, self.pb_height = 0, 1
        self.pb_right = self.pb_left + self.pb_width
        self.pb_top   = self.pb_bottom + self.pb_height

        self.pb_perc_offset = 1.0
        self.fig_pb = plt.figure()
        self.fig_pb.patch.set_facecolor('lightgrey')
        self.can_pb = FigureCanvas(self.fig_pb)
        
        # self.pb_bg = np.ones((self.pb_h, self.pb_w))
        self.pb_status = "0/0 completed"
        self.pb_perc = 0
        self.fig_pb.subplots_adjust(wspace=0.0, hspace=0.05, 
                                    top=0.5, bottom=0, left=0, right=1)
        self.pb_ax = self.fig_pb.add_subplot(1,1,1)
        self.create_progbar_rect()
        

    def create_progbar_rect(self):
        # percentage text
        self.pb_ax.text(0.5 * self.pb_right, self.pb_bottom, "{0:.2f} %".format(self.pb_perc), fontdict={'family':'Arial', 'color':'black', 
                                                    'weight': 'bold','size': '10', 'ha':'center', 
                                                    'va':'bottom'}, transform=self.pb_ax.transAxes)
        # Draw a rectangle in figure coordinates ((0, 0) is bottom left and (1, 1) is 
        # # upper right).
        pbar = Rectangle((self.pb_left, self.pb_bottom), width=self.pb_perc * (self.pb_width/100), 
                          height=self.pb_height, transform=self.pb_ax.transAxes, facecolor='green', linewidth=0)
        self.pb_ax.add_patch(pbar)
        self.pb_ax.set_title(self.pb_status, loc='left', fontdict={"family":"Arial", "size":'10'})
        self.pb_ax.set_axis_off()


    def update_progressbar(self, perc=0, status=None):
        if None != status:
            self.pb_perc = perc
            self.pb_status = status

        self.pb_ax.clear()
        self.create_progbar_rect()
        self.can_pb.draw()