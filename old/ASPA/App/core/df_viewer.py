from qtpy.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class DataFrameViewer(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DataFrame Viewer')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        self.set_table_data()

        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def set_table_data(self):
        self.table_widget.setRowCount(len(self.df.index))
        self.table_widget.setColumnCount(len(self.df.columns))
        self.table_widget.setHorizontalHeaderLabels(self.df.columns)

        for row in range(len(self.df.index)):
            for col in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iat[row, col]))
                self.table_widget.setItem(row, col, item)

        self.table_widget.resizeColumnsToContents()
