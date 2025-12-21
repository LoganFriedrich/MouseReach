import sys
import pandas as pd
from qtpy.QtWidgets import QApplication, QMainWindow, QTableView, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from qtpy.QtCore import Qt, QAbstractTableModel, QVariant

class PandasTableModel(QAbstractTableModel):
    def __init__(self, data_df):
        super().__init__()
        self.data_df = data_df
        self.filtered_df = data_df.copy()

    def rowCount(self, parent=None):
        return self.filtered_df.shape[0]

    def columnCount(self, parent=None):
        return self.filtered_df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self.filtered_df.iloc[index.row(), index.column()])
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.filtered_df.columns[section])
            if orientation == Qt.Vertical:
                return str(self.filtered_df.index[section])
        return QVariant()

    def filter(self, age_filter, city_filter, name_filter):
        filtered_df = self.data_df

        if age_filter:
            filtered_df = filtered_df[filtered_df['Age'].apply(lambda x: age_filter.lower() in str(x).lower())]

        if city_filter:
            filtered_df = filtered_df[filtered_df['City'].apply(lambda x: city_filter.lower() in str(x).lower())]

        if name_filter:
            filtered_df = filtered_df[filtered_df['Name'].apply(lambda x: name_filter.lower() in str(x).lower())]

        self.filtered_df = filtered_df
        self.layoutChanged.emit()

class FilterTableApp(QMainWindow):
    def __init__(self, data_df):
        super().__init__()

        self.data_df = data_df
        self.table_model = PandasTableModel(data_df)
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        self.age_filter_line_edit = QLineEdit()
        self.age_filter_line_edit.setPlaceholderText("Filter by Age")
        self.age_filter_line_edit.textChanged.connect(self.filter_data)

        self.city_filter_line_edit = QLineEdit()
        self.city_filter_line_edit.setPlaceholderText("Filter by City")
        self.city_filter_line_edit.textChanged.connect(self.filter_data)

        self.name_filter_line_edit = QLineEdit()
        self.name_filter_line_edit.setPlaceholderText("Filter by Name")
        self.name_filter_line_edit.textChanged.connect(self.filter_data)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by Age:"))
        filter_layout.addWidget(self.age_filter_line_edit)

        filter_layout.addWidget(QLabel("Filter by City:"))
        filter_layout.addWidget(self.city_filter_line_edit)

        filter_layout.addWidget(QLabel("Filter by Name:"))
        filter_layout.addWidget(self.name_filter_line_edit)

        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)

        self.layout.addLayout(filter_layout)
        self.layout.addWidget(self.table_view)

        self.setCentralWidget(self.central_widget)

    def filter_data(self):
        age_filter = self.age_filter_line_edit.text()
        city_filter = self.city_filter_line_edit.text()
        name_filter = self.name_filter_line_edit.text()

        self.table_model.filter(age_filter, city_filter, name_filter)

def main():
    # Sample DataFrame
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 22, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
    }
    data_df = pd.DataFrame(data)

    app = QApplication(sys.argv)
    window = FilterTableApp(data_df)
    window.setWindowTitle('Pandas DataFrame in PyQt Table with Filters')
    window.setGeometry(100, 100, 800, 400)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
