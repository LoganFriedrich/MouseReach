
from qtpy.QtWidgets import * 
from qtpy.QtGui import QStandardItemModel, QStandardItem, QPainter
from qtpy.QtCore import Qt

class CheckableComboBox(QComboBox):
    
    # constructor
    def __init__(self, parent=None, width=140):
        super(CheckableComboBox, self).__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setMinimumWidth(width) # pixels
        # self.view().pressed.connect(self.handleItemPressed)
        # Checked event
        self.model().itemChanged.connect(self.handleItemChecked)
        self.selected_text = "All"

    # action called when item get checked
    def do_action(self):
        self.count = 0
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                self.count += 1
        print("Checked number : " +str(self.count))

    # when any item get pressed
    def handleItemChecked(self, item):

        # getting the item
        # item = self.model().itemFromIndex(index)

        # 'All' is checked
        if item.text() == "All":
            self.checkAll(item.checkState())
        else:
            # making it unchecked
            item.setCheckState(item.checkState())


        # update the text
        self.updateText()

    # update the text
    def updateText(self):
        text = ""
        self.count = self.model().rowCount()
        print(f"Number of items in the model: {self.count}")
        for i in range(self.count):
            item = self.model().item(i)
            if item is None:
                print(f"Item at index {i} is None")
                continue
            print(f"Item at index {i}: {item.text()}, CheckState: {item.checkState()}")
            if item.checkState() == Qt.Checked:
                if item.text() == "All":
                    text = "All"
                    break
                else:
                    text += item.text() + ","
        if text.endswith(","):
            text = text[:-1]  # Remove the trailing comma and space
        print(f"Setting combobox text to: {text}")
        self.selected_text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.selected_text
        self.style().drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option, painter)
        self.style().drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option, painter)

    

    # add item to the combo box
    def addItem(self, text):
        item = QStandardItem(text)
        # item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        # item.setData(Qt.Unchecked, Qt.CheckStateRole)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts):
        for text in texts:
            self.addItem(text)

    # get the checked items
    def checkedItems(self):
        items = []
        self.count = self.model().rowCount()
        for i in range(self.count):
            item = self.model().item(i)
            if item.checkState() == Qt.Checked:
                items.append(item.text())
        return items
    
    # Check all items if 'All' is checked
    def checkAll(self, state):
        print(f"Setting check state to: {state}")
        self.count = self.model().rowCount()
        self.model().itemChanged.disconnect(self.handleItemChecked)
        for i in range(self.count):
            item = self.model().item(i)
            if item is None:
                print(f"Item at index {i} is None")
            else:
                print(f"Setting check state for item at index {i}")
                item.setCheckState(state)

        self.model().itemChanged.connect(self.handleItemChecked)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    combo = CheckableComboBox(window)
    combo.addItem("All")
    combo.addItem("Item 1")
    combo.addItem("Item 2")
    combo.addItem("Item 3")
    combo.addItem("Item 4")
    layout.addWidget(combo)
    window.show()
    sys.exit(app.exec_())