import logging

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget

log = logging.getLogger(__name__)


class FiltersWidget(QWidget):
    returnPressed = Signal()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        # Check for return keys pressed
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.returnPressed.emit()
