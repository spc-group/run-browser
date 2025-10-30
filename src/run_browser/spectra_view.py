import logging
from pathlib import Path

import xarray as xr
from matplotlib.colors import TABLEAU_COLORS
from qtpy import QtCore, QtWidgets, uic

log = logging.getLogger("run_browser")
COLORS = list(TABLEAU_COLORS.values())


class SpectraView(QtWidgets.QWidget):
    ui_file = Path(__file__).parent / "spectra_view.ui"
    array: xr.DataArray

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi(self.ui_file, self)

    @QtCore.Slot()
    def plot(self, array: xr.DataArray):
        """Plot a dataset as a stack of set of spectral lines."""
        self.array = array
        self.clear()
        self._plot()
        ylabel, xlabel = array.dims[-2:]
        plot_item = self.ui.plot_widget.getPlotItem()
        print(xlabel, ylabel)
        plot_item.setLabels(left=ylabel, bottom=xlabel)

    def _plot(self):
        frame = self.array[self.ui.z_slider.value()]
        plot_widget = self.ui.plot_widget
        xdim = self.array.dims[2]
        xdata = self.array.coords[xdim]
        for idx, line in enumerate(frame):
            color = COLORS[idx % len(COLORS)]
            plot_widget.plot(xdata, line, pen=color)

    def clear(self):
        self.ui.plot_widget.getPlotItem().clear()
