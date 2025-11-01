import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import qtawesome as qta
import xarray as xr
from matplotlib.colors import TABLEAU_COLORS
from numpy.typing import NDArray
from qtpy import QtCore, QtWidgets, uic

axes = namedtuple("axes", ("z", "y", "x"))

log = logging.getLogger(__name__)
COLORS = list(TABLEAU_COLORS.values())


class FramesetImageView(pg.ImageView):
    def __init__(self, *args, view=None, **kwargs):
        if view is None:
            view = pg.PlotItem()
        super().__init__(*args, view=view, **kwargs)
        self.timeLine.setPen((255, 255, 0, 200), width=5)
        self.timeLine.setHoverPen("r", width=5)
        self.setColorMap(pg.colormap.get("viridis"))
        # Add tabs so we can switch between image and spectra plot views
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setObjectName("tab_widget")
        # # Images tabs
        self.image_page = QtWidgets.QWidget()
        self.image_page.setObjectName("image_page")
        self.image_layout = QtWidgets.QVBoxLayout(self.image_page)
        self.image_layout.setObjectName("image_layout")
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_widget.addTab(self.image_page, "&Images")
        # Spectra tab
        self.spectra_page = QtWidgets.QWidget()
        self.spectra_layout = QtWidgets.QVBoxLayout(self.spectra_page)
        self.spectra_layout.setContentsMargins(0, 0, 0, 0)
        self.spectra_view = pg.PlotWidget()
        self.spectra_layout.addWidget(self.spectra_view)
        self.tab_widget.addTab(self.spectra_page, "&Spectra")
        # Restructure the layout with the new tab widget
        self.image_view = self.ui.gridLayout.takeAt(0).widget()
        self.image_layout.addWidget(self.image_view)
        self.ui.gridLayout.addWidget(self.tab_widget, 0, 0, 2, 1)
        # Link the y axes
        self.view.setXLink(self.spectra_view.getPlotItem())
        self.roi.removeHandle(1)  # remove the rotation handle

    def updateImage(self, *args, **kwargs):
        ## Redraw image on screen
        super().updateImage(*args, **kwargs)
        self.update_spectra()

    def update_spectra(self):
        if self.image is None:
            return

        image = self.getProcessedImage()

        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == "col-major":
            axorder = ["t", "x", "y", "c"]
        else:
            axorder = ["t", "y", "x", "c"]
        axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
        image = image.transpose(axorder)
        print(f"{self.imageItem.axisOrder=}")
        # Select time index
        if self.axes["t"] is not None:
            self.ui.roiPlot.show()
            image = image[self.currentIndex]
        # Do spectra plotting
        spectra = image
        plot_item = self.spectra_view.getPlotItem()
        # Update existing items
        for data_item, ydata in zip(plot_item.dataItems, spectra):
            data_item.setData(ydata)
        # Plot new items
        new_rows = spectra[len(plot_item.dataItems) :]
        for row_idx, ydata in enumerate(new_rows):
            color = COLORS[row_idx % len(COLORS)]
            plot_item.plot(ydata, pen=color, name=str(row_idx), clear=False)


class FramesetView(QtWidgets.QWidget):
    ui_file = Path(__file__).parent / "frameset_view.ui"
    spectra: np.ndarray | None = None
    aggregators = {
        "Mean": np.mean,
        "Median": np.median,
        "StDev": np.std,
    }
    datasets: dict[str, np.ndarray] | None = None
    data_frames: dict[str, pd.DataFrame] | None = None

    dataset_selected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi(self.ui_file, self)
        # Set up control widgets
        self.ui.lock_aspect_button.toggled.connect(
            self.frame_view.getView().setAspectLocked
        )
        self.ui.lock_aspect_button.toggled.connect(self.toggle_lock_icon)
        self.toggle_lock_icon(True)

    def toggle_lock_icon(self, state: bool):
        """Toggle the lock icon on the lock aspect button."""
        if state:
            icon = qta.icon("fa6s.lock")
        else:
            icon = qta.icon("fa6s.unlock")
        self.ui.lock_aspect_button.setIcon(icon)

    def row_count(self, layout: QtWidgets.QGridLayout) -> int:
        """How many rows in *layout* actually contain widgets."""
        rows = 0
        for i in range(layout.count()):
            row, _, span, _ = layout.getItemPosition(i)
            rows = max(rows, row + span)
        return rows

    @QtCore.Slot()
    def plot(self, array: xr.DataArray):
        """Plot a dataset as a stack of frames."""
        # Start with a clear plot
        self.clear()
        view = self.ui.frame_view
        # Determine how to plot the time series values
        tvals = list(array.coords.values())[0]
        # Do image plotting
        print(f"view: {array.values.shape}")
        view.setImage(array.values, xvals=tvals.values, axes={"t": 0, "y": 1, "x": 2})

    def apply_roi(self, arr: NDArray) -> NDArray:
        im_view = self.ui.frame_view
        if not im_view.ui.roiBtn.isChecked():
            return arr
        roi = im_view.roi
        arr = roi.getArrayRegion(data=arr, img=im_view.imageItem, axes=(1, 2))
        return arr

    def clear(self):
        im_plot = self.ui.frame_view
        im_plot.clear()
