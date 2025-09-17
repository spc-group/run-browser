import logging
from pathlib import Path

import numpy as np
import pyqtgraph
import xarray as xr
from matplotlib.colors import TABLEAU_COLORS
from pyqtgraph import ImageView, PlotItem
from qtpy import QtWidgets, uic
from qtpy.QtCore import Slot
from scipy.interpolate import griddata

log = logging.getLogger(__name__)
colors = list(TABLEAU_COLORS.values())


pyqtgraph.setConfigOption("imageAxisOrder", "row-major")


class GridImageView(ImageView):
    def __init__(self, *args, view=None, **kwargs):
        if view is None:
            view = PlotItem()
        super().__init__(*args, view=view, **kwargs)


class GridplotView(QtWidgets.QWidget):
    """Handles the plotting of tabular data that was taken on a grid."""

    ui_file = Path(__file__).parent / "gridplot_view.ui"

    def __init__(self, parent=None):
        self.data_keys = {}
        self.independent_hints = []
        self.dependent_hints = []
        self.dataframes = {}
        self.metadata = {}
        super().__init__(parent)
        self.ui = uic.loadUi(self.ui_file, self)
        # Prepare plotting style
        vbox = self.ui.plot_widget.ui.roiPlot.getPlotItem().getViewBox()
        vbox.setBackgroundColor("k")
        # Connect internal signals/slots

    def regrid(
        self,
        points: np.ndarray,
        values: np.ndarray,
        shape: tuple[int],
        extent: tuple[tuple[float, float], tuple[float, float]],
    ):
        """Calculate a new image with a shape based on metadata."""
        # Prepare new regular grid to interpolate to
        (ymin, ymax), (xmin, xmax) = extent
        ystep, xstep = (npts * 1j for npts in shape)
        # Explicitly create slices to make type-checkers happy
        slices = [slice(ymin, ymax, ystep), slice(xmin, xmax, xstep)]
        yy, xx = np.mgrid[*slices]
        xi = np.c_[yy.flatten(), xx.flatten()]
        # Interpolate
        new_values = griddata(points, values, xi, method="cubic")
        return new_values

    @Slot()
    @Slot(dict)
    def plot(self, dataset: xr.DataArray):
        """Take loaded run data and plot it.

        Parameters
        ==========
        dataframe
          The gridded data array to plot. Data should be a 2D array.
        """
        self.clear_plot()
        # Plot this run's data
        if not (2 <= dataset.ndim <= 3):
            log.warning(f"Cannot plot image with {dataset.ndim} dimensions.")
        self.ui.plot_widget.setImage(dataset.values, autoRange=False)
        # Set axis labels
        img_item = self.ui.plot_widget.getImageItem()
        try:
            ylabel, xlabel = self.independent_hints
        except ValueError:
            log.warning(
                f"Could not determine grid labels from hints: {self.independent_hints}"
            )
        else:
            view = self.ui.plot_widget.view
            view.setLabels(left=ylabel, bottom=xlabel)
        # Set axes extent
        ycoords, xcoords = dataset.coords.values()
        xmin: float
        xmax: float
        ymin: float
        ymax: float
        xmin, xmax = np.min(xcoords.values), np.max(xcoords.values)
        ymin, ymax = np.min(ycoords.values), np.max(ycoords.values)
        x: float = xmin
        y: float = ymin
        w: float = xmax - xmin
        h: float = ymax - ymin
        img_item.setRect(x, y, w, h)

    def clear(self):
        """Reset the page to look blank."""
        self.clear_plot()

    def clear_plot(self):
        self.ui.plot_widget.getImageItem().clear()
        self.data_items = {}
