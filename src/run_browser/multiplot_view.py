import logging
from collections.abc import Generator, Sequence
from itertools import count
from pathlib import Path
from typing import Any

import xarray as xr
from pyqtgraph.graphicsItems import PlotItem
from qtpy import QtWidgets, uic
from qtpy.QtCore import Slot

log = logging.getLogger(__name__)


class MultiplotView(QtWidgets.QWidget):
    _multiplot_items: dict[tuple[int, int], PlotItem]
    ui_file = Path(__file__).parent / "multiplot_view.ui"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = uic.loadUi(self.ui_file, self)

    @Slot(list)
    @Slot()
    def plot(self, datasets: Sequence[xr.Dataset]) -> None:
        """Take loaded run data and plot small multiples.

        Parameters
        ==========
        dataframes
          Dictionary with pandas series for each run.
        xcolumn
          The name of the column in each dataframe that will be
          plotted on the horizontal axis. All other columns will be
          plotted on the vertical axis.

        """
        ysignals_ = {sig for ds in datasets for sig in ds.keys()}
        ysignals = sorted(ysignals_, key=str.lower)
        # Plot the runs
        self.clear_plot()
        for ysignal, plot_item in zip(ysignals, self.multiplot_items()):
            plot_item.setTitle(ysignal)
            for data in datasets:
                if ysignal not in data:
                    continue
                arr = data[ysignal]
                xdata = list(arr.coords.values())[0]
                plot_item.plot(xdata, arr.values)

    def clear_plot(self):
        """Remove all existing multiplot items from the view."""
        self.ui.plot_widget.clear()
        self._multiplot_items = {}

    def multiplot_items(self, n_cols: int = 3) -> Generator[PlotItem, Any, None]:
        """Generate plot multiples, creating new ones on the fly.

        Existing plot widgets will be yielded first, then after all
        existing plot items, new items are generated and added to the
        view.

        """
        view = self.ui.plot_widget
        item0 = None
        for idx in count():
            row = int(idx / n_cols)
            col = idx % n_cols
            # Make a new plot item if one doesn't exist
            if (row, col) not in self._multiplot_items:
                self._multiplot_items[(row, col)] = view.addPlot(row=row, col=col)
            new_item = self._multiplot_items[(row, col)]
            # Link the X-axes together
            if item0 is None:
                item0 = new_item
            else:
                new_item.setXLink(item0)
            # Resize the viewing area to fit the contents
            width = view.width()
            plot_width = width / n_cols
            view.setFixedHeight(1200)
            yield new_item
