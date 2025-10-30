import numpy as np
import pytest
import xarray as xr
from pyqtgraph import PlotWidget

from run_browser.spectra_view import COLORS, SpectraView


@pytest.fixture()
def view(qtbot):
    fs_view = SpectraView()
    qtbot.addWidget(fs_view)
    return fs_view


def test_load_ui(view):
    """Make sure widgets were loaded from the UI file."""
    assert isinstance(view.ui.plot_widget, PlotWidget)


def test_plot(view):
    arr = np.arange(16 * 8 * 4).reshape(16, 8, 4)
    ds = xr.DataArray(
        arr,
        coords={
            "frame": range(16),
            "row": np.linspace(0, 80, num=8),
            "energy": range(4),
        },
    )
    view.plot(ds)
    # Check the plot annotations
    plot_item = view.ui.plot_widget.getPlotItem()
    assert plot_item.getAxis("left").label.toPlainText() == "row "
    assert plot_item.getAxis("bottom").label.toPlainText() == "energy "
    # Get the individual curves that were plotted
    assert len(plot_item.dataItems) == 8
    colors = [item.opts["pen"] for item in plot_item.dataItems]
    assert colors == COLORS[:8]
    item = plot_item.dataItems[1]
    xdata, ydata = item.getData()
    np.testing.assert_equal(xdata, ds.coords["energy"])
    np.testing.assert_equal(ydata, arr[0][1])


# def test_apply_no_roi(view):
#     """Do we get the array back if no ROI is set?"""
#     arr = np.random.rand(8, 8, 8)
#     view.ui.frame_view.ui.roiBtn.setChecked(False)
#     new_arr = view.apply_roi(arr)
#     assert new_arr is arr


# def test_apply_roi(view):
#     """Can we read the current ROI from the frameset tab?"""
#     arr = np.random.rand(16, 8, 4)
#     view.ui.frame_view.ui.roiBtn.setChecked(True)
#     new_arr = view.apply_roi(arr)
#     assert new_arr is not arr
#     assert new_arr.shape[0] == 16
