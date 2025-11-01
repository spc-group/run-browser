import numpy as np
import pytest
import xarray as xr
from pyqtgraph import ImageView

from run_browser.frameset_view import FramesetImageView, FramesetView
from run_browser.testing import block_signals


@pytest.fixture()
def view(qtbot):
    fs_view = FramesetView()
    qtbot.addWidget(fs_view)
    return fs_view


def test_load_ui(view):
    """Make sure widgets were loaded from the UI file."""
    assert isinstance(view.ui.frame_view, ImageView)


def test_plot_frames(view):
    ds = xr.DataArray(
        np.random.rand(16, 8, 4),
        coords={"frame": range(16), "row": range(8), "column": range(4)},
    )
    view.plot(ds)
    im_plot = view.ui.frame_view
    assert np.array_equal(im_plot.image, ds.values)


def test_plot_spectra(view):
    ds = xr.DataArray(
        np.random.rand(16, 8, 4),
        coords={"frame": range(16), "row": range(8), "column": range(4)},
    )
    view.plot(ds)
    plot_item = view.ui.frame_view.spectra_view.getPlotItem()
    assert len(plot_item.dataItems) == 8
    # Plot again to make sure they don't duplicate plots
    view.plot(ds)
    assert len(plot_item.dataItems) == 8


def test_apply_no_roi(view):
    """Do we get the array back if no ROI is set?"""
    arr = np.random.rand(8, 8, 8)
    view.ui.frame_view.ui.roiBtn.setChecked(False)
    new_arr = view.apply_roi(arr)
    assert new_arr is arr


def test_apply_roi(view):
    """Can we read the current ROI from the frameset tab?"""
    arr = np.random.rand(16, 8, 4)
    view.ui.frame_view.ui.roiBtn.setChecked(True)
    new_arr = view.apply_roi(arr)
    assert new_arr is not arr
    assert new_arr.shape[0] == 16


def test_spectra_plot_widget():
    """Check that the imageview has both an imageplot and a spectra plot."""
    view = FramesetImageView()
    tab_widget = view.tab_widget
    assert view.ui.gridLayout.itemAtPosition(0, 0).widget() is tab_widget


def test_show_spectra_roi():
    view = FramesetImageView()
    assert not view.spectra_region.isVisible()
    # Check the ROI button
    with block_signals(view.ui.roiBtn):
        view.ui.roiBtn.setChecked(True)
    view.roiClicked()
    assert view.spectra_region.isVisible()
    # Uncheck the button again
    with block_signals(view.ui.roiBtn):
        view.ui.roiBtn.setChecked(False)
    view.roiClicked()
    assert not view.spectra_region.isVisible()


def test_link_spectra_roi():
    view = FramesetImageView()
    view.roiChanged()
    assert view.spectra_region.getRegion() == (0, 10)


def test_link_image_roi():
    view = FramesetImageView()
    with block_signals(view.spectra_region):
        view.spectra_region.setRegion((5, 20))
    view.update_image_roi()
    assert tuple(view.roi.size()) == (15.0, 10.0)
    assert tuple(view.roi.pos()) == (5.0, 0.0)
