import asyncio
import datetime as dt
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
import time_machine
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog

from run_browser.main_window import RunBrowserMainWindow, block_signals


@pytest_asyncio.fixture()
async def window(qtbot, mocker, tiled_client):
    mocker.patch(
        "run_browser.widgets.ExportDialog.exec_",
        return_value=QFileDialog.Accepted,
    )
    mocker.patch(
        "run_browser.widgets.ExportDialog.selectedFiles",
        return_value=["/net/s255data/export/test_file.nx"],
    )
    mocker.patch("run_browser.client.DatabaseWorker.export_runs")
    mocker.patch(
        "run_browser.main_window.list_profiles",
        return_value={
            "cortex": Path("/tmp/cortex"),
            "fedorov": Path("/tmp/fedorov"),
        },
    )
    mocker.patch(
        "run_browser.main_window.get_default_profile_name", return_value="cortex"
    )
    window = RunBrowserMainWindow()
    qtbot.addWidget(window)
    window.clear_filters()
    # Wait for the initial database load to process
    window.db.catalog = tiled_client
    try:
        yield window
    finally:
        # Make sure all the db tasks have a chance to finish cleanly
        [task.cancel() for task in window._running_db_tasks.values()]


@pytest.mark.asyncio
async def test_db_task(window):
    async def test_coro():
        return 15

    result = await window.db_task(test_coro())
    assert result == 15


@pytest.mark.asyncio
async def test_db_task_interruption(window):
    async def test_coro(sleep_time):
        await asyncio.sleep(sleep_time)
        return sleep_time

    # Create an existing task that will be cancelled
    task_1 = window.db_task(test_coro(1.0), name="testing")
    # Now execute another task
    result = await window.db_task(test_coro(0.01), name="testing")
    assert result == 0.01
    # Check that the first one was cancelled
    with pytest.raises(asyncio.exceptions.CancelledError):
        await task_1
    assert task_1.done()
    assert task_1.cancelled()


@pytest.mark.asyncio
async def test_load_runs(window):
    await window.load_runs()
    assert window.runs_model.rowCount() > 0
    assert window.ui.runs_total_label.text() == str(window.runs_model.rowCount())


@pytest.mark.asyncio
async def test_active_uids(window):
    await window.load_runs()
    # No rows at first
    assert window.active_uids() == set()
    # Check a row
    row, col = (0, 0)
    window.ui.runs_model.item(row, col).setCheckState(Qt.Checked)
    # Now there are some selected rows
    assert len(window.active_uids()) == 1


@pytest.mark.asyncio
async def test_metadata(window, qtbot, mocker):
    window.ui.metadata_tab.display_metadata = mocker.MagicMock()
    window.active_uids = mocker.MagicMock(
        return_value={"85573831-f4b4-4f64-b613-a6007bf03a8d"}
    )
    new_md = await window.update_metadata()
    assert "85573831-f4b4-4f64-b613-a6007bf03a8d" in new_md
    assert window.ui.metadata_tab.display_metadata.called


def test_busy_hints_run_widgets(window):
    """Check that the window widgets get disabled during DB hits."""
    with window.busy_hints(run_widgets=True, run_table=False):
        # Are widgets disabled in the context block?
        assert not window.ui.detail_tabwidget.isEnabled()
    # Are widgets re-enabled outside the context block?
    assert window.ui.detail_tabwidget.isEnabled()


def test_busy_hints_run_table(window):
    """Check that the all_runs table view gets disabled during DB hits."""
    with window.busy_hints(run_table=True, run_widgets=False):
        # Are widgets disabled in the context block?
        assert not window.ui.run_tableview.isEnabled()
    # Are widgets re-enabled outside the context block?
    assert window.ui.run_tableview.isEnabled()


def test_busy_hints_filters(window):
    """Check that the all_runs table view gets disabled during DB hits."""
    with window.busy_hints(run_table=False, run_widgets=False, filter_widgets=True):
        # Are widgets disabled in the context block?
        assert not window.ui.filters_widget.isEnabled()
    # Are widgets re-enabled outside the context block?
    assert window.ui.filters_widget.isEnabled()


def test_busy_hints_status(window, mocker):
    """Check that any busy_hints displays the message "Loading…"."""
    spy = mocker.spy(window, "show_message")
    with window.busy_hints(run_table=True, run_widgets=False):
        # Are widgets disabled in the context block?
        assert not window.ui.run_tableview.isEnabled()
        assert spy.call_count == 1
    # Are widgets re-enabled outside the context block?
    assert spy.call_count == 2
    assert window.ui.run_tableview.isEnabled()


def test_busy_hints_multiple(window):
    """Check that multiple busy hints can co-exist."""
    # Next the busy_hints context to mimic multiple async calls
    with window.busy_hints(run_widgets=True):
        # Are widgets disabled in the outer block?
        assert not window.ui.detail_tabwidget.isEnabled()
        with window.busy_hints(run_widgets=True):
            # Are widgets disabled in the inner block?
            assert not window.ui.detail_tabwidget.isEnabled()
        # Are widgets still disabled in the outer block?
        assert not window.ui.detail_tabwidget.isEnabled()
    # Are widgets re-enabled outside the context block?
    assert window.ui.detail_tabwidget.isEnabled()


@pytest.mark.asyncio
async def test_update_combobox_items(window):
    """Check that the comboboxes get the distinct filter fields."""
    await window.update_combobox_items()
    # Some of these have filters are disabled because they are slow
    # with sqlite They may be re-enabled when switching to postgres
    assert window.ui.filter_plan_combobox.count() > 0
    assert window.ui.filter_sample_combobox.count() > 0
    assert window.ui.filter_formula_combobox.count() > 0
    assert window.ui.filter_scan_combobox.count() > 0
    assert window.ui.filter_edge_combobox.count() > 0
    assert window.ui.filter_exit_status_combobox.count() > 0
    assert window.ui.filter_proposal_combobox.count() > 0
    assert window.ui.filter_esaf_combobox.count() > 0
    assert window.ui.filter_beamline_combobox.count() > 0


@pytest.mark.asyncio
async def test_export_button_enabled(window):
    assert not window.export_button.isEnabled()
    # Update the list with 1 run and see if the control gets enabled
    window.selected_runs = [{}]
    window.update_export_button()
    assert window.export_button.isEnabled()
    # Update the list with multiple runs and see if the control gets disabled
    window.selected_runs = [{}, {}]
    window.update_export_button()
    assert not window.export_button.isEnabled()


@pytest.mark.asyncio
async def test_export_button_clicked(window, mocker, qtbot):
    # Set up a run to be tested against
    run = AsyncMock()
    run.formats.return_value = [
        "application/json",
        "application/x-hdf5",
        "application/x-nexus",
    ]
    window.selected_runs = [run]
    window.update_export_button()
    # Clicking the button should open a file dialog
    await window.export_runs()
    assert window.export_dialog.exec_.called
    assert window.export_dialog.selectedFiles.called
    # Check that file filter names are set correctly
    # (assumes application/json is available on every machine)
    assert "JSON document (*.json)" in window.export_dialog.nameFilters()
    # Check that the file was saved
    assert window.db.export_runs.called
    files = window.export_dialog.selectedFiles.return_value
    assert window.db.export_runs.call_args.args == (files,)
    assert window.db.export_runs.call_args.kwargs["formats"] == ["application/json"]


fake_time = dt.datetime(2022, 8, 19, 19, 10, 51).astimezone()


@time_machine.travel(fake_time, tick=False)
def test_default_filters(window):
    window.reset_default_filters()
    assert window.ui.filter_exit_status_combobox.currentText() == "success"
    # Test datetime filters
    assert window.ui.filter_after_checkbox.isChecked()
    last_week = dt.datetime(2022, 8, 12, 19, 10, 51)
    after_filter_time = window.ui.filter_after_datetimeedit.dateTime()
    after_filter_time = dt.datetime.fromtimestamp(after_filter_time.toSecsSinceEpoch())
    assert after_filter_time == last_week
    next_week = dt.datetime(2022, 8, 26, 19, 10, 51)
    before_filter_time = window.ui.filter_before_datetimeedit.dateTime()
    before_filter_time = dt.datetime.fromtimestamp(
        before_filter_time.toSecsSinceEpoch()
    )
    assert before_filter_time == next_week


def test_time_filters(window):
    """Check that the before and after datetime filters are activated."""
    window.ui.filter_after_checkbox.setChecked(False)
    window.ui.filter_before_checkbox.setChecked(False)
    filters = window.filters()
    assert "after" not in filters
    assert "before" not in filters
    window.ui.filter_after_checkbox.setChecked(True)
    window.ui.filter_before_checkbox.setChecked(True)
    filters = window.filters()
    assert "after" in filters
    assert "before" in filters


@pytest.mark.asyncio
async def test_update_internal_data(window, qtbot, mocker):
    window.active_uids = mocker.MagicMock(
        return_value={"85573831-f4b4-4f64-b613-a6007bf03a8d"}
    )
    with block_signals(window.ui.stream_combobox, window.ui.x_signal_combobox):
        window.ui.stream_combobox.addItem("primary")
        window.ui.x_signal_combobox.addItem("x")
    window.ui.multiplot_tab.plot = mocker.MagicMock()
    await window.update_internal_data()
    # Check that the plotting routines were called correctly
    assert window.ui.multiplot_tab.plot.called
    args, kwargs = window.ui.multiplot_tab.plot.call_args
    datasets = args[0]
    assert len(datasets) == 1
    ds = datasets[0]
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.coords


@pytest.mark.asyncio
async def test_update_selected_data(window, qtbot, mocker):
    window.active_uids = mocker.MagicMock(return_value={"xarray_run"})
    window.selected_uid = mocker.MagicMock(return_value="xarray_run")
    with block_signals(
        window.ui.stream_combobox,
        window.ui.x_signal_combobox,
        window.ui.y_signal_combobox,
        window.ui.r_signal_combobox,
    ):
        window.ui.stream_combobox.addItem("primary")
        window.ui.x_signal_combobox.addItem("mono-energy")
        window.ui.y_signal_combobox.addItem("It-net_count")
        window.ui.r_signal_combobox.addItem("I0-net_count")
    # Check that the clients got called
    window.ui.lineplot_tab.plot = mocker.MagicMock()
    window.ui.gridplot_tab.plot = mocker.MagicMock()
    window.ui.frameset_tab.plot = mocker.MagicMock()
    window.ui.spectra_tab.plot = mocker.MagicMock()
    await window.update_selected_data()
    assert window.ui.lineplot_tab.plot.called
    args, kwargs = window.ui.lineplot_tab.plot.call_args
    dataset = args[0]
    assert isinstance(dataset, xr.Dataset)
    arr = dataset["xarray_run"]
    assert "mono-energy" in arr.coords
    assert dataset.attrs["data_label"] == "It-net_count"


@pytest.mark.asyncio
async def test_update_no_data_selected(window, qtbot, mocker):
    window.active_uids = mocker.MagicMock(return_value={})
    window.selected_uid = mocker.MagicMock(return_value=None)
    # Check that the clients got called
    window.ui.lineplot_tab.plot = mocker.MagicMock()
    window.ui.gridplot_tab.plot = mocker.MagicMock()
    window.ui.frameset_tab.plot = mocker.MagicMock()
    window.ui.spectra_tab.plot = mocker.MagicMock()
    await window.update_selected_data()
    # All the tab views should be disabled
    assert not window.ui.lineplot_tab.plot.called
    assert not window.ui.detail_tabwidget.isTabEnabled(window.Tabs.LINE)
    assert not window.ui.gridplot_tab.plot.called
    # assert not window.ui.detail_tabwidget.isTabEnabled(window.Tabs.GRID)
    assert not window.ui.frameset_tab.plot.called
    assert not window.ui.detail_tabwidget.isTabEnabled(window.Tabs.FRAMES)
    assert not window.ui.spectra_tab.plot.called
    assert not window.ui.detail_tabwidget.isTabEnabled(window.Tabs.SPECTRA)


@pytest.mark.asyncio
async def test_profile_choices(window):
    combobox = window.ui.profile_combobox
    items = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert items == ["cortex", "fedorov"]


@pytest.mark.asyncio
async def test_stream_choices(window, mocker):
    window.active_uids = mocker.MagicMock(
        return_value={"85573831-f4b4-4f64-b613-a6007bf03a8d"}
    )
    await window.update_streams()
    combobox = window.ui.stream_combobox
    items = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert items == ["primary", "baseline"]


@pytest.mark.asyncio
async def test_signal_options(window, mocker):
    """
    We need to know:
    - data_keys
    - independent hints (scan axes)
    - dependent hints (device hints)

    Used '64e85e20-106c-48e6-b643-77e9647b0242' for testing in the
    haven-dev catalog.

    """
    window.active_uids = mocker.MagicMock(
        return_value={"85573831-f4b4-4f64-b613-a6007bf03a8d"}
    )
    with block_signals(window.ui.stream_combobox, window.ui.use_hints_checkbox):
        await window.update_streams()
        window.ui.stream_combobox.setCurrentText("primary")
        window.ui.use_hints_checkbox.setChecked(False)
    # Check that we got the right signals in the right order
    await window.update_signal_widgets()
    expected_signals = [
        "energy_energy",
        "ge_8element",
        "ge_8element-deadtime_factor",
        "I0-net_count",
        "seq_num",
    ]
    combobox = window.ui.x_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == expected_signals
    combobox = window.ui.y_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == expected_signals
    combobox = window.ui.r_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == expected_signals


@pytest.mark.asyncio
async def test_hinted_signal_options(window, mocker):
    """
    We need to know:
    - data_keys
    - independent hints (scan axes)
    - dependent hints (device hints)

    Used '64e85e20-106c-48e6-b643-77e9647b0242' for testing in the
    haven-dev catalog.

    """
    window.active_uids = mocker.MagicMock(
        return_value={"85573831-f4b4-4f64-b613-a6007bf03a8d"}
    )
    with block_signals(window.ui.stream_combobox, window.ui.use_hints_checkbox):
        window.ui.stream_combobox.addItem("primary")
        window.ui.use_hints_checkbox.setChecked(True)
    await window.update_signal_widgets()
    # Check hinted X signals
    combobox = window.ui.x_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == ["aerotech_horiz", "aerotech_vert"]
    # Check hinted Y signals
    expected_signals = [
        "aerotech_horiz",
        "aerotech_vert",
        "CdnI0_net_counts",
        "CdnIPreKb_net_counts",
        "CdnIt_net_counts",
        "I0_net_counts",
        "Ipre_KB_net_counts",
        "Ipreslit_net_counts",
        "It_net_counts",
    ]
    combobox = window.ui.y_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == expected_signals
    # Check hinted reference signals
    combobox = window.ui.r_signal_combobox
    signals = [combobox.itemText(idx) for idx in range(combobox.count())]
    assert signals == expected_signals


data_reductions = [
    # (roi, array, expected)
    (np.linspace(1003, 1025, num=51), np.linspace(1003, 1025, num=51)),
    (np.array([[0, 1], [2, 3]]), np.array([1, 5])),
]


@pytest.mark.parametrize("arr,expected", data_reductions)
def test_reduce_nd_array(window, arr, expected):
    np.testing.assert_array_equal(window.reduce_nd_array(arr), expected)


def test_prepare_1d_data(window):
    with block_signals(window.ui.x_signal_combobox, window.ui.y_signal_combobox):
        window.ui.x_signal_combobox.addItem("mono-energy")
        window.ui.y_signal_combobox.addItem("It-net_count")
        window.ui.r_signal_combobox.addItem("I0-net_count")
        window.ui.r_operator_combobox.setCurrentText("÷")
    dataset = {
        "run1": xr.Dataset(
            {
                "I0-net_count": np.linspace(9658, 10334, num=51),
                "It-net_count": np.linspace(1003, 1025, num=51),
                "mono-energy": np.linspace(8325, 8355, num=51),
            }
        ),
    }
    new_data = window.prepare_1d_dataset(dataset)
    expected = xr.Dataset(
        {
            "run1": xr.DataArray(
                np.linspace(1003, 1025, num=51) / np.linspace(9658, 10334, num=51),
                coords={"mono-energy": np.linspace(8325, 8355, num=51)},
            ),
        }
    )
    assert new_data.equals(expected)
    assert new_data.attrs["data_label"] == "It-net_count ÷ I0-net_count"
    assert new_data.attrs["coord_label"] == "mono-energy"


def test_prepare_grid_data(window):
    with block_signals(window.ui.x_signal_combobox, window.ui.y_signal_combobox):
        window.ui.x_signal_combobox.addItem("mono-energy")
        window.ui.y_signal_combobox.addItem("I0-net_count")
    grid_shape = (15, 11)
    yy, xx = np.mgrid[:15, :11]
    data = xr.Dataset(
        {
            "I0-net_count": np.linspace(9658, 10334, num=np.prod(grid_shape)),
            "aerotech-vert": yy.flatten(),
            "aerotech-horiz": xx.flatten(),
        },
    )
    # Create the new dataset
    new_data = window.prepare_grid_dataset(
        data,
        grid_shape=grid_shape,
        extent=[],
        coord_signals=["aerotech-vert", "aerotech-horiz"],
    )
    # Verify the new dataset
    expected = xr.DataArray(
        np.linspace(9658, 10334, num=np.prod(grid_shape)).reshape(grid_shape),
        coords={
            "aerotech-vert": np.arange(15),
            "aerotech-horiz": np.arange(11),
        },
    )
    assert new_data.equals(expected)


def test_prepare_volume_data(window):
    with block_signals(window.ui.x_signal_combobox, window.ui.y_signal_combobox):
        window.ui.x_signal_combobox.addItem("mono-energy")
        window.ui.y_signal_combobox.addItem("vortex")
    shape = (16, 8, 4)
    data = xr.Dataset(
        {
            "vortex": xr.DataArray(
                np.linspace(0, 100, num=np.prod(shape)).reshape(shape)
            ),
            "mono-energy": np.linspace(8325, 8355, num=16),
        },
    )
    # Create the new dataset
    new_data = window.prepare_volume_dataset(data)
    # Verify the new dataset
    expected = xr.DataArray(
        data.vortex.values,
        coords={
            "mono-energy": data["mono-energy"],
            "coord_1": range(8),
            "coord_2": range(4),
        },
        name="vortex",
    )
    assert new_data.equals(expected)


@pytest.mark.xfail
def test_label_from_metadata():
    assert False


def test_axis_labels(window):
    with block_signals(
        window.ui.x_signal_combobox,
        window.ui.y_signal_combobox,
        window.ui.r_operator_combobox,
        window.ui.r_signal_combobox,
        window.ui.invert_checkbox,
        window.ui.logarithm_checkbox,
        window.ui.gradient_checkbox,
    ):
        window.ui.x_signal_combobox.addItem("signal_x")
        window.ui.y_signal_combobox.addItem("signal_y")
        window.ui.r_signal_combobox.addItem("signal_r")
        window.ui.r_operator_combobox.setCurrentText("+")
        window.ui.invert_checkbox.setChecked(True)
        window.ui.logarithm_checkbox.setChecked(True)
        window.ui.gradient_checkbox.setChecked(True)
    x_label, y_label = window.axis_labels()
    assert x_label == "signal_x"
    assert y_label == "∇(ln((signal_y + signal_r)⁻))"


def test_swap_signals(window):
    signal_names = ["It-net_current", "I0-net_current"]
    with block_signals(window.ui.y_signal_combobox, window.ui.r_signal_combobox):
        window.ui.y_signal_combobox.addItems(signal_names)
        window.ui.y_signal_combobox.setCurrentText(signal_names[0])
        window.ui.r_signal_combobox.addItems(signal_names)
        window.ui.r_signal_combobox.setCurrentText(signal_names[1])
        window.swap_signals()
    # Make sure the signals were actually swapped
    assert window.ui.y_signal_combobox.currentText() == signal_names[1]
    assert window.ui.r_signal_combobox.currentText() == signal_names[0]


@pytest.mark.xfail
def test_update_plot_mean(view):
    view.independent_hints = ["energy_energy"]
    view.dependent_hints = ["I0-net_current"]
    # view.data_keys = data_keys
    view.ui.aggregator_combobox.setCurrentText("StDev")
    # Update the plots
    # view.plot(dataframes)
    # Check the data were plotted
    plot_item = view.ui.plot_widget.getPlotItem()
    assert len(plot_item.dataItems) == 1


# -----------------------------------------------------------------------------
# :author:    Mark Wolfman
# :email:     wolfman@anl.gov
# :copyright: Copyright © 2023, UChicago Argonne, LLC
#
# Distributed under the terms of the 3-Clause BSD License
#
# The full license is in the file LICENSE, distributed with this software.
#
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# -----------------------------------------------------------------------------
