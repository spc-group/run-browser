import asyncio
import datetime as dt
import enum
import logging
import warnings
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import qtawesome as qta
import xarray as xr
from numpy.typing import NDArray
from qasync import asyncSlot
from qtpy import uic
from qtpy.QtCore import QDateTime, Qt
from qtpy.QtGui import QIcon, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QApplication, QComboBox, QMainWindow
from tiled.client import from_profile_async
from tiled.profiles import get_default_profile_name, list_profiles

from run_browser.client import DatabaseWorker, DataSignal

log = logging.getLogger(__name__)


DEFAULT_PROFILE = get_default_profile_name()
if DEFAULT_PROFILE is None:
    msg = (
        "No default Tiled profile set. "
        "See https://blueskyproject.io/tiled/how-to/profiles.html"
    )
    log.warning(msg)
    warnings.warn(msg)


reference_operators = {
    "": lambda x, y: x,
    "+": np.add,
    "−": np.subtract,
    "×": np.multiply,
    "÷": np.divide,
}


def apply_reference(
    a: NDArray, b: NDArray, operator: Callable[[NDArray, NDArray], NDArray]
) -> NDArray:
    """Apply a reference correction to the data."""
    # Determine shape corrections if needed
    if a.ndim > b.ndim:
        new_shape = [*b.shape, *([1] * (a.ndim - b.ndim))]
        b = np.reshape(b, new_shape)
    elif a.ndim < b.ndim:
        new_shape = [*a.shape, *([1] * (b.ndim - a.ndim))]
        a = np.reshape(a, new_shape)
    return operator(a, b)


def cancellable(fn):
    @wraps(fn)
    async def inner(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except asyncio.exceptions.CancelledError:
            log.warning(f"Cancelled task {fn}")

    return inner


@contextmanager
def block_signals(*widgets):
    """Disable Qt signals so tests can be set up."""
    for widget in widgets:
        widget.blockSignals(True)
    try:
        yield
    finally:
        for widget in widgets:
            widget.blockSignals(False)


class RunBrowserMainWindow(QMainWindow):
    ui_file = Path(__file__).parent / "main_window.ui"

    runs_model: QStandardItemModel
    _run_col_names: Sequence = [
        "✓",
        "Plan",
        "Sample",
        "Scan",
        "Edge",
        "Exit Status",
        "Datetime",
        "UID",
    ]

    _running_db_tasks: Mapping

    # Counter for keeping track of UI hints for long DB hits
    _busy_hinters: Counter

    class Tabs(enum.IntEnum):
        METADATA = 0
        MULTIPLOT = 1
        LINE = 2
        GRID = 3
        FRAMES = 4
        SPECTRA = 5
        VOLUME = 6

    def __init__(
        self, *args, merge_streams: bool = True, plot_spectra: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.load_ui(merge_streams=merge_streams)
        self._plot_spectra = plot_spectra
        self._running_db_tasks = {}
        self._busy_hinters = Counter()
        self.reset_default_filters()
        self.db = DatabaseWorker()
        if not plot_spectra:
            self.frameset_tab.frame_view.tab_widget.removeTab(1)

    def show_message(self, message: str, delay: int | None = None):
        log.info(message)

    def load_profiles(self):
        """Prepare to use a set of databases accessible through *tiled_client*."""
        profile_names = list_profiles().keys()
        self.ui.profile_combobox.addItems(profile_names)
        self.ui.profile_combobox.setCurrentText(DEFAULT_PROFILE)

    def enable_reference_combobox(self, operator: str):
        self.r_signal_combobox.setEnabled(bool(operator))

    @asyncSlot(str)
    @cancellable
    async def change_catalog(self, profile_name: str = DEFAULT_PROFILE):
        """Activate a different catalog in the Tiled server."""
        if profile_name == "":
            self.db.catalog = None
            return
        self.db.catalog = await from_profile_async(profile_name)
        await self.db_task(
            asyncio.gather(self.load_runs(), self.update_combobox_items()),
            name="change_catalog",
        )

    def db_task(self, coro, name="default task"):
        """Executes a co-routine as a database task. Existing database
        tasks with the same *name* get cancelled.

        """
        # Check for existing tasks
        has_previous_task = name in self._running_db_tasks.keys()
        task_is_running = has_previous_task and not self._running_db_tasks[name].done()
        if task_is_running:
            self._running_db_tasks[name].cancel("New database task started.")
        # Wait on this task to be done
        new_task = asyncio.ensure_future(coro)
        self._running_db_tasks[name] = new_task
        return new_task

    @asyncSlot()
    async def reload_runs(self):
        """A simple wrapper to make load_runs a slot."""
        await self.load_runs()

    @cancellable
    async def load_runs(self):
        """Get the list of available runs based on filters."""
        with self.busy_hints(run_widgets=True, run_table=True, filter_widgets=False):
            runs = await self.db_task(
                self.db.load_all_runs(self.filters()),
                name="load all runs",
            )
            # Update the table view data model
            self.runs_model.clear()
            self.runs_model.setHorizontalHeaderLabels(self._run_col_names)
            for run in runs:
                checkbox = QStandardItem(True)
                checkbox.setCheckable(True)
                checkbox.setCheckState(Qt.Unchecked)
                items = [checkbox]
                items += [QStandardItem(val) for val in run.values()]
                self.ui.runs_model.appendRow(items)
            # Adjust the layout of the data table
            sort_col = self._run_col_names.index("Datetime")
            self.ui.run_tableview.sortByColumn(sort_col, Qt.DescendingOrder)
            self.ui.run_tableview.resizeColumnsToContents()
            # Let slots know that the model data have changed
            self.runs_total_label.setText(str(self.ui.runs_model.rowCount()))

    def clear_filters(self):
        self.ui.filter_plan_combobox.setCurrentText("")
        self.ui.filter_sample_combobox.setCurrentText("")
        self.ui.filter_formula_combobox.setCurrentText("")
        self.ui.filter_scan_combobox.setCurrentText("")
        self.ui.filter_edge_combobox.setCurrentText("")
        self.ui.filter_exit_status_combobox.setCurrentText("")
        self.ui.filter_user_combobox.setCurrentText("")
        self.ui.filter_proposal_combobox.setCurrentText("")
        self.ui.filter_esaf_combobox.setCurrentText("")
        self.ui.filter_beamline_combobox.setCurrentText("")
        self.ui.filter_after_checkbox.setChecked(False)
        self.ui.filter_before_checkbox.setChecked(False)
        self.ui.filter_full_text_lineedit.setText("")
        self.ui.filter_standards_checkbox.setChecked(False)

    def reset_default_filters(self):
        self.clear_filters()
        self.ui.filter_exit_status_combobox.setCurrentText("success")
        self.ui.filter_after_checkbox.setChecked(True)
        last_week = dt.datetime.now().astimezone() - dt.timedelta(days=7)
        last_week = QDateTime.fromSecsSinceEpoch(int(last_week.timestamp()))
        self.ui.filter_after_datetimeedit.setDateTime(last_week)
        next_week = dt.datetime.now().astimezone() + dt.timedelta(days=7)
        next_week = QDateTime.fromSecsSinceEpoch(int(next_week.timestamp()))
        self.ui.filter_before_datetimeedit.setDateTime(next_week)

    async def update_combobox_items(self):
        """"""
        filter_boxes = {
            "start.plan_name": self.ui.filter_plan_combobox,
            "start.sample_name": self.ui.filter_sample_combobox,
            "start.sample_formula": self.ui.filter_formula_combobox,
            "start.scan_name": self.ui.filter_scan_combobox,
            "start.edge": self.ui.filter_edge_combobox,
            "stop.exit_status": self.ui.filter_exit_status_combobox,
            "start.proposal_id": self.ui.filter_proposal_combobox,
            "start.esaf_id": self.ui.filter_esaf_combobox,
            "start.beamline": self.ui.filter_beamline_combobox,
        }
        # Clear old entries first so we don't have stale ones
        for key, cb in filter_boxes.items():
            cb.clear()
        # Populate with new results
        async for field_name, fields in self.db.distinct_fields():
            cb = filter_boxes[field_name]
            old_value = cb.currentText()
            cb.addItems(fields)
            cb.setCurrentText(old_value)

    def load_ui(self, merge_streams: bool = True):
        self.ui = uic.loadUi(self.ui_file, self)
        self.load_models()
        self.load_profiles()
        # Disable the merge signals checkbox if the feature is disabled
        if not merge_streams:
            self.ui.merge_streams_checkbox.setVisible(False)
        # Add window icon
        root_dir = Path(__file__).parent.absolute()
        icon_path = root_dir / "favicon.png"
        self.setWindowIcon(QIcon(str(icon_path)))
        # Set icons for the tabs
        self.ui.detail_tabwidget.setTabIcon(self.Tabs.METADATA, qta.icon("fa6s.list"))
        self.ui.detail_tabwidget.setTabIcon(
            self.Tabs.MULTIPLOT, qta.icon("fa6s.chart-line")
        )
        self.ui.detail_tabwidget.setTabIcon(self.Tabs.LINE, qta.icon("fa6s.chart-line"))
        self.ui.detail_tabwidget.setTabIcon(
            self.Tabs.GRID, qta.icon("fa6s.table-cells")
        )
        self.ui.detail_tabwidget.setTabIcon(self.Tabs.FRAMES, qta.icon("fa6s.images"))
        self.ui.detail_tabwidget.setTabIcon(self.Tabs.SPECTRA, qta.icon("fa6s.images"))
        # Enable the datetime pickers when activated
        self.ui.filter_after_checkbox.setChecked(True)
        self.ui.filter_before_checkbox.setChecked(False)
        self.ui.filter_before_checkbox.toggled.connect(
            self.ui.filter_before_datetimeedit.setEnabled
        )
        self.ui.filter_after_checkbox.toggled.connect(
            self.ui.filter_after_datetimeedit.setEnabled
        )
        # Setup controls for select which run to show
        self.ui.runs_model.dataChanged.connect(self.update_all_views)
        self.ui.run_tableview.selectionModel().selectionChanged.connect(
            self.update_all_views
        )
        self.ui.refresh_runs_button.setIcon(qta.icon("fa6s.arrows-rotate"))
        self.ui.refresh_runs_button.clicked.connect(self.reload_runs)
        self.ui.reset_filters_button.clicked.connect(self.reset_default_filters)
        # Select a new catalog
        self.ui.profile_combobox.currentTextChanged.connect(self.change_catalog)
        # Respond to filter controls getting updated
        self.ui.filters_widget.returnPressed.connect(self.refresh_runs_button.click)
        # Respond to controls for the current run
        self.ui.reload_plots_button.clicked.connect(self.update_all_views)
        self.ui.stream_combobox.currentTextChanged.connect(self.update_signal_widgets)
        # Respond to signal selection widgets
        self.ui.use_hints_checkbox.stateChanged.connect(self.update_signal_widgets)
        self.ui.use_hints_checkbox.stateChanged.connect(self.update_internal_data)
        self.ui.x_signal_combobox.currentTextChanged.connect(self.update_internal_data)
        self.ui.x_signal_combobox.currentTextChanged.connect(self.update_selected_data)
        self.ui.v_signal_combobox.currentTextChanged.connect(self.update_selected_data)
        self.ui.swap_button.setIcon(qta.icon("mdi.swap-horizontal"))
        self.ui.swap_button.clicked.connect(self.swap_signals)
        self.ui.r_operator_combobox.currentTextChanged.connect(
            self.enable_reference_combobox
        )
        self.ui.r_operator_combobox.currentTextChanged.connect(
            self.update_selected_data
        )
        self.ui.r_signal_combobox.currentTextChanged.connect(self.update_selected_data)
        self.ui.logarithm_checkbox.stateChanged.connect(self.update_selected_data)
        self.ui.invert_checkbox.stateChanged.connect(self.update_selected_data)
        self.ui.gradient_checkbox.stateChanged.connect(self.update_selected_data)
        # Connect window controls
        self.ui.exit_action.triggered.connect(QApplication.quit)

    def swap_signals(self):
        """Swap the value and reference signals."""
        new_r = self.ui.v_signal_combobox.currentText()
        new_y = self.ui.r_signal_combobox.currentText()
        self.ui.v_signal_combobox.setCurrentText(new_y)
        self.ui.r_signal_combobox.setCurrentText(new_r)

    @asyncSlot()
    @cancellable
    async def update_signal_widgets(self):
        """Update the UI based on new data keys and hints.

        If any of *data_keys*, *independent_hints* or
        *dependent_hints* are used, then the last seen values will be
        used.

        """
        data_signals, ihints, dhints = await self.data_signals()
        # Decide whether we want to use hints
        use_hints = self.ui.use_hints_checkbox.isChecked()
        xsigs, vsigs = data_signals, data_signals
        if use_hints:
            xsigs = [sig for sig in xsigs if sig.is_scan_dimension]
            vsigs = [sig for sig in vsigs if sig.is_hinted]
        # Update the UI
        comboboxes = [
            self.ui.x_signal_combobox,
            self.ui.v_signal_combobox,
            self.ui.r_signal_combobox,
        ]
        for combobox, new_signals in zip(comboboxes, [xsigs, vsigs, vsigs]):
            self._set_combobox_signals(combobox, new_signals)

    def _set_combobox_signals(self, combobox: QComboBox, signals: Sequence[DataSignal]):
        # If we're using merged streams, we want to include the stream at the end
        active_stream = self.ui.stream_combobox.currentText()
        # Now go through and add the options
        old_value = combobox.currentText()
        signals = sorted(signals, key=lambda sig: sig.name.lower())
        with block_signals(combobox):
            combobox.clear()
            for signal in signals:
                if signal.stream in [active_stream, None]:
                    text = signal.name
                else:
                    text = f"{signal.name} ({signal.stream})"
                combobox.addItem(text, userData=signal)
                if text == old_value:
                    combobox.setCurrentText(text)

    def auto_range(self):
        self.plot_1d_view.autoRange()

    def update_busy_hints(self):
        """Enable/disable UI elements based on the active hinters."""
        # Widgets for showing plots for runs
        if self._busy_hinters["run_widgets"] > 0:
            self.ui.detail_tabwidget.setEnabled(False)
        else:
            # Re-enable the run widgets
            self.ui.detail_tabwidget.setEnabled(True)
        # Widgets for selecting which runs to show
        if self._busy_hinters["run_table"] > 0:
            self.ui.run_tableview.setEnabled(False)
        else:
            # Re-enable the run widgets
            self.ui.run_tableview.setEnabled(True)
        # Widgets for filtering runs
        if self._busy_hinters["filters_widget"] > 0:
            self.ui.filters_widget.setEnabled(False)
        else:
            self.ui.filters_widget.setEnabled(True)
        # Update status message in message bars
        if len(list(self._busy_hinters.elements())) > 0:
            self.show_message("Loading…")
        else:
            self.show_message("Done.", 5000)

    @contextmanager
    def busy_hints(self, run_widgets=True, run_table=True, filter_widgets=True):
        """A context manager that displays UI hints when slow operations happen.

        Arguments can be used to control which widgets are modified.

        Usage:

        .. code-block:: python

            with self.busy_hints():
                self.db_task(self.slow_operation)

        Parameters
        ==========
        run_widgets
          Disable the widgets for viewing individual runs.
        run_table
          Disable the table for selecting runs to view.
        filter_widgets
          Disable the filter comboboxes, etc.

        """
        # Update the counters for keeping track of concurrent contexts
        hinters = {
            "run_widgets": run_widgets,
            "run_table": run_table,
            "filters_widget": filter_widgets,
        }
        hinters = [name for name, include in hinters.items() if include]
        self._busy_hinters.update(hinters)
        # Update the UI (e.g. disable widgets)
        self.update_busy_hints()
        # Run the innner context code
        try:
            yield
        finally:
            # Re-enable widgets if appropriate
            self._busy_hinters.subtract(hinters)
            self.update_busy_hints()

    @asyncSlot()
    async def update_streams(self, *args):
        """Update the list of available streams to choose from."""
        stream_names = await self.db.stream_names(self.active_uids())
        # Sort so that "primary" is first
        stream_names = sorted(stream_names, key=lambda x: x != "primary")
        with block_signals(self.ui.stream_combobox):
            self.ui.stream_combobox.clear()
            self.ui.stream_combobox.addItems(stream_names)
            if "primary" in stream_names:
                self.ui.stream_combobox.setCurrentText("primary")
        await self.update_signal_widgets()

    async def active_streams(self):
        """Return names of the current "active" streams.

        If merged streams are enabled, then this will be everything
        except the baseline. Otherwise it will be a length-1 list with
        the current selected stream.

        """
        if self.ui.merge_streams_checkbox.isChecked():
            streams = await self.db.stream_names(self.active_uids())
            streams = [stream for stream in streams if stream != "baseline"]
        else:
            streams = [self.ui.stream_combobox.currentText()]
        return streams

    @asyncSlot()
    async def update_metadata(self, *args) -> dict[str, dict]:
        """Render metadata for the runs into the metadata widget."""
        # Combine the metadata in a human-readable output
        new_md = await self.db_task(
            self.db.metadata(uids=self.active_uids()), "metadata"
        )
        self.ui.metadata_tab.display_metadata(new_md)
        return new_md

    @asyncSlot()
    @cancellable
    async def update_all_views(self):
        """Get new data, and update all the plots.

        If a *uid* is provided, only the plots matching the scan with
        *uid* will be updated.
        """
        await self.update_streams()
        await asyncio.gather(
            self.update_metadata(),
            self.update_internal_data(),
            self.update_selected_data(),
        )

    async def data_signals(self, uids=None) -> tuple[dict, set[str], set[str]]:
        """Get valid keys and hints for the selected UIDs."""
        streams = await self.active_streams()
        if uids is None:
            uids = self.active_uids()
        # Do the database hit
        with self.busy_hints(run_widgets=True, run_table=False, filter_widgets=False):
            data_keys, hints = await asyncio.gather(
                self.db_task(
                    self.db.data_signals(uids, streams), "update data signals"
                ),
                self.db_task(self.db.hints(uids, streams), "update data hints"),
            )
        independent_hints, dependent_hints = hints
        return data_keys, set(independent_hints), set(dependent_hints)

    def axis_labels(self):
        xlabel = self.ui.x_signal_combobox.currentText()
        ylabel = self.ui.v_signal_combobox.currentText()
        rlabel = self.ui.r_signal_combobox.currentText()
        roperator = self.ui.r_operator_combobox.currentText()
        if roperator != "":
            ylabel = f"{ylabel} {roperator} {rlabel}"
        if self.ui.invert_checkbox.isChecked():
            ylabel = f"({ylabel})⁻¹"
        if self.ui.logarithm_checkbox.isChecked():
            ylabel = f"ln({ylabel})"
        if self.ui.gradient_checkbox.isChecked():
            ylabel = f"∇({ylabel})"
        return xlabel, ylabel

    # def label_from_metadata(self, start_doc: Mapping) -> str:
    #     # Determine label from metadata
    #     uid = start_doc.get("uid", "")
    #     sample_name = start_doc.get("sample_name")
    #     scan_name = start_doc.get("scan_name")
    #     sample_formula = start_doc.get("sample_formula")
    #     if sample_name is not None and sample_formula is not None:
    #         sample_name = f"{sample_name} ({sample_formula})"
    #     elif sample_formula is not None:
    #         sample_name = sample_formula
    #     md_values = [val for val in [sample_name, scan_name] if val is not None]
    #     # Use the full UID unless we have something else to show
    #     if len(md_values) > 0:
    #         uid = uid.split("-")[0]
    #     # Build the label
    #     label = " — ".join([uid, *md_values])
    #     if start_doc.get("is_standard", False):
    #         label = f"{label} ★"
    #     return label

    @asyncSlot()
    @cancellable
    async def update_internal_data(self) -> dict[str, pd.DataFrame]:
        """Load only signals for the "internal" part of the run, and plot."""
        stream = self.ui.stream_combobox.currentText()
        uids = self.active_uids()
        x_signal = self.ui.x_signal_combobox.currentText()
        use_hints = self.ui.use_hints_checkbox.isChecked()
        _, hints = await self.db.hints(uids, streams=[stream])
        if stream == "":
            dataframes = {}
            log.info("Not loading dataframes for empty stream.")
        else:
            with self.busy_hints(
                run_widgets=True, run_table=False, filter_widgets=False
            ):
                dataframes = await self.db.dataframes(uids, stream)

        def should_plot(col):
            return (col != x_signal) and (col in hints or not use_hints)

        # Convert to standard format datasets
        def to_dataset(df):
            coords = {x_signal: df[x_signal].values}
            return xr.Dataset(
                {
                    col: xr.DataArray(df[col].values, coords=coords)
                    for col in df.columns
                    if should_plot(col)
                }
            )

        datasets = [
            to_dataset(df) for df in dataframes.values() if x_signal in df.columns
        ]
        self.ui.multiplot_tab.plot(datasets)
        return dataframes

    def reduce_nd_array(self, array: NDArray) -> NDArray:
        """Convert an ND array to a 1D array"""
        if array.ndim == 1:
            return array
        new_array = self.ui.frameset_tab.apply_roi(array)
        new_array = np.sum(new_array, axis=tuple(range(1, new_array.ndim)))
        return new_array

    def prepare_1d_dataset(self, datasets: dict[str, xr.Dataset]) -> xr.Dataset:
        """Convert runs' datasets into a single dataset with coords suitable
        for line plots.

        Data arrays in the set may have the attr *selected* which
        indicates they should be highlighted somehow.

        """
        x_signal = self.ui.x_signal_combobox.currentText()
        y_signal = self.ui.v_signal_combobox.currentText()
        reference_operator = reference_operators[
            self.ui.r_operator_combobox.currentText()
        ]
        r_signal = self.ui.r_signal_combobox.currentText()
        x_label, y_label = self.axis_labels()
        data_vars = {}
        for label, ds in datasets.items():
            reference_selected = self.ui.r_operator_combobox.currentText() != ""
            try:
                arr = ds[y_signal].values
                xdata = ds[x_signal].values
                ref_data = ds[r_signal].values if reference_selected else None
            except KeyError as exc:
                log.info(f"Could not load {exc} for {label}")
                continue
            if reference_selected:
                arr = apply_reference(arr, ref_data, operator=reference_operator)
            arr = self.reduce_nd_array(arr)
            # Apply plotting modifiers (logarithm, gradient, etc)

            if self.ui.invert_checkbox.isChecked():
                arr = 1 / arr
            if self.ui.logarithm_checkbox.isChecked():
                arr = np.log(arr)
            if self.ui.gradient_checkbox.isChecked():
                arr = np.gradient(arr, xdata)
            data_vars[label] = xr.DataArray(
                arr,
                coords={x_signal: xdata},
                name=y_signal,
            )
        new_dataset = xr.Dataset(
            data_vars,
            attrs={
                "coord_label": x_label,
                "data_label": y_label,
            },
        )
        return new_dataset

    def prepare_grid_dataset(
        self,
        dataset: xr.Dataset,
    ) -> xr.DataArray:
        """Convert runs' datasets into a single dataset with coords suitable
        for plotting on a 2D grid.

        """
        grid_shape = dataset.attrs["scan_shape"]
        scan_dims = dataset.attrs["scan_dimensions"]
        ndims = len(scan_dims)
        coords = {
            signal: np.median(
                dataset[signal].values.reshape(grid_shape),
                axis=(i for i in range(ndims) if i != dim_num),
            )
            for dim_num, signal in enumerate(scan_dims)
        }
        v_signal = self.ui.v_signal_combobox.currentText()
        new_dataset = xr.DataArray(
            dataset[v_signal].values.reshape(grid_shape),
            coords=coords,
            name=v_signal,
        )
        return new_dataset

    def prepare_volume_dataset(self, dataset: xr.Dataset) -> xr.DataArray:
        """Convert run dataset into a new dataset with coords."""
        x_signal = self.ui.x_signal_combobox.currentText()
        y_signal = self.ui.v_signal_combobox.currentText()
        vals = dataset[y_signal].values
        extra_coords = {
            f"coord_{idx+1}": np.arange(size) for idx, size in enumerate(vals.shape[1:])
        }
        new_array = xr.DataArray(
            vals,
            coords={x_signal: dataset[x_signal].values, **extra_coords},
            name=y_signal,
        )
        return new_array

    @asyncSlot()
    @cancellable
    async def update_selected_data(self):
        """Load new data for the selected signals and plot it."""
        # Clear the plots for better user experience
        self.ui.lineplot_tab.clear()
        self.ui.gridplot_tab.clear()
        self.ui.frameset_tab.clear()
        # Figure out what we're plotting
        streams = await self.active_streams()
        selected_uid = self.selected_uid()
        uids = self.active_uids()
        xsig = self.x_signal_combobox.currentText()
        ysig = self.v_signal_combobox.currentText()
        r_enabled = self.r_operator_combobox.currentText() != ""
        rsig = self.r_signal_combobox.currentText() if r_enabled else None
        # Need to include independent hints for maps
        _, ihints, _ = await self.data_signals()
        with self.busy_hints(run_widgets=True, run_table=False, filter_widgets=False):
            datasets = await self.db_task(
                self.db.datasets(
                    uids, streams=streams, variables=[xsig, ysig, rsig, *ihints]
                ),
                "update_datasets",
            )
            # # Keep for easier debugging
            # datasets = await self.db.datasets(
            #     uids, stream, xcolumn=xsig, ycolumn=ysig, rcolumn=rsig
            # )
        # 1D line plots
        if len(datasets) > 0:
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.LINE, True)
            line_data = self.prepare_1d_dataset(datasets)
            self.ui.lineplot_tab.plot(line_data)
        else:
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.LINE, False)
        # Grid plot
        selected_dataset = datasets[selected_uid] if selected_uid is not None else None
        scan_shape = (
            selected_dataset.attrs.get("scan_shape", [])
            if selected_dataset is not None
            else []
        )
        if len(scan_shape) >= 2:
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.GRID, True)
            grid_data = self.prepare_grid_dataset(selected_dataset)
            self.ui.gridplot_tab.plot(grid_data)
        else:
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.GRID, False)
            self.ui.gridplot_tab.clear()
        # Volume-based data views
        if selected_dataset is not None and selected_dataset[ysig].ndim == 3:
            volume_data = self.prepare_volume_dataset(datasets[selected_uid])
            self.ui.frameset_tab.plot(volume_data)
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.FRAMES, True)
        else:
            self.ui.detail_tabwidget.setTabEnabled(self.Tabs.FRAMES, False)
            self.ui.frameset_tab.clear()

    def active_uids(self) -> list[str]:
        """UIDS of runs that are checked or selected in the run list."""
        uids = self.checked_uids()
        if (selected := self.selected_uid()) is not None:
            uids.append(selected)
        return uids

    def selected_uid(self) -> str | None:
        """The UID of the run currently selected in the list."""
        uid_col = self._run_col_names.index("UID")
        selected = self.ui.run_tableview.selectedIndexes()
        if len(selected) == 0:
            return None
        uid = selected[0].siblingAtColumn(uid_col).data()
        return uid

    def checked_uids(self) -> list[str]:
        """The UIDs of the runs currently checked in the list."""
        # Get UID's from the selection
        uid_col = self._run_col_names.index("UID")
        cbox_col = 0
        model = self.runs_model
        uids = [
            model.item(row_idx, uid_col).text()
            for row_idx in range(self.runs_model.rowCount())
            if model.item(row_idx, cbox_col).checkState() == Qt.Checked
        ]
        return list(dict.fromkeys(uids))

    def filters(self, *args):
        new_filters = {
            "plan": self.ui.filter_plan_combobox.currentText(),
            "sample": self.ui.filter_sample_combobox.currentText(),
            "formula": self.ui.filter_formula_combobox.currentText(),
            "scan": self.ui.filter_scan_combobox.currentText(),
            "edge": self.ui.filter_edge_combobox.currentText(),
            "exit_status": self.ui.filter_exit_status_combobox.currentText(),
            "user": self.ui.filter_user_combobox.currentText(),
            "proposal": self.ui.filter_proposal_combobox.currentText(),
            "esaf": self.ui.filter_esaf_combobox.currentText(),
            "beamline": self.ui.filter_beamline_combobox.currentText(),
            "full_text": self.ui.filter_full_text_lineedit.text(),
        }
        # Special handling for the time-based filters
        if self.ui.filter_after_checkbox.isChecked():
            after = self.ui.filter_after_datetimeedit.dateTime().toSecsSinceEpoch()
            new_filters["after"] = after
        if self.ui.filter_before_checkbox.isChecked():
            before = self.ui.filter_before_datetimeedit.dateTime().toSecsSinceEpoch()
            new_filters["before"] = before
        # Limit the search to standards only
        if self.ui.filter_standards_checkbox.isChecked():
            new_filters["standards_only"] = True
        # Only include values that were actually filled in
        null_values = ["", False]
        new_filters = {k: v for k, v in new_filters.items() if v not in null_values}
        return new_filters

    def load_models(self):
        # Set up the model
        self.runs_model = QStandardItemModel()
        # Add the model to the UI element
        self.ui.run_tableview.setModel(self.runs_model)
