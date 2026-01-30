import asyncio
import datetime as dt
import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import httpx
import numpy as np
import pandas as pd
import xarray as xr
from tiled import queries
from tiled.client.container import Container

log = logging.getLogger(__name__)


class Merged:
    def __repr__(self):
        return "<Merge streams>"

    def __str__(self):
        return "<Merged>"


MERGED = Merged()


def dimension_hints(start_doc, streams: Sequence[str]):
    """Get scan dimension information from a run's start doc."""
    dim_md = start_doc.get("hints", {}).get("dimensions", [])
    return [sig for signals, strm in dim_md if strm in streams for sig in signals]


@dataclass(frozen=True, eq=True)
class DataSignal:
    name: str
    stream: str | None
    path_parts: Sequence[str]
    is_scan_dimension: bool = False
    is_hinted: bool = False


class DatabaseWorker:
    selected_runs: Sequence = []
    profile: str = ""
    catalog: Container
    stream_prefix: str = ""

    def __init__(self, stream_prefix: str):
        self.stream_prefix = stream_prefix
        super().__init__()

    async def _stream_names(self, run, prefix: str = "") -> list[str]:
        streams_node = run
        streams = [f"{prefix}{key}" async for key in streams_node.keys()]
        # Check if we have an old run with the "streams" namespace
        if streams == ["streams"]:
            return await self._stream_names(await run["streams"], prefix="streams/")
        return streams

    async def stream_names(self, uids: Sequence[str]) -> list[str]:
        runs = self.runs(uids)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._stream_names(run)) async for run in runs.values()
            ]
        # Flatten the lists
        all_streams = [task.result() for task in tasks]
        streams = [stream for streams in all_streams for stream in streams]
        return list(set(streams))

    async def _data_signals(self, run, streams: Sequence[str]) -> list[DataSignal]:
        """Get metadata about the available data signals for a single run.

        Parameters
        ==========
        run:
          Tiled container for the target run.
        streams:
          Names of the streams to retrieve data keys for.

        """

        async def get_stream_data_signals(run, stream_name: str):
            stream = await run[f"{self.stream_prefix}{stream_name}"]
            stream_md = stream.metadata.get("data_keys", {})
            ihints, dhints = await self._get_hints(run, streams=[stream_name])
            signals = [
                DataSignal(
                    name=name,
                    path_parts=stream.path_parts,
                    stream=stream_name,
                    is_scan_dimension=name in ihints,
                    is_hinted=name in dhints,
                )
                for name, data_key in stream_md.items()
            ]
            return signals

        data_signals = await asyncio.gather(
            *[get_stream_data_signals(run, stream) for stream in streams]
        )
        # Flatten the nested lists
        return [sig for signals in data_signals for sig in signals]

    async def data_signals(
        self, uids: Sequence[str], streams: Sequence[str]
    ) -> list[DataSignal]:
        """Get metadata about the available data signals for a set of runs.

        Parameters
        ==========
        run:
          Tiled container for the target run.
        streams:
          Names of the streams to retrieve data keys for.

        """
        runs = self.runs(uids)

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._data_signals(run, streams))
                async for run in runs.values()
            ]
        signals = [signal for task in tasks for signal in task.result()]
        # Add a sequence num key since it is not included in the stream
        signals.append(DataSignal(name="seq_num", stream=None, path_parts=[]))
        return signals

    def runs(self, uids: Sequence[str]) -> Container:
        return self.catalog.search(queries.In("start.uid", uids))

    async def dataframes(self, uids: Sequence[str], stream: str) -> dict:
        """Return the internal dataframes for selected runs as {uid: dataframe}."""
        runs = self.runs(uids)

        async def get_data_frame(run):
            try:
                node = await run[f"{self.stream_prefix}{stream}/internal"]
            except KeyError as exc:
                # log.(exc)
                return xr.Dataset({})
            else:
                return await node.read()

        async with asyncio.TaskGroup() as tg:
            tasks = {
                uid: tg.create_task(get_data_frame(run))
                async for uid, run in runs.items()
            }
        results = {uid: task.result() for uid, task in tasks.items()}
        return results

    async def datasets(
        self,
        uids: Sequence[str],
        streams: Sequence[str],
        variables: Sequence[str],
    ) -> dict[str, xr.DataArray]:
        """Return data for selected runs as {uid: xarray.Dataset}."""
        if len(uids) == 0:
            return {}
        runs = self.runs(uids)

        async def get_xarray(run, stream: str):
            start_doc = run.metadata.get("start", {})
            dimensions = dimension_hints(start_doc, streams=[stream])
            node = await run[f"{self.stream_prefix}{stream}"]
            if not hasattr(node, "read"):
                log.warning(f"Node {node} cannot be read.")
                ds = xr.Dataset({})
            ds = await node.read(variables=variables)
            ds.attrs["scan_dimensions"] = dimensions
            if "shape" in start_doc:
                ds.attrs["scan_shape"] = start_doc["shape"]
            return ds

        async with asyncio.TaskGroup() as tg:
            tasks = {
                uid: tg.create_task(get_xarray(run, stream))
                for stream in streams
                async for uid, run in runs.items()
            }
        results = {uid: task.result() for uid, task in tasks.items()}
        return results

    def filtered_runs(self, filters: Mapping):
        log.debug(f"Filtering nodes: {filters}")
        filter_params = {
            # filter_name: (query type, metadata key)
            "plan": (queries.Eq, "start.plan_name"),
            "sample": (queries.Contains, "start.sample_name"),
            "formula": (queries.Contains, "start.sample_formula"),
            "scan": (queries.Contains, "start.scan_name"),
            "edge": (queries.Contains, "start.edge"),
            "exit_status": (queries.Eq, "stop.exit_status"),
            "user": (queries.Contains, "start.proposal_users"),
            "proposal": (queries.Eq, "start.proposal_id"),
            "esaf": (queries.Eq, "start.esaf_id"),
            "beamline": (queries.Eq, "start.beamline"),
            "before": (partial(queries.Comparison, "le"), "stop.time"),
            "after": (partial(queries.Comparison, "ge"), "start.time"),
            "full_text": (queries.FullText, ""),
            "standards_only": (queries.Eq, "start.is_standard"),
        }
        # Apply filters
        runs = self.catalog
        for filter_name, filter_value in filters.items():
            if filter_name not in filter_params:
                continue
            Query, md_name = filter_params[filter_name]
            if Query is queries.FullText:
                query = Query(filter_value)
            else:
                query = Query(md_name, filter_value)
            runs = runs.search(query)
        return runs

    async def distinct_fields(self) -> AsyncGenerator[tuple[str, list], None]:
        """Get distinct metadata fields for filterable metadata."""
        # Some of these are disabled since they take forever
        # (could be re-enabled when switching to postgres)
        target_fields = [
            "start.plan_name",
            "start.sample_name",
            "start.sample_formula",
            "start.scan_name",
            "start.edge",
            "stop.exit_status",
            "start.proposal_id",
            "start.esaf_id",
            "start.beamline",
        ]
        # Get fields from the database
        try:
            distinct = await self.catalog.distinct(*target_fields)
        except httpx.ReadTimeout as exc:
            log.exception(exc)
            return
        for field_name, values in (distinct)["metadata"].items():
            yield field_name, [info["value"] for info in values]

    async def load_all_runs(self, filters: Mapping = {}):
        all_runs = []
        runs = self.filtered_runs(filters=filters)
        async for run in runs.values():
            # Get meta-data documents
            metadata = run.metadata
            start_doc = metadata.get("start")
            if start_doc is None:
                log.debug(f"Skipping run with no start doc: {run.path}")
                continue
            stop_doc = metadata.get("stop")
            if stop_doc is None:
                stop_doc = {}
            # Get a human-readable timestamp for the run
            timestamp = start_doc.get("time")
            if timestamp is None:
                run_datetime = ""
            else:
                run_datetime_ = dt.datetime.fromtimestamp(timestamp)
                run_datetime = run_datetime_.strftime("%Y-%m-%d %H:%M:%S")
            # Get the X-ray edge scanned
            edge = start_doc.get("edge")
            E0 = start_doc.get("E0")
            E0_str = "" if E0 is None else str(E0)
            if edge and E0:
                edge_str = f"{edge} ({E0} eV)"
            elif edge:
                edge_str = edge
            elif E0:
                edge_str = E0_str
            else:
                edge_str = ""
            # Combine sample and formula together
            sample_name = start_doc.get("sample_name", "")
            formula = start_doc.get("sample_formula", "")
            if sample_name and formula:
                sample_str = f"{sample_name} ({formula})"
            else:
                sample_str = sample_name or formula
            # Build the table item
            # Get sample data from: dd80f432-c849-4749-a8f3-bdeec6f9c1f0
            run_data = OrderedDict(
                plan_name=start_doc.get("plan_name", ""),
                sample_name=sample_str,
                scan_name=start_doc.get("scan_name", ""),
                edge=edge_str,
                exit_status=stop_doc.get("exit_status", ""),
                run_datetime=run_datetime,
                uid=start_doc.get("uid", ""),
            )
            all_runs.append(run_data)
        return all_runs

    async def _get_hints(self, run, streams: Sequence[str]):
        run_md = run.metadata
        # Get hints for the independent (X)
        independent = dimension_hints(run_md.get("start", {}), streams=streams)

        # Get hints for the dependent (Y) axes
        async def get_stream_hints(run, stream):
            stream_md = (await run[f"{self.stream_prefix}{stream}"]).metadata
            stream_md.get("hints", {})
            return [
                hint
                for dev_hints in stream_md.get("hints", {}).values()
                for hint in dev_hints.get("fields", [])
            ]

        aws = [get_stream_hints(run, stream) for stream in streams]
        dependent = [sig for sigs in await asyncio.gather(*aws) for sig in sigs]
        return independent, dependent

    async def hints(
        self, uids: Sequence[str], streams: Sequence[str] = "primary"
    ) -> tuple[set, set]:
        """Get hints for this stream, as two lists.

        (*independent_hints*, *dependent_hints*)

        *independent_hints* are those operated by the experiment,
         while *dependent_hints* are those measured as a result.
        """
        runs = await asyncio.gather(*(self.catalog[uid] for uid in uids))
        all_hints = await asyncio.gather(
            *(self._get_hints(run, streams=streams) for run in runs)
        )
        # Flatten arrays
        ihints_: Sequence[str]
        dhints_: Sequence[str]
        try:
            ihints_, dhints_ = zip(*all_hints)
        except ValueError:
            ihints_, dhints_ = [], []
        ihints: set[str] = {hint for hints in ihints_ for hint in hints}
        dhints: set[str] = {hint for hints in dhints_ for hint in hints}
        return ihints, dhints

    async def signal_names(self, stream: str, *, hinted_only: bool = False):
        """Get a list of valid signal names (data columns) for selected runs.

        Parameters
        ==========
        stream
          The Tiled stream name to fetch.
        hinted_only
          If true, only signals with the kind="hinted" parameter get
          picked.

        """
        xsignals, ysignals = [], []
        for run in self.selected_runs:
            if hinted_only:
                xsig, ysig = await run.hints(stream=stream)
            else:
                df = await run.data(stream=stream)
                xsig = ysig = df.columns
            xsignals.extend(xsig)
            ysignals.extend(ysig)
        # Remove duplicates
        xsignals = list(dict.fromkeys(xsignals))
        ysignals = list(dict.fromkeys(ysignals))
        return list(xsignals), list(ysignals)

    async def metadata(self, uids: Sequence[str]) -> dict[str, dict]:
        """Get all metadata for the selected runs in one big dictionary."""
        return {uid: run.metadata async for uid, run in self.runs(uids).items()}

    async def load_selected_runs(self, uids):
        assert False, "Deprecated"
        # Prepare the query for finding the runs
        uids = list(dict.fromkeys(uids))
        # Retrieve runs from the database
        runs = await asyncio.gather(*(self.catalog[uid] for uid in uids))
        self.selected_runs = runs
        return self.selected_runs

    async def images(self, signal: str, stream: str):
        """Load the selected runs as 2D or 3D images suitable for plotting."""
        images = OrderedDict()
        for idx, run in enumerate(self.selected_runs):
            # Load datasets from the database
            try:
                image = await run.__getitem__(signal, stream=stream)
            except KeyError as exc:
                log.exception(exc)
            else:
                images[run.uid] = image
        return images

    async def all_signals(self, stream: str, *, hinted_only=False) -> dict:
        """Produce dataframes with all signals for each run.

        The keys of the dictionary are the labels for each curve, and
        the corresponding value is a pandas dataframe with the scan data.

        """
        xsignals, ysignals = await self.signal_names(
            hinted_only=hinted_only, stream=stream
        )
        # Build the dataframes
        dfs = OrderedDict()
        for run in self.selected_runs:
            # Get data from the database
            df = await run.data(signals=xsignals + ysignals, stream=stream)
            dfs[run.uid] = df
        return dfs

    async def signals(
        self,
        x_signal,
        y_signal,
        r_signal=None,
        *,
        stream: str,
        use_log=False,
        use_invert=False,
        use_grad=False,
        uids: Sequence[str] | None = None,
    ) -> Mapping:
        """Produce a dictionary with the 1D datasets for plotting.

        The keys of the dictionary are the labels for each curve, and
        the corresponding value is a pandas dataset with the data for
        each signal.

        Parameters
        ==========
        uids
          If not ``None``, only runs with UIDs listed in this
          parameter will be included.

        """
        # Check for sensible inputs
        use_reference = r_signal is not None
        if "" in [x_signal, y_signal] or (use_reference and r_signal == ""):
            msg = (
                f"Empty signal name requested: x={repr(x_signal)}, y={repr(y_signal)},"
                f" r={repr(r_signal)}"
            )
            log.debug(msg)
            raise ValueError(msg)
        signals = [x_signal, y_signal]
        if use_reference:
            signals.append(r_signal)
        # Remove duplicates
        signals = list(dict.fromkeys(signals).keys())
        # Build the dataframes
        dfs = OrderedDict()
        for run in self.selected_runs:
            # Check that the UID matches
            if uids is not None and run.uid not in uids:
                break
            # Get data from the database
            df = await run.data(signals=signals, stream=stream)
            # Check for missing signals
            missing_x = x_signal not in df.columns and df.index.name != x_signal
            missing_y = y_signal not in df.columns
            missing_r = r_signal not in df.columns
            if missing_x or missing_y or (use_reference and missing_r):
                log.warning(
                    f"Could not find signals {x_signal=}, {y_signal=} and {r_signal=} in {df.columns}"
                )
                continue
            # Apply transformations
            if use_reference:
                df[y_signal] /= df[r_signal]
            if use_log:
                df[y_signal] = np.log(df[y_signal])
            if use_invert:
                df[y_signal] *= -1
            if use_grad:
                df[y_signal] = np.gradient(df[y_signal], df[x_signal])
            series = pd.Series(df[y_signal].values, index=df[x_signal].values)
            dfs[run.uid] = series
        return dfs

    async def export_runs(self, filenames: Sequence[str], formats: Sequence[str]):
        for filename, run, format in zip(filenames, self.selected_runs, formats):
            await run.export(filename, format=format)


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
