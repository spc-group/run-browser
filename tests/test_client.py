import re

import httpx
import pandas as pd
import pytest
import pytest_asyncio
import xarray as xr

from run_browser.client import DatabaseWorker

run_metadata_urls = re.compile(
    r"^http://localhost:8000/api/v1/metadata/([a-z]+)%2F([-a-z0-9]+)$"
)


@pytest.fixture()
def run_metadata_api(httpx_mock):
    def respond_with_metadata(request: httpx.Request):
        url = str(request.url)
        match = run_metadata_urls.match(url)
        if match is None:
            raise ValueError(f"Could not match URL {request.url}")
        else:
            catalog, uid = match.groups()
        md = {
            "data": {
                "attributes": {
                    "metadata": {
                        "start": {
                            "uid": uid,
                        }
                    }
                }
            }
        }
        return httpx.Response(
            status_code=200,
            json=md,
        )

    httpx_mock.add_callback(
        callback=respond_with_metadata,
        url=run_metadata_urls,
        is_reusable=True,
    )


@pytest_asyncio.fixture()
async def worker(tiled_client):
    worker = DatabaseWorker(stream_prefix="")
    worker.catalog = tiled_client
    return worker


def md_to_json(metadata):
    response = {"data": {"attributes": {"metadata": metadata}}}
    return response


@pytest.mark.asyncio
async def test_data_signals(worker):
    uids = [
        "85573831-f4b4-4f64-b613-a6007bf03a8d",
        "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    ]
    data_signals = await worker.data_signals(uids, streams=["primary"])
    expected_signals = sorted(
        [
            "CdnI0_net_counts",
            "CdnIPreKb_net_counts",
            "CdnIt_net_counts",
            "I0_net_counts",
            "Ipre_KB_net_counts",
            "Ipreslit_net_counts",
            "It_net_counts",
            "aerotech_vert",
            "aerotech_horiz",
            "It-mcs-scaler-channels-3-net_count",
            "ge_8element",
            "ge_8element-deadtime_factor",
            "energy_energy",
            "I0-net_count",
            "seq_num",
        ]
    )
    signal_names = sorted([sig.name for sig in data_signals])
    assert signal_names == expected_signals
    dimension_signal = {sig.name: sig for sig in data_signals}["aerotech_horiz"]
    assert dimension_signal.is_scan_dimension


@pytest.mark.asyncio
async def test_data_signals_merged_streams(worker):
    """Can we combine multiple streams into a single datakey."""
    uids = ["fly_scan"]
    data_signals = await worker.data_signals(uids, streams=["It", "I0"])
    expected_signals = sorted(
        [
            "I0-net_count",
            "It-net_count",
            "seq_num",
        ]
    )
    signal_names = sorted([sig.name for sig in data_signals])
    assert signal_names == expected_signals


@pytest.mark.asyncio
async def test_metadata(worker):
    uids = [
        "85573831-f4b4-4f64-b613-a6007bf03a8d",
        "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    ]
    md = await worker.metadata(uids)
    assert list(md.keys()) == uids
    assert md[uids[0]]["start"]["uid"] == uids[0]


@pytest.mark.asyncio
async def test_data_frames(worker, tiled_client):
    uids = [
        "85573831-f4b4-4f64-b613-a6007bf03a8d",
        "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    ]
    data_frames = await worker.dataframes(uids, "primary")
    # Check the results
    assert isinstance(data_frames["85573831-f4b4-4f64-b613-a6007bf03a8d"], pd.DataFrame)


@pytest.mark.asyncio
async def test_datasets(worker, tiled_client):
    uids = ["xarray_run"]
    arrays = await worker.datasets(
        uids,
        streams=["primary"],
        variables=["mono-energy", "It-net_count", "I0-net_count"],
    )
    # Check the results
    ds = arrays["xarray_run"]
    assert isinstance(ds, xr.Dataset)
    assert "I0-net_count" in ds.variables
    assert "mono-energy" in ds.variables
    assert "It-net_count" in ds.variables
    assert "other_signal" not in ds.variables
    assert ds.attrs["scan_dimensions"] == ["mono-energy", "aerotech-horiz"]
    assert ds.attrs["scan_shape"] == [2, 2]


@pytest.mark.asyncio
async def test_hints(worker):
    uids = [
        "85573831-f4b4-4f64-b613-a6007bf03a8d",
        "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
    ]
    ihints, dhints = await worker.hints(uids, ["primary"])
    assert ihints == {"aerotech_vert", "aerotech_horiz"}


@pytest.mark.asyncio
async def test_filter_runs(worker):
    runs = await worker.load_all_runs(filters={"plan": "xafs_scan"})
    # Check that the runs were filtered
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_distinct_fields(worker):
    distinct_fields = [field async for field in worker.distinct_fields()]
    keys, fields = zip(*distinct_fields)
    # Check that the dictionary has the right structure
    for key in ["start.plan_name"]:
        assert key in keys


@pytest.mark.asyncio
async def test_stream_names(worker):
    stream_names = await worker.stream_names(["85573831-f4b4-4f64-b613-a6007bf03a8d"])
    assert sorted(stream_names) == ["baseline", "primary"]


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
