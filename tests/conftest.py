import numpy as np
import pandas as pd
import pytest_asyncio
import xarray as xr
from tiled.adapters.dataframe import DataFrameAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.adapters.xarray import DatasetAdapter
from tiled.client import from_context_async
from tiled.client.context import Context
from tiled.server.app import build_app


def md_to_json(metadata):
    response = {"data": {"attributes": {"metadata": metadata}}}
    return response


tree = MapAdapter(
    {
        "85573831-f4b4-4f64-b613-a6007bf03a8d": MapAdapter(
            {
                "baseline": MapAdapter({}),
                "primary": MapAdapter(
                    {
                        "internal": DataFrameAdapter.from_pandas(
                            pd.DataFrame(
                                {
                                    "x": 1 * np.ones(10),
                                    "y": 2 * np.ones(10),
                                    "z": 3 * np.ones(10),
                                }
                            ),
                            npartitions=3,
                        ),
                    },
                    metadata={
                        "data_keys": {
                            "I0-net_count": {
                                "dtype": "number",
                            },
                            "ge_8element": {},
                            "ge_8element-deadtime_factor": {},
                            "energy_energy": {},
                            "aerotech_horiz": {},
                            "aerotech_vert": {},
                            "CdnI0_net_counts": {},
                            "CdnIPreKb_net_counts": {},
                            "CdnIt_net_counts": {},
                            "I0_net_counts": {},
                            "Ipre_KB_net_counts": {},
                            "Ipreslit_net_counts": {},
                            "I0_net_counts": {},
                            "Ipre_KB_net_counts": {},
                            "Ipreslit_net_counts": {},
                            "It_net_counts": {},
                        },
                        "hints": {
                            "Ipreslit": {"fields": ["Ipreslit_net_counts"]},
                            "CdnIPreKb": {"fields": ["CdnIPreKb_net_counts"]},
                            "I0": {"fields": ["I0_net_counts"]},
                            "CdnIt": {"fields": ["CdnIt_net_counts"]},
                            "aerotech_vert": {"fields": ["aerotech_vert"]},
                            "aerotech_horiz": {"fields": ["aerotech_horiz"]},
                            "Ipre_KB": {"fields": ["Ipre_KB_net_counts"]},
                            "CdnI0": {"fields": ["CdnI0_net_counts"]},
                            "It": {"fields": ["It_net_counts"]},
                        },
                    },
                ),
            },
            metadata={
                "start": {
                    "uid": "85573831-f4b4-4f64-b613-a6007bf03a8d",
                    "hints": {
                        "dimensions": [
                            [["aerotech_vert"], "primary"],
                            [["aerotech_horiz"], "primary"],
                        ],
                    },
                    "plan_name": "scan",
                    "beamline": "25-ID-C",
                    "sample_name": "Miso",
                    "sample_formula": "Ms",
                    "scan_name": "Dinner",
                    "proposal_id": "000001",
                    "esaf_id": "123456",
                },
                "stop": {
                    "exit_status": "success",
                },
            },
        ),
        "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f": MapAdapter(
            {
                "primary": MapAdapter(
                    {
                        "internal": DataFrameAdapter.from_pandas(
                            pd.DataFrame(
                                {
                                    "x": 1 * np.ones(10),
                                    "y": 2 * np.ones(10),
                                    "z": 3 * np.ones(10),
                                }
                            ),
                            npartitions=3,
                        ),
                    },
                    metadata={
                        "data_keys": {
                            "It-mcs-scaler-channels-3-net_count": {
                                "dtype": "number",
                            }
                        },
                        "hints": {
                            "Ipreslit": {"fields": ["Ipreslit_net_counts"]},
                            "CdnIPreKb": {"fields": ["CdnIPreKb_net_counts"]},
                            "I0": {"fields": ["I0_net_counts"]},
                            "CdnIt": {"fields": ["CdnIt_net_counts"]},
                            "aerotech_vert": {"fields": ["aerotech_vert"]},
                            "aerotech_horiz": {"fields": ["aerotech_horiz"]},
                            "Ipre_KB": {"fields": ["Ipre_KB_net_counts"]},
                            "CdnI0": {"fields": ["CdnI0_net_counts"]},
                            "It": {"fields": ["It_net_counts"]},
                        },
                    },
                ),
            },
            metadata={
                "start": {
                    "uid": "7d1daf1d-60c7-4aa7-a668-d1cd97e5335f",
                    "hints": {
                        "dimensions": [
                            [["aerotech_vert"], "primary"],
                            [["aerotech_horiz"], "primary"],
                        ],
                    },
                    "plan_name": "xafs_scan",
                    "edge": "Ni-K",
                },
                "stop": {
                    "exit_status": "success",
                },
            },
        ),
        "xarray_run": MapAdapter(
            {
                "primary": DatasetAdapter.from_dataset(
                    xr.Dataset(
                        {
                            "I0-net_count": (
                                "mono-energy",
                                [200, 300, 250, 350],
                            ),
                            "It-net_count": (
                                "mono-energy",
                                [200, 300, 250, 350],
                            ),
                            "other_signal": (
                                "mono-energy",
                                [10, 122, 13345, 159832],
                            ),
                            "mono-energy": ("mono-energy", [0, 1, 2, 3]),
                            "aerotech-horiz": (
                                "mono-energy",
                                [-50, -20, 10, 40],
                            ),
                        }
                    ),
                ),
            },
            metadata={
                "start": {
                    "uid": "xarray_run",
                    "shape": (2, 2),
                    "hints": {
                        "dimensions": [
                            [["mono-energy"], "primary"],
                            [["aerotech-horiz"], "primary"],
                        ],
                    },
                },
            },
        ),
        "xarray_line_scan": MapAdapter(
            {
                "primary": DatasetAdapter.from_dataset(
                    xr.Dataset(
                        {
                            "I0-net_count": (
                                "mono-energy",
                                [200, 300, 250, 350],
                            ),
                            "It-net_count": (
                                "mono-energy",
                                [200, 300, 250, 350],
                            ),
                            "other_signal": (
                                "mono-energy",
                                [10, 122, 13345, 159832],
                            ),
                            "mono-energy": ("mono-energy", [0, 1, 2, 3]),
                            "aerotech-horiz": (
                                "mono-energy",
                                [-50, -20, 10, 40],
                            ),
                        }
                    ),
                ),
            },
            metadata={
                "start": {
                    "uid": "xarray_line_scan",
                    # Has dimension hints, but no shape
                    "hints": {
                        "dimensions": [
                            [["mono-energy"], "primary"],
                            [["aerotech-horiz"], "primary"],
                        ],
                    },
                },
            },
        ),
        "fly_scan": MapAdapter(
            {
                "It": MapAdapter(
                    {},
                    metadata={
                        "data_keys": {
                            "It-net_count": {
                                "dtype": "number",
                            }
                        },
                        "hints": {},
                    },
                ),
                "I0": MapAdapter(
                    {},
                    metadata={
                        "data_keys": {
                            "I0-net_count": {
                                "dtype": "number",
                            }
                        },
                        "hints": {},
                    },
                ),
            },
            metadata={
                "start": {
                    "uid": "fly_scan",
                    "hints": {
                        "dimensions": [
                            [["aerotech-horiz"], "primary"],
                        ],
                    },
                },
            },
        ),
        # Old run with the `streams/` namespace
        "old_streams_scan": MapAdapter(
            {
                "streams": MapAdapter(
                    {
                        "primary": MapAdapter({}),
                        "baseline": MapAdapter({}),
                    },
                ),
            },
            metadata={
                "start": {
                    "uid": "old_streams_scan",
                }
            },
        ),
    }
)


@pytest_asyncio.fixture()
async def tiled_client():
    async with Context.from_app(build_app(tree), awaitable=True) as context:
        client = await from_context_async(context)
        yield client
