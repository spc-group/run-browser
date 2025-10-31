import argparse
import asyncio
import logging
import sys

import httpx
from qasync import QApplication, QEventLoop

from run_browser.main_window import RunBrowserMainWindow

log = logging.getLogger("run_browser")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run-browser",
        description="Live viewer for seeing SPC-group data in the database",
    )
    parser.add_argument(
        "--merge-streams",
        action="store_true",
        help="Enable experimental support for plotting signals from different streams.",
    )
    parser.add_argument(
        "--plot-spectra",
        action="store_true",
        help="Enable experimental support for plotting area detectors as spectra.",
    )

    args, extra_args = parser.parse_known_args(sys.argv)

    logging.basicConfig(level=logging.INFO)

    app = QApplication(extra_args)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    main_window = RunBrowserMainWindow(
        merge_streams=args.merge_streams, plot_spectra=args.plot_spectra
    )
    main_window.show()

    async def start(window, event):
        try:
            await window.change_catalog()
        except httpx.ConnectTimeout as ex:
            log.exception(ex)
        await event.wait()

    # for 3.11 or older use qasync.run instead of asyncio.run
    asyncio.run(start(main_window, app_close_event), loop_factory=QEventLoop)


if __name__ == "__main__":
    sys.exit(main())
