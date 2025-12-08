import argparse
import asyncio
import logging
import sys

import httpx
import stamina
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
        "--debug",
        action="store_true",
        help="Enable debugging mode. Disables stamin retries.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", help="Increase verbosity level.", default=0
    )
    # Temporary bridge to get to a Tiled version that doesn't use the "streams/" node
    parser.add_argument(
        "--stream-prefix",
        default="streams/",
        help="Older versions of Tiled use an intermediate 'streams/' node for each run.",
    )

    args, extra_args = parser.parse_known_args(sys.argv)

    if args.debug:
        stamina.set_active(False)

    # Set up logging
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    max_verbosity = len(log_levels) - 1
    log_level = log_levels[min(args.verbose, max_verbosity)]
    logging.basicConfig(level=log_level)

    app = QApplication(extra_args)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    main_window = RunBrowserMainWindow(
        merge_streams=args.merge_streams, stream_prefix=args.stream_prefix
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
