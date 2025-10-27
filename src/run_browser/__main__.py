import argparse
import asyncio
import sys

from qasync import QApplication, QEventLoop

from run_browser.main_window import RunBrowserMainWindow


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

    args, extra_args = parser.parse_known_args(sys.argv)

    app = QApplication(extra_args)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    main_window = RunBrowserMainWindow(merge_streams=args.merge_streams)
    main_window.show()

    async def start(window, event):
        await window.change_catalog()
        await event.wait()

    # for 3.11 or older use qasync.run instead of asyncio.run
    asyncio.run(start(main_window, app_close_event), loop_factory=QEventLoop)


if __name__ == "__main__":
    sys.exit(main())
