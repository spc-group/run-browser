import asyncio
import sys

from qasync import QApplication, QEventLoop

from run_browser.main_window import RunBrowserMainWindow


def main(argv=None):
    app = QApplication(sys.argv)

    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    main_window = RunBrowserMainWindow()
    main_window.show()

    async def start(window, event):
        await window.change_catalog()
        await event.wait()

    # for 3.11 or older use qasync.run instead of asyncio.run
    asyncio.run(start(main_window, app_close_event), loop_factory=QEventLoop)


if __name__ == "__main__":
    sys.exit(main())
