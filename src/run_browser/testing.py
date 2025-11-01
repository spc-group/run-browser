from contextlib import contextmanager


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
