"""Read-only terminal UI for Cortex Training logs and scheduling events.

The formatting helpers import cleanly without ``textual``; the app and entry
point use the package's standard TUI dependencies.
"""

from cortex_training.tui.format import format_event, format_log_entry  # noqa: F401
