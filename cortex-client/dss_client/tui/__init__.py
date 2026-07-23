"""Read-only terminal UI for Neutrino logs and ZMD scheduling events.

The formatting helpers import cleanly without ``textual``; the app and entry
point require the optional ``dss-client[tui]`` extra.
"""

from dss_client.tui.format import format_event, format_log_entry  # noqa: F401
