from __future__ import annotations

from radar_sim.logging_utils import write_run_log


def pytest_terminal_summary(terminalreporter, exitstatus: int, config) -> None:
    stats = terminalreporter.stats
    collected = getattr(terminalreporter, "_numcollected", 0)
    lines = [
        "Pytest session report",
        f"Exit status: {exitstatus}",
        f"Collected tests: {collected}",
        f"Command: {' '.join(str(arg) for arg in config.invocation_params.args)}",
    ]

    for key in ("passed", "failed", "skipped", "error", "xfailed", "xpassed"):
        count = len(stats.get(key, []))
        if count:
            lines.append(f"{key}: {count}")

    if stats.get("failed"):
        lines.append("")
        lines.append("Failed tests:")
        for report in stats["failed"]:
            lines.append(f"  - {report.nodeid}")

    if stats.get("error"):
        lines.append("")
        lines.append("Errored tests:")
        for report in stats["error"]:
            lines.append(f"  - {report.nodeid}")

    path = write_run_log(run_type="pytest", name="tests", content="\n".join(lines))
    terminalreporter.write_line(f"Saved test log: {path}")
