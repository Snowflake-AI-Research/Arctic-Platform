# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from arctic_platform.correctness.harness.report import write


def test_reports_are_timestamped_and_latest_links_move_without_deleting_history(tmp_path: Path) -> None:
    write([], {}, tmp_path, timestamp="2026-09-24-15-50")
    write([], {}, tmp_path, timestamp="2026-09-24-15-51")

    assert (tmp_path / "report-2026-09-24-15-50.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-50.json").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.json").is_file()
    assert (tmp_path / "report.md").is_symlink()
    assert (tmp_path / "report.json").is_symlink()
    assert (tmp_path / "report.md").readlink() == Path("report-2026-09-24-15-51.md")
    assert (tmp_path / "report.json").readlink() == Path("report-2026-09-24-15-51.json")
