# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""V1 (verl-project/verl main) compatibility wrappers for the Arctic backend.

Kept in a separate subpackage so V0 (Snowflake-AI-Research/verl) users can
continue to import ``arctic_platform.integrations.verl`` without paying for
V1's transfer_queue / TransferQueue import overhead, and so V1 users can plug
in via the same plugin bootstrap without pulling in V0-only imports.
"""
