# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""pikepdf global settings."""

from __future__ import annotations

from pikepdf._core import (
    get_decimal_precision,
    set_decimal_precision,
    set_flate_compression_level,
)

__all__ = [
    'get_decimal_precision',
    'set_decimal_precision',
    'set_flate_compression_level',
]
