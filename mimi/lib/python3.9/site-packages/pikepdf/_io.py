# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from contextlib import contextmanager, suppress
from io import TextIOBase
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Generator


def check_stream_is_usable(stream: IO) -> None:
    """Check that a stream is seekable and binary."""
    if isinstance(stream, TextIOBase):
        raise TypeError("stream must be binary (no transcoding) and seekable")


def check_different_files(file1: str | PathLike, file2: str | PathLike) -> None:
    """Check that two files are different."""
    with suppress(FileNotFoundError):
        if Path(file1) == Path(file2) or Path(file1).samefile(Path(file2)):
            raise ValueError(
                "Cannot overwrite input file. Open the file with "
                "pikepdf.open(..., allow_overwriting_input=True) to "
                "allow overwriting the input file."
            )


@contextmanager
def atomic_overwrite(filename: Path) -> Generator[IO[bytes], None, None]:
    """Atomically ovewrite a file.

    If the destination file does not exist, it is created. If writing fails,
    the destination file is deleted.

    If the destination file does exist, a temporaryfile is created in the same
    directory, and data is written to that file. If writing succeeds, the temporary
    file is renamed to the destination file. If writing fails, the temporary file
    is deleted and the original destination file is left untouched.
    """
    try:
        # Try to create the file using exclusive creation mode
        stream = filename.open("xb")
    except FileExistsError:
        pass
    else:
        # We were able to create the file, so we can use it directly
        try:
            with stream:
                yield stream
        except Exception:
            # ...but if an error occurs while using it, clean up
            with suppress(FileNotFoundError):
                filename.unlink()
            raise
        return

    # If we get here, the file already exists. Use a temporary file, then rename
    # it to the destination file if we succeed. Destination file is not touched
    # if we fail.

    with filename.open("ab") as stream:
        pass  # Confirm we will be able to write to the indicated destination

    with NamedTemporaryFile(
        dir=filename.parent, prefix=f".pikepdf.{filename.name}", delete=False
    ) as tf:
        try:
            yield tf
        except Exception:
            tf.close()
            Path(tf.name).unlink()
            raise
        tf.flush()
        tf.close()
        Path(tf.name).replace(filename)
