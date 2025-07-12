"""
Script to generate and update the `RcParamKeyType` Literal type alias in
the `matplotlib.typing` module based on the current set of rcParams keys.

This automates keeping the typing definitions up to date with Matplotlib's
runtime rcParams keys, improving type safety and IDE autocomplete.

Note:
    This script overwrites the block between markers:
    # --- START GENERATED RcParamKeyType ---
    and
    # --- END GENERATED RcParamKeyType ---
    in the typing file with the newly generated Literal block.
"""

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid GUI errors during import
import re

# Path to the typing file to update — adjust if your directory structure differs
typing_file = "lib/matplotlib/typing.py"

def generate_rcparamkeytype_literal(keys: list[str]) -> str:
    """
    Generate the Literal block string for RcParamKeyType with the given keys.

    Parameters
    ----------
    keys : list[str]
        Sorted list of rcParams keys as strings.

    Returns
    -------
    str
        A formatted string defining RcParamKeyType as a typing.Literal of keys,
        wrapped with start and end generation markers.
    """
    keys_str = ",\n    ".join(f'"{key}"' for key in keys)
    return (
        "# --- START GENERATED RcParamKeyType ---\n"
        "RcParamKeyType: TypeAlias = Literal[\n"
        f"    {keys_str},\n"
        "]\n"
        "# --- END GENERATED RcParamKeyType ---"
    )

def update_typing_file(path: str, new_block: str) -> None:
    """
    Replace the existing RcParamKeyType block in the typing file with new_block.

    Parameters
    ----------
    path : str
        File path of the typing file to update.

    new_block : str
        The new Literal block string to insert.

    Raises
    ------
    RuntimeError
        If the start and end markers are not found in the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"# --- START GENERATED RcParamKeyType ---\n.*?# --- END GENERATED RcParamKeyType ---",
        re.DOTALL,
    )

    new_content, count = pattern.subn(new_block, content)
    if count == 0:
        raise RuntimeError(f"Markers not found in {path}. Please ensure the markers exist.")

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

if __name__ == "__main__":
    # Retrieve and sort rcParams keys from Matplotlib runtime
    rc_keys = sorted(matplotlib.rcParams.keys())

    # Generate new Literal block
    new_block = generate_rcparamkeytype_literal(rc_keys)

    # Update typing.py file in place
    update_typing_file(typing_file, new_block)

    print(f"Updated {typing_file} with {len(rc_keys)} RcParamKeyType keys.")
