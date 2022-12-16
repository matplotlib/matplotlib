from ._contourpy import FillType, LineType, ZInterp


def as_fill_type(fill_type):
    """Coerce a FillType or string value to a FillType.

    Args:
        fill_type (FillType or str): Value to convert.

    Return:
        FillType: Converted value.
    """
    if isinstance(fill_type, str):
        fill_type = FillType.__members__[fill_type]
    return fill_type


def as_line_type(line_type):
    """Coerce a LineType or string value to a LineType.

    Args:
        line_type (LineType or str): Value to convert.

    Return:
        LineType: Converted value.
    """
    if isinstance(line_type, str):
        line_type = LineType.__members__[line_type]
    return line_type


def as_z_interp(z_interp):
    """Coerce a ZInterp or string value to a ZInterp.

    Args:
        z_interp (ZInterp or str): Value to convert.

    Return:
        ZInterp: Converted value.
    """
    if isinstance(z_interp, str):
        z_interp = ZInterp.__members__[z_interp]
    return z_interp
