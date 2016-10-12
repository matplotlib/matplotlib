try:
    import cairocffi as cairo
except ImportError:
    try:
        import cairo
    except ImportError:
        raise ImportError(
            "Cairo backend requires that cairocffi or pycairo is installed.")
    else:
        HAS_CAIRO_CFFI = False
else:
    HAS_CAIRO_CFFI = True

_version_required = (1, 2, 0)
if cairo.version_info < _version_required:
    raise ImportError("Pycairo %d.%d.%d is installed\n"
                      "Pycairo %d.%d.%d or later is required"
                        % (cairo.version_info + _version_required))
backend_version = cairo.version
del _version_required
