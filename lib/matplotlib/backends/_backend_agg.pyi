# Stubs for the _backend_agg C-extension.

class BufferRegion: ...

class RendererAgg:
    def __init__(
        self,
        width: int,
        height: int,
        dpi: float,
    ) -> None: ...
