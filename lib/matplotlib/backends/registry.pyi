from enum import Enum


class BackendFilter(Enum):
    INTERACTIVE: int
    INTERACTIVE_NON_WEB: int
    NON_INTERACTIVE: int


class BackendRegistry:
    def __init__(self) -> None: ...
    def framework_to_backend(self, interactive_framework: str) -> str | None: ...
    def list_builtin(self, filter_: BackendFilter | None) -> list[str]: ...


backendRegistry: BackendRegistry
