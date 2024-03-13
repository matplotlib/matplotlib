from enum import Enum


class BackendFilter(Enum):
    INTERACTIVE: int
    NON_INTERACTIVE: int


class BackendRegistry:
    def backend_for_gui_framework(self, interactive_framework: str) -> str | None: ...
    def list_builtin(self, filter_: BackendFilter | None) -> list[str]: ...


backend_registry: BackendRegistry
