from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Tuple, Dict, Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

@dataclass
class LayerSpec:
    method: str
    data: Tuple[Any, ...]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    axis: Literal['primary', 'secondary'] = 'primary'

class OverlayCoordinator:
    # Z-order defaults
    _ZORDER_MAP = {
        'bar': 1, 'hist': 1, 'fill_between': 1,
        'plot': 2, 'step': 2, 'errorbar': 2,
        'scatter': 3, 'annotate': 3, 'text': 3
    }

    def __init__(self, ax: Optional[Axes] = None):
        if ax is None:
            import matplotlib.pyplot as plt
            self.ax_primary = plt.gca()
        else:
            self.ax_primary = ax
        self.ax_secondary: Optional[Axes] = None
        
        # Check if the primary axis already has a twin (shared x)
        # to avoid creating a duplicate if possible, though explicit twinx is safer logic for this specific requirement.
        # We will initialize ax_secondary only when requested.

    def add_layer(self, layer: LayerSpec):
        target_ax = self.ax_primary
        
        if layer.axis == 'secondary':
            if self.ax_secondary is None:
                self.ax_secondary = self.ax_primary.twinx()
            target_ax = self.ax_secondary

        # Determine zorder
        method_name = layer.method
        base_zorder = self._ZORDER_MAP.get(method_name, 2.5) # Default to 2.5 if unknown
        
        # Allow user override in kwargs, else use calculated default
        kwargs = layer.kwargs.copy()
        if 'zorder' not in kwargs:
            kwargs['zorder'] = base_zorder

        # Call the plotting method
        if hasattr(target_ax, method_name):
            plot_method = getattr(target_ax, method_name)
            plot_method(*layer.data, **kwargs)
        else:
            raise AttributeError(f"Axes object has no method '{method_name}'")

    def finalize(self):
        # Unify legends
        handles1, labels1 = self.ax_primary.get_legend_handles_labels()
        if self.ax_secondary:
            handles2, labels2 = self.ax_secondary.get_legend_handles_labels()
        else:
            handles2, labels2 = [], []
        
        if handles1 or handles2:
            self.ax_primary.legend(handles1 + handles2, labels1 + labels2)

        # state integrity: ensure current axis is primary
        import matplotlib.pyplot as plt
        plt.sca(self.ax_primary)
