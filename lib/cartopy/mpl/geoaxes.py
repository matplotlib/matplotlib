from matplotlib.axes import Axes

class GeoAxes(Axes):
    def get_title_top(self) -> float:
        """
        Calculate the top position of the title for geographic projections.

        Returns
        -------
        float
            The top edge position of the title in axis coordinates,
            adjusted for geographic projection.
        """
        base_top = super().get_title_top()
        if self.projection.is_geodetic():
            return base_top + 0.02
        return base_top 