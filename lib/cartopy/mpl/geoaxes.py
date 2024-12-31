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
        # Get base value from parent class
        base_top = super().get_title_top()
        
        # Adjustment based on projection type
        if self.projection.is_geodetic():
            # Extra padding for geodetic projections
            return base_top + 0.02
        
        return base_top 