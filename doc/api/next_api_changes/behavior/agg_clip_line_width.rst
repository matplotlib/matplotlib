Agg no longer cuts off thick lines just inside the canvas edge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Agg backend now grows the canvas clipping rectangle by half the line width
before clipping paths and collections. Thick lines whose centre lies just outside
the canvas, or that end close to its edge, are now drawn up to the edge instead
of being cut off early. Images saved with Agg can therefore differ slightly near
the figure border.
