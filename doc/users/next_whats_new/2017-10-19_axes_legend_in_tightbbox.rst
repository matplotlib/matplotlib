Axes legends now included in tight_bbox
---------------------------------------

Legends created via ``ax.legend()`` can sometimes overspill the limits of
the axis.  Tools like ``fig.tight_layout()`` and
``fig.savefig(bbox_inches='tight')`` would clip these legends.  A  change
was made to include them in the ``tight`` calculations.
