``copy_from_bbox`` and blitting now round fractional bboxes outwards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy_from_bbox`` snapped the requested bbox to whole pixels in a way that
shrank the saved region: Agg truncated all four edges, whose upper ones are
exclusive, and Cairo rounded inwards on both sides.  Either way, a bbox with
fractional edges lost the pixels it only partially covered, so those pixels were
restored by neither ``restore_region`` nor blitting.  The Qt, GTK3Agg and WxAgg
canvases rounded the same way when deciding which part of the widget to repaint,
so those pixels were never sent to the screen either.  All of them now round
outwards so that the region covers every pixel the bbox touches.  A bbox with
integer edges is unaffected.
