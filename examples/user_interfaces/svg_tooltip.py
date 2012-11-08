"""
SVG tooltip example
===================

This example shows how to create a tooltip that will show up when 
hovering over a matplotlib patch. 

Although it is possible to create the tooltip from CSS or javascript, 
here we create it in matplotlib and simply toggle its visibility on 
when hovering over the patch. This approach provides total control over 
the tooltip placement and appearance, at the expense of more code up
front. 

The alternative approach would be to put the tooltip content in `title` 
atttributes of SVG objects. Then, using an existing js/CSS library, it 
would be relatively straightforward to create the tooltip in the 
browser. The content would be dictated by the `title` attribute, and 
the appearance by the CSS. 


:author: David Huard
"""


import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from StringIO import StringIO

ET.register_namespace("","http://www.w3.org/2000/svg")

fig, ax = plt.subplots()

# Create patches to which tooltips will be assigned.
circle = plt.Circle((0,0), 5, fc='blue')
rect = plt.Rectangle((-5, 10), 10, 5, fc='green')

ax.add_patch(circle)
ax.add_patch(rect)

# Create the tooltips
circle_tip = ax.annotate('This is a blue circle.', 
            xy=(0,0), 
            xytext=(30,-30), 
            textcoords='offset points', 
            color='w', 
            ha='left', 
            bbox=dict(boxstyle='round,pad=.5', fc=(.1,.1,.1,.92), ec=(1.,1.,1.), lw=1, zorder=1),
            )
            
rect_tip = ax.annotate('This is a green rectangle.', 
            xy=(-5,10), 
            xytext=(30,40), 
            textcoords='offset points', 
            color='w', 
            ha='left', 
            bbox=dict(boxstyle='round,pad=.5', fc=(.1,.1,.1,.92), ec=(1.,1.,1.), lw=1, zorder=1),
            )
            

# Set id for the patches    
for i, t in enumerate(ax.patches):
    t.set_gid('patch_%d'%i)

# Set id for the annotations
for i, t in enumerate(ax.texts):
    t.set_gid('tooltip_%d'%i)


# Save the figure in a fake file object
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')

f = StringIO()
plt.savefig(f, format="svg")

# --- Add interactivity ---

# Create XML tree from the SVG file.
tree, xmlid = ET.XMLID(f.getvalue())
tree.set('onload', 'init(evt)')

# Hide the tooltips
for i, t in enumerate(ax.texts):
    el = xmlid['tooltip_%d'%i]
    el.set('visibility', 'hidden')

# Assign onmouseover and onmouseout callbacks to patches.        
for i, t in enumerate(ax.patches):
    el = xmlid['patch_%d'%i]
    el.set('onmouseover', "ShowTooltip(this)")
    el.set('onmouseout', "HideTooltip(this)")

# This is the script defining the ShowTooltip and HideTooltip functions.
script = """
    <script type="text/ecmascript">
    <![CDATA[
    
    function init(evt) {
        if ( window.svgDocument == null ) {
            svgDocument = evt.target.ownerDocument;
            }
        }
        
    function ShowTooltip(obj) {
        var cur = obj.id.slice(-1);
        
        var tip = svgDocument.getElementById('tooltip_' + cur);
        tip.setAttribute('visibility',"visible")
        }
        
    function HideTooltip(obj) {
        var cur = obj.id.slice(-1);
        var tip = svgDocument.getElementById('tooltip_' + cur);
        tip.setAttribute('visibility',"hidden")
        }
        
    ]]>
    </script>
    """

# Insert the script at the top of the file and save it.
tree.insert(0, ET.XML(script))
ET.ElementTree(tree).write('svg_tooltip.svg')
