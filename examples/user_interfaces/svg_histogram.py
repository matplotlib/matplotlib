#!/usr/bin/env python
#-*- encoding:utf-8 -*-

"""
Demonstrate how to create an interactive histogram, in which bars
are hidden or shown by cliking on legend markers. 

The interactivity is encoded in ecmascript and inserted in the SVG code
in a post-processing step. To render the image, open it in a web
browser. SVG is supported in most web browsers used by Linux and OSX 
users. Windows IE9 supports SVG, but earlier versions do not. 

__author__="david.huard@gmail.com"

"""

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from StringIO import StringIO

plt.rcParams['svg.embed_char_paths'] = 'none'

# Apparently, this `register_namespace` method works only with 
# python 2.7 and up and is necessary to avoid garbling the XML name
# space with ns0.
ET.register_namespace("","http://www.w3.org/2000/svg")

    
def python2js(d):
    """Return a string representation of a python dictionary in 
    ecmascript object syntax."""
    
    objs = []
    for key, value in d.items():
        objs.append( key + ':' + str(value) )
        
    return '{' + ', '.join(objs) + '}'


# --- Create histogram, legend and title ---
plt.figure()
r = np.random.randn(100)
r1 = r + 1
labels = ['Rabbits', 'Frogs']
H = plt.hist([r,r1], label=labels)
containers = H[-1]
leg = plt.legend(frameon=False)
plt.title("""From a web browser, click on the legend 
marker to toggle the corresponding histogram.""")


# --- Add ids to the svg objects we'll modify 

hist_patches = {}
for ic, c in enumerate(containers):
    hist_patches['hist_%d'%ic] = []
    for il, element in enumerate(c):
        element.set_gid('hist_%d_patch_%d'%(ic, il))
        hist_patches['hist_%d'%ic].append('hist_%d_patch_%d'%(ic,il))    

# Set ids for the legend patches    
for i, t in enumerate(leg.get_patches()):
    t.set_gid('leg_patch_%d'%i)
    
# Save SVG in a fake file object.
f = StringIO()
plt.savefig(f, format="svg")

# Create XML tree from the SVG file.
tree, xmlid = ET.XMLID(f.getvalue())


# --- Add interactivity ---

# Add attributes to the patch objects.    
for i, t in enumerate(leg.get_patches()):
    el = xmlid['leg_patch_%d'%i]
    el.set('cursor', 'pointer')
    el.set('opacity', '1.0')
    el.set('onclick', "toggle_element(evt, 'hist_%d')"%i)
    
# Create script defining the function `toggle_element`. 
# We create a global variable `container` that stores the patches id 
# belonging to each histogram. Then a function "toggle_element" sets the 
# visibility attribute of all patches of each histogram and the opacity
# of the marker itself. 

script = """
<script type="text/ecmascript">
<![CDATA[
var container = %s 

function toggle_element(evt, element) {

    var names = container[element]
    var el, state;
    
    state = evt.target.getAttribute("opacity") == 1.0 || 
                evt.target.getAttribute("opacity") == null;
                
    if (state) {
        evt.target.setAttribute("opacity", 0.5);
        
        for (var i=0; i < names.length; i++) {
            el = document.getElementById(names[i]);    
            el.setAttribute("visibility","hidden");
            }
        }
        
    else {
        evt.target.setAttribute("opacity", 1);
        
        for (var i=0; i < names.length; i++) {
            el = document.getElementById(names[i]);    
            el.setAttribute("visibility","visible");
            }
        
        };
    };
]]>
</script>
"""%python2js(hist_patches)

# Insert the script and save to file.
tree.insert(0, ET.XML(script))

ET.ElementTree(tree).write("svg_histogram.svg")

    

