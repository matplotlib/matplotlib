"""
=======================
SVG Tooltip: Line or Area
=======================

This example shows an example of a tooltip added to a SVG plot popular
with timeseries plots showing the values at the x position where the 
mouse is hovering.

The tooltip is made by dynamically inserting a SVG forignObject with a 
table, a dotted line at the current x and coloured dots along this line
highlighting the values in case the lines are hard to distinguish.
This currently works for area, line plots but currently for bar plots
correctly.

:author: Dylan Jay
"""


import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from io import BytesIO

ET.register_namespace("", "http://www.w3.org/2000/svg")

fig, ax = plt.subplots()

# Create patches to which tooltips will be assigned.
rect1 = plt.Rectangle((10, -20), 10, 5, fc='blue')
rect2 = plt.Rectangle((-20, 15), 10, 5, fc='green')

shapes = [rect1, rect2]
labels = ['This is a blue rectangle.', 'This is a green rectangle']

for i, (item, label) in enumerate(zip(shapes, labels)):
    patch = ax.add_patch(item)
    annotate = ax.annotate(labels[i], xy=item.get_xy(), xytext=(0, 0),
                           textcoords='offset points', color='w', ha='center',
                           fontsize=8, bbox=dict(boxstyle='round, pad=.5',
                                                 fc=(.1, .1, .1, .92),
                                                 ec=(1., 1., 1.), lw=1,
                                                 zorder=1))

    ax.add_patch(patch)
    patch.set_gid('mypatch_{:03d}'.format(i))
    annotate.set_gid('mytooltip_{:03d}'.format(i))

# Save the figure in a fake file object
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')

f = BytesIO()
plt.savefig(f, format="svg")

# --- Add interactivity ---


def svg_hover(plt, path, legend, df, *displays, labels=[]):
    f = BytesIO()
    plt.savefig(f, format="svg")

    # Create XML tree from the SVG file.
    tree, xmlid = ET.XMLID(f.getvalue())
    tree.set('onload', 'init(event)')

    colours = []
    legends = []
    circles = []
    for number, patch in enumerate(legend.get_patches() or legend.get_lines()):
        text = legend.get_texts()[number].get_text()
        text = html.escape(text).encode('ascii', 'xmlcharrefreplace').decode("utf8")
        color = list(patch.get_facecolor() if hasattr(patch, "get_facecolor") else patch.get_color())
        legends.append(text)
        colour = matplotlib.colors.to_hex(color, keep_alpha=False)
        colours.append(colour)
        circles.append(f'<circle id="dot_{number}" r="5" fill="{colour}" />')

    # insert svg to for tooltip in - https://codepen.io/billdwhite/pen/rgEbc
    linesvg = f"""
    <g id="date_line" xmlns="http://www.w3.org/2000/svg" pointer-events="none" visibility="hidden">
        <line x1="500" y1="0" x2="500" y2="2000"  style="fill:none;stroke:#808080;stroke-dasharray:3.7,1.6;stroke-dashoffset:0;"/>
        {"".join(circles)}
    </g>
    """
    xmlid["figure_1"].append(ET.XML(linesvg))
    tooltipsvg = f"""
      <g  xmlns="http://www.w3.org/2000/svg" pointer-events="none" class="tooltip mouse" visibility="hidden" style="background:#0000ff50;">
            <foreignObject id="tooltiptext" width="700" height="750" style="overflow:visible">
            <body xmlns="http://www.w3.org/1999/xhtml" >
            <div style="border:1px solid white; padding: 10px; color: white;  display:table; background-color: rgb(0, 0, 0, 0.60); font-family: 'DejaVu Sans', sans-serif;">
                <table id="tooltip_table">
                </table>
            </div>
            </body>
            </foreignObject>
    </g>
    """
    xmlid["figure_1"].append(ET.XML(tooltipsvg))
    xmlid["figure_1"].set("fill", "black")  # some browsers don't seem to respect background

    # This is the script defining the ShowTooltip and HideTooltip functions.
    script = """
        <script type="text/ecmascript" xmlns="http://www.w3.org/2000/svg">
        <![CDATA[

        function init(event) {
            var tooltip = d3.select("g.tooltip.mouse");
            var line = d3.select("g#date_line line");
            var plot = d3.select("#patch_2");
            var offset = plot.node().getBBox().x;
            var date_label = d3.select("#date");
            // var border = d3.select("#tooltiprect");
            var gap = 15;
            let padding = 4;

            d3.select("#figure_1").on("mousemove", function (evt) {
                // from https://codepen.io/billdwhite/pen/rgEbc
                tooltip.attr('visibility', "visible")
                var plotpos = d3.pointer(evt, plot.node())[0] - offset;
                var index = Math.round(plotpos / plot.node().getBBox().width * (data[0].index.length-1));
                var date = data[0].index[index];
                if (date) {
                    date = date.split("T")[0];
                } else {
                    tooltip.attr('visibility', "hidden");
                    d3.select("g#date_line").attr('visibility', "hidden");
                    d3.select("#legend_1").attr('visibility', "visible");
                    return;
                }
                //date_label.node().textContent = date;
                values = [];
                for ( let number = 0; number < legends.length; number++ ) {
                    var row = [data[0].data[index][number], legends[number], colours[number]];
                    for (let d = 1; d < data.length; d++) {
                        row.push(data[d].data[index][number])
                    }
                    values.push(row);
                }
                values.sort(function(a,b) {return a[0] - b[0]});
                values.reverse();

                table = "<html:tr><html:th>"+(new Date(date)).toDateString()+"</html:th>";
                for (let l = 0; l < labels.length; l++) {
                    table += "<html:th>"+labels[l]+"</html:th>";
                }
                table += "</html:tr>";
                for (let col = 0; col < values.length; col++) {
                    var colour = values[col][2];
                    table += "<html:tr><html:td style='color:" + colour + "'>" + values[col][1] + "</html:td>";
                    for ( let number = 3; number < values[col].length; number++ ) {
                        table += "<html:td style='text-align: right'>" + values[col][number] + "</html:td>";
                    }
                    table += "</html:tr>";
                }
                d3.select("#tooltip_table").html(table);

                var mouseCoords = d3.pointer(evt, tooltip.node().parentElement);
                let tooltipbox = d3.select("#tooltiptext div").node();
                let width = tooltipbox.clientWidth;
                var x = mouseCoords[0] - width - gap*2;
                if (x < 0) {
                    x = mouseCoords[0] + gap;
                }
                tooltip
                    .attr("transform", "translate("
                        + (x) + ","
                        + (mouseCoords[1] - tooltipbox.clientHeight/2) + ")");
                line.attr("x1", mouseCoords[0]);
                line.attr("x2", mouseCoords[0]);
                let top = plot.node().getBBox().y;
                let bottom = top + plot.node().getBBox().height;
                line.attr("y1", top);
                line.attr("y2", bottom);
                d3.select("#date_line").attr('visibility', "visible");
                d3.select("#legend_1").attr('visibility', "hidden");

                // Move the dots
                for (let col = 0; col < legends.length; col++) {
                    let dot = d3.select("#dot_"+col);
                    dot.attr('cy', bottom - (data[0].data[index][col] * (bottom - top)) );
                    dot.attr('cx', mouseCoords[0]);
                    if (data[0].data[index][col] == null) {
                        dot.attr("visibility", "hidden");
                    }
                    else {
                        dot.attr("visibility", "visible");
                    }
                }


            })
            .on("mouseout", function () {
                d3.select("#date_line").attr('visibility', "hidden");
                return tooltip.attr('visibility', "hidden");
                d3.select("#legend_1").attr('visibility', "visible");
            });

        }
        """
    # TODO: get rid of redudant index lists
    data = [d.to_json(orient="split", date_format="iso") for d in [df] + list(displays)]
    if not labels:
        labels = [""] * len(displays)
    script += f"""
        var data = [{",".join(data)}];
        var colours = {json.dumps(colours)};
        var legends = {json.dumps(legends)};
        var labels = {json.dumps(labels)};
        ]]>
        </script>
        """

    # Insert the script at the top of the file and save it.
    tree.insert(0, ET.XML(script))
    tree.insert(0, ET.XML('<script href="https://d3js.org/d3.v7.min.js" xmlns="http://www.w3.org/2000/svg"></script>'))

    ET.ElementTree(tree).write(path)



