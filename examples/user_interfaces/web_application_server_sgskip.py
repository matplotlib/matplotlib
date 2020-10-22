"""
=============================================
Embedding in a web application server (Flask)
=============================================

When using Matplotlib in a web server it is strongly recommended to not use
pyplot (pyplot maintains references to the opened figures to make
`~.matplotlib.pyplot.show` work, but this will cause memory leaks unless the
figures are properly closed).

Since Matplotlib 3.1, one can directly create figures using the `.Figure`
constructor and save them to in-memory buffers.  In older versions, it was
necessary to explicitly instantiate an Agg canvas (see e.g.
:doc:`/gallery/user_interfaces/canvasagg`).

The following example uses Flask_, but other frameworks work similarly:

.. _Flask: https://flask.palletsprojects.com

"""

import base64
from io import BytesIO

from flask import Flask
from matplotlib.figure import Figure

app = Flask(__name__)


@app.route("/")
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

#############################################################################
#
# Since the above code is a Flask application, it should be run using the
# `flask command-line tool <https://flask.palletsprojects.com/en/master/cli/>`_
# Assuming that the working directory contains this script:
#
# Unix-like systems
#
# .. code-block:: console
#
#  FLASK_APP=web_application_server_sgskip flask run
#
# Windows
#
# .. code-block:: console
#
#  set FLASK_APP=web_application_server_sgskip
#  flask run
#
#
# Clickable images for HTML
# -------------------------
#
# Andrew Dalke of `Dalke Scientific <http://www.dalkescientific.com>`_
# has written a nice `article
# <http://www.dalkescientific.com/writings/diary/archive/2005/04/24/interactive_html.html>`_
# on how to make html click maps with Matplotlib agg PNGs.  We would
# also like to add this functionality to SVG.  If you are interested in
# contributing to these efforts that would be great.
