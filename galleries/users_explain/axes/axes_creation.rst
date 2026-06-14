#################################
Overview of Axes creation methods
#################################

Depending on the use case, different methods for creating an Axes are most useful.
The following diagram gives an overview by characterizing the desired Axes.

Additionally, most of the methods come in two flavors: A `.pyplot` function and a
method on a `.Figure` or `~.axes.Axes`.

``plt.subplots()`` and ``plt.subplot_mosaic()`` return a Figure and one or more Axes.
They are the most common stating points for a Matplotlib plot. All other methods
only return an Axes.

.. graphviz::

    digraph {
        node[shape=diamond, width=2.7, height=0.8]
            q_fullsize; q_regular_grid; q_general_grid; q_inset; q_twin;
        node[shape=none, margin=0]
            full_size;

        legend [label=<
    <FONT FACE="helvetica-bold"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200">pyplot functions</TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200">Figure/Axes methods</TD></TR>
    <TR><TD BGCOLOR="white" ALIGN="left" WIDTH="200"><FONT FACE="helvetica" COLOR="gray50">less used variants</FONT></TD></TR>
    </TABLE></FONT>>]


        q_fullsize [label="full size?"]
        full_size [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200">plt.subplots()</TD><TD BGCOLOR="gray90" ALIGN="left">Figure.subplots()</TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200"><FONT COLOR="gray50">plt.subplot()</FONT></TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200"><FONT COLOR="gray50">Figure.add_subplot()</FONT></TD></TR>
    </TABLE></FONT>>]
        q_regular_grid [label="regular grid?"]
        regular_grid [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left">plt.subplots(n, m)</TD><TD BGCOLOR="gray90" ALIGN="left">Figure.subplots(n, m)</TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200"><FONT COLOR="gray50">plt.subplot(n, m, k)</FONT></TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200"><FONT COLOR="gray50">Figure.add_subplot(n, m, k)</FONT></TD></TR>
    </TABLE></FONT>>]
        q_general_grid [label="general grid?"]
        general_grid [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200">plt.subplot_mosaic()</TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200">Figure.subplot_mosaic()</TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left"><FONT COLOR="gray50">plt.subplot2grid()</FONT></TD><TD BGCOLOR="gray90" ALIGN="left"><FONT COLOR="gray50">GridSpec.subplots()</FONT></TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left"><FONT COLOR="gray50">plt.subplots(gridspec_kw)</FONT></TD><TD BGCOLOR="gray90" ALIGN="left"><FONT COLOR="gray50">Figure.subplots(gridspec_kw)</FONT></TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left"><FONT COLOR="gray50">plt.subplot(subplotspec)</FONT></TD><TD BGCOLOR="gray90" ALIGN="left"><FONT COLOR="gray50">Figure.subplot(subplotspec)</FONT></TD></TR>
    </TABLE></FONT>>]
        q_inset [label="inset?"]
        inset [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5" ALIGN="left">
    <TR><TD BGCOLOR="white" ALIGN="left" WIDTH="200"></TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200">Axes.inset_axes()</TD></TR>
    </TABLE></FONT>>]
        q_twin [label="twin?"]
        twin [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="200">plt.twinx()</TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="200">Axes.twinx()</TD></TR>
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left">plt.twiny()</TD><TD BGCOLOR="gray90" ALIGN="left">Axes.twiny()</TD></TR>
    </TABLE></FONT>>]
        axes [label=<
    <FONT FACE="helvetica"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" CELLPADDING="5">
    <TR><TD BGCOLOR="#C5E5F7" ALIGN="left" WIDTH="150">plt.axes()</TD><TD BGCOLOR="gray90" ALIGN="left" WIDTH="150">Figure.add_axes()</TD></TR>
    </TABLE></FONT>>]

        legend -> full_size [ style = invis ];

        q_fullsize -> q_regular_grid [ label = "No" ];
        q_regular_grid -> q_general_grid [label = "No" ];
        q_fullsize -> full_size [ label = "Yes" ];
        q_regular_grid -> regular_grid[ label = "Yes" ];
        q_general_grid -> general_grid [label = "Yes"];
        q_general_grid -> q_inset [label = "No"];
        q_inset -> inset[label="Yes"];
        q_inset -> q_twin [label="No"]
        q_twin -> twin[label="Yes"]
        q_twin -> axes[label="No"]

        {
            rank=same;
            q_fullsize full_size
        }
        {
            rank=same;
            q_regular_grid regular_grid
        }
        {
            rank=same;
            q_general_grid general_grid
        }
        {
            rank=same;
            q_inset inset
        }
        {
            rank=same;
            q_twin twin
        }
    }
