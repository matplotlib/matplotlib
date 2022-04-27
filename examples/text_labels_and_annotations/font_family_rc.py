"""
===========================
Configuring the font family
===========================

You can explicitly set which font family is picked up, either by specifying
family names of fonts installed on user's system, or generic-families
(e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
or a combination of both.
(see :doc:`font tutorial </tutorials/text/text_props>`)

In the example below, we are overriding the default sans-serif generic family
to include a specific (Tahoma) font. (Note that the best way to achieve this
would simply be to prepend 'Tahoma' in 'font.family')

The default family is set with the font.family rcparam,
e.g. ::

  rcParams['font.family'] = 'sans-serif'

and for the font.family you set a list of font styles to try to find
in order::

  rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                 'Lucida Grande', 'Verdana']

.. redirect-from:: /examples/font_family_rc_sgskip

"""

# Chooses default sans-serif font:
# -----------------------------------------------------

import matplotlib.pyplot as plt


def print_text(text):
    fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
    ax.text(0.01, 0.2, text, size=40)
    ax.axis("off")
    plt.show()


plt.rcParams["font.family"] = "sans-serif"
print_text("Hello World! 01")


#################################################################
#
# Chose sans-serif font and specify to it to "Nimbus Sans":
# -----------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
print_text("Hello World! 02")


#################################################################
#
# Chose sans-serif font and specify to it to "Humor Sans":
# -----------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Humor Sans"]
print_text("Hello World! 03")


##################################################################
#
# Chooses default monospace font:
# -----------------------------------------------------

plt.rcParams["font.family"] = "monospace"
print_text("Hello World! 04")


##################################################################
#
# Chose monospace font and specify to it to "FreeMono":
# -----------------------------------------------------
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["FreeMono"]
print_text("Hello World! 05")


##################################################################
#
# Print all available font paths (this is OS specific):
# -----------------------------------------------------

import matplotlib.font_manager
fpaths = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
for i in fpaths:
    print(i)
