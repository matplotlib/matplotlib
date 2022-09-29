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

.. redirect-from:: /gallery/font_family_rc_sgskip



The font font.family defaults are OS dependent and can be viewed with
"""
import matplotlib.pyplot as plt

print(plt.rcParams["font.sans-serif"][0])
print(plt.rcParams["font.monospace"][0])


#################################################################
# Choose default sans-serif font

def print_text(text):
    fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
    ax.text(0.5, 0.5, text, ha='center', va='center', size=40)
    ax.axis("off")
    plt.show()


plt.rcParams["font.family"] = "sans-serif"
print_text("Hello World! 01")


#################################################################
# Choose sans-serif font and specify to it to "Nimbus Sans"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
print_text("Hello World! 02")


##################################################################
# Choose default monospace font

plt.rcParams["font.family"] = "monospace"
print_text("Hello World! 03")


###################################################################
# Choose monospace font and specify to it to "FreeMono"

plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["FreeMono"]
print_text("Hello World! 04")
