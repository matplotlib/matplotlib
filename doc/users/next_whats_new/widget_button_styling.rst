Custom styling of button widgets
--------------------------------

Additional custom styling of button widgets may be achieved via the
*label_props* and *radio_props* arguments to `.RadioButtons`; and the
*label_props*, *frame_props*, and *check_props* arguments to `.CheckButtons`.

.. plot::

   from matplotlib.widgets import CheckButtons, RadioButtons

   fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 2), width_ratios=[1, 2])
   default_rb = RadioButtons(ax[0, 0], ['Apples', 'Oranges'])
   styled_rb = RadioButtons(ax[0, 1], ['Apples', 'Oranges'],
                            label_props={'color': ['red', 'orange'],
                                         'fontsize': [16, 20]},
                            radio_props={'edgecolor': ['red', 'orange'],
                                         'facecolor': ['mistyrose', 'peachpuff']})

   default_cb = CheckButtons(ax[1, 0], ['Apples', 'Oranges'],
                             actives=[True, True])
   styled_cb = CheckButtons(ax[1, 1], ['Apples', 'Oranges'],
                            actives=[True, True],
                            label_props={'color': ['red', 'orange'],
                                         'fontsize': [16, 20]},
                            frame_props={'edgecolor': ['red', 'orange'],
                                         'facecolor': ['mistyrose', 'peachpuff']},
                            check_props={'color': ['darkred', 'darkorange']})

   ax[0, 0].set_title('Default')
   ax[0, 1].set_title('Stylized')
