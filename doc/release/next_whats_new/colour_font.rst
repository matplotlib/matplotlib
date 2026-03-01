Support for (some) colour fonts
-------------------------------

Various colour fonts (e.g., emoji fonts) are now supported. Note, that certain
newer, more complex fonts are `not yet supported
<https://github.com/matplotlib/matplotlib/issues/31206>`__.


.. plot::

    from pathlib import Path
    from matplotlib.font_manager import FontProperties

    zwj = '\U0000200D'
    adult = '\U0001F9D1'
    man = '\U0001F468'
    woman = '\U0001F469'
    science = '\U0001F52C'
    technology = '\U0001F4BB'
    skin_tones = ['', *(chr(0x1F3FB + i) for i in range(5))]

    text = '\n'.join([
        ''.join(person + tone + zwj + occupation for tone in skin_tones)
        for person in [adult, man, woman]
        for occupation in [science, technology]
    ])

    path = Path(plt.__file__).parent / 'tests/data/OpenMoji-color-glyf_colr_0-subset.ttf'

    fig = plt.figure(figsize=(6.4, 4.8))
    fig.text(0.5, 0.5, text,
             font=FontProperties(fname=path), fontsize=40,
             horizontalalignment='center', verticalalignment='center')
