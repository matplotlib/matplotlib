Additional format string options in `~matplotlib.axes.Axes.bar_label`
---------------------------------------------------------------------

The ``fmt`` argument of `~matplotlib.axes.Axes.bar_label` now accepts
{}-style format strings:

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt

    fruit_names = ['Coffee', 'Salted Caramel', 'Pistachio']
    fruit_counts = [4000, 2000, 7000]

    fig, ax = plt.subplots()
    bar_container = ax.bar(fruit_names, fruit_counts)
    ax.set(ylabel='pints sold', title='Gelato sales by flavor', ylim=(0, 8000))
    ax.bar_label(bar_container, fmt='{:,.0f}')

It also accepts callables:

.. plot::
    :include-source: true

    animal_names = ['Lion', 'Gazelle', 'Cheetah']
    mph_speed = [50, 60, 75]

    fig, ax = plt.subplots()
    bar_container = ax.bar(animal_names, mph_speed)
    ax.set(ylabel='speed in MPH', title='Running speeds', ylim=(0, 80))
    ax.bar_label(
        bar_container, fmt=lambda x: '{:.1f} km/h'.format(x * 1.61)
    )
