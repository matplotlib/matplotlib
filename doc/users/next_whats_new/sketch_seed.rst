``sketch_seed`` parameter for rcParams
--------------------------------------

`~matplotlib.rcParams` now has a new parameter ``path.sketch_seed``.
Its default value is 0 and accepted values are any non negative integer.
This allows the user to set the seed for the internal pseudo random number generator in one of three ways.

1) Directly changing the rcParam:

    rcParams['path.sketch_seed'] = 20

2) Passing a value to the new *seed* parameter of `~matplotlib.pyplot.xkcd` function:

    plt.xkcd(seed=20)

3) Passing a value to the new *seed* parameter of matplotlib.artist.set_sketch_params function:

    ln = plt.plot(x, y)
    ln[0].set_sketch_params(seed = 20)

The seed will also have a changing characteristic for every artist which will be done in a deterministic manner.


.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    with plt.xkcd():
        rcParams['path.sketch_seed']=0
        rcParams['path.sketch']=(2,120,40)
        pat,txt=plt.pie([10,20,30,40],wedgeprops={'edgecolor':'black'})
        plt.legend(pat,['first','second','third','fourth'],loc='best')
        plt.title("seed = 0")
    plt.show()

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    fig, ax = plt.subplots()
    x = np.linspace(0.7, 1.42, 100)
    y = x ** 2
    ln = ax.plot(x, y, color='black')
    ln[0].set_sketch_params(100, 100, 20, 40)
    plt.title("seed = 40")
    plt.show()

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    with plt.xkcd(seed=19680801):
        import matplotlib
        from matplotlib import gridspec

        rcParams['path.sketch']=(2,120,40)

        pat,txt=plt.pie([10,20,30,40],wedgeprops={'edgecolor':'black'})
        plt.legend(pat,['first','second','third','fourth'],loc='best')
        plt.title("seed = 19680801")
    plt.show()
