.. _content-tutorials:

Tutorials
=========

The goal of the tutorials is to guide the reader through the stages of using Matplotlib
to build specific visualization. The tutorials focus on learning how to combine the
features and components that were explained in the user guide.

Format
------

As with the user guide, the tutorials should aim to unpack information in chunks and
build on the previous section.

Where the user guide explains topics like plotting and animation separately, a tutorial
walks the reader through the steps involved in animating a line plot. Generally the
content is limited to describing what is happening at each stage- for example there's no
explanation of why an update function is written-and instead the reader is linked to an
explanation.

#. First we start by stating the objective:

   .. code-block:: rst

      The goal of this tutorial is to create an animated sin wave.

#. Then we describe what needs to be in place to do the task. Here that is the data and
   object that we will animate:

   .. code-block:: rst

      First lets generate a sin wave::

        x = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(x)

      Then we plot an empty line and capture the returned line object::

        fig, ax = plt.subplot()
        l, _ = ax.plot(0,0)

#. With our pieces in place, we instruct the reader on how to animate the object:

   .. code-block:: rst

      Next we write a function that changes the plot on each frame. Here we grow the sin
      curve on each update by setting the new x and y data::

        def update(frame):

          l.set_data(x[:i], y[:i])

#. Next we show them how to generate the animation and describe the arguments:

   .. code-block:: rst

      Lastly we add an animation writer. Here we specify 30 milliseconds between each
      of the 40 frames::

        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

#. Then we put all the steps together to make the animated sin wave:

   .. code-block:: rst

      Now lets put it all together so we can plot an animated sin curve::

        x = np.linspace(0, 2*np.pi, 1000)
        y = np.sin(x)

        fig, ax = plt.subplot()
        l, _ = ax.plot(0,0)

        def update(frame):
            l.set_data(x[:i], y[:i])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

#. Finally, we close with a follow up call to action to learn about the underlying
   concepts:

   .. code-block:: rst

      For more information on animations and lines, see:
      * :ref:`animations`
      * ``:ref:Line2d``


Please note that while the aim is to show how to animate a sin curve, the focus is
always on making the animation. Generally explanations of domain should be limited to
providing contextually necessary information, and tutorials that are heavily domain
specific may be more appropriate for the Scientific Python
`blog <https://blog.scientific-python.org/>`_.
