.. _users-guide-index:

.. redirect-from:: /contents

##########
User guide
##########

.. grid:: 1 1 2 2

   .. grid-item-card:: Starting information
      :padding: 2

      .. plot::

         x = np.linspace(0, 2, 100)  # Sample data.

         fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
         ax.scatter(x, np.random.randn(len(x)) + x**2,
                    s=13 * np.random.rand(len(x)), c=np.random.randn(len(x)),
                    label='noisy data')
         ax.plot(x, x, label='linear')  # Plot some data on the axes.
         ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
         ax.plot(x, x**3, label='cubic')  # ... and some more.
         ax.set_xlabel('x label')  # Add an x-label to the axes.
         ax.set_ylabel('y label')  # Add a y-label to the axes.
         ax.set_title("Simple plot")  # Add a title to the axes.
         ax.legend()  # Add a legend.

      .. toctree::
         :maxdepth: 1

         getting_started/index.rst
         installing/index.rst
         faq/index.rst

   .. grid-item-card:: Users guide
      :padding: 2

      .. toctree::
         :maxdepth: 2

         explain/index.rst

   .. grid-item-card:: Tutorials and examples
      :padding: 2

      .. toctree::
         :maxdepth: 1

         ../plot_types/index.rst
         ../gallery/index.rst
         ../tutorials/index.rst
         resources/index.rst


      .. raw:: html

         <div class="grid__intro" id="image_rotator"></div>


   .. grid-item-card:: More information
      :padding: 2

      .. toctree::
         :maxdepth: 1

         Reference <../api/index.rst>
         Contribute <../devel/index.rst>
         Releases <release_notes.rst>

      .. toctree::
         :maxdepth: 2

         project/index.rst
