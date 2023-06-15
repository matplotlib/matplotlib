.. _users-guide-index:

.. redirect-from:: /contents

##########
User guide
##########

.. grid:: 1 1 2 2

   .. grid-item-card:: Starting information
      :padding: 2

      .. plot::

         rng = np.random.default_rng(seed=19680808)
         x = np.linspace(0, 4, 1000)  # Sample data.
         y = rng.normal(size=len(x)) * 1.5 + x**2 + np.cumsum(rng.normal(size=len(x))) / 6
         x = x[::10]
         y = y[::10]
         fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

         ax.plot(x, x**2, label='underlying data', linewidth=4, alpha=0.6, color='k')
         ax.scatter(x, y, s=13 * rng.random(size=len(x)), c=rng.normal(size=len(x)),
               label='noisy data')
         # p = np.polyfit(x, y, deg=1)
         # print(p)
         p = np.array([ 3.81283983, -2.00111268])
         out = np.polyval(p, x)
         ax.plot(x, out, label='linear fit')  # Plot some data on the axes.
         # p = np.polyfit(x, y, deg=2)
         # print(p)
         p = np.array([ 1.18076933, -0.86768725,  1.05989268])
         out = np.polyval(p, x)
         ax.plot(x, out, label='quadratic fit')
         ax.set_xlabel('x label')
         ax.set_ylabel('y label')
         ax.set_title("Simple plot")
         ax.legend()


      .. toctree::
         :maxdepth: 1

         getting_started/index.rst
         installing/index.rst
         FAQ: How-to and troubleshooting <faq/index.rst>

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
