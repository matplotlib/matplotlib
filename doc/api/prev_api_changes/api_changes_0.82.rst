Changes for 0.82
================

.. code-block:: text

  - toolbar import change in GTKAgg, GTKCairo and WXAgg

  - Added subplot config tool to GTK* backends -- note you must now
    import the NavigationToolbar2 from your backend of choice rather
    than from backend_gtk because it needs to know about the backend
    specific canvas -- see examples/embedding_in_gtk2.py.  Ditto for
    wx backend -- see examples/embedding_in_wxagg.py


  - hist bin change

      Sean Richards notes there was a problem in the way we created
      the binning for histogram, which made the last bin
      underrepresented.  From his post:

        I see that hist uses the linspace function to create the bins
        and then uses searchsorted to put the values in their correct
        bin. That's all good but I am confused over the use of linspace
        for the bin creation. I wouldn't have thought that it does
        what is needed, to quote the docstring it creates a "Linear
        spaced array from min to max". For it to work correctly
        shouldn't the values in the bins array be the same bound for
        each bin? (i.e. each value should be the lower bound of a
        bin). To provide the correct bins for hist would it not be
        something like

        def bins(xmin, xmax, N):
          if N==1: return xmax
          dx = (xmax-xmin)/N # instead of N-1
          return xmin + dx*arange(N)


       This suggestion is implemented in 0.81.  My test script with these
       changes does not reveal any bias in the binning

        from matplotlib.numerix.mlab import randn, rand, zeros, Float
        from matplotlib.mlab import hist, mean

        Nbins = 50
        Ntests = 200
        results = zeros((Ntests,Nbins), typecode=Float)
        for i in range(Ntests):
            print 'computing', i
            x = rand(10000)
            n, bins = hist(x, Nbins)
            results[i] = n
        print mean(results)
