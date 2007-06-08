from pylab import plotfile, show

fname = 'data/msft.csv'    

# test 1; use ints
plotfile(fname, (0,5,6))

# test 2; use names
plotfile(fname, ('date', 'volume', 'adj_close'))

# test 3; use semilogy for volume
plotfile(fname, ('date', 'volume', 'adj_close'), plotfuncs={'volume': 'semilogy'})

# test 4; use semilogy for volume
plotfile(fname, (0,5,6), plotfuncs={5:'semilogy'})

# test 5; use bar for volume
plotfile(fname, (0,5,6), plotfuncs={5:'bar'})

show()


