# Some functions to load a return data for the plot demos

from Numeric import fromstring, argsort, take
import Numeric as numpy
def get_two_stock_data():
    """
    load stock time and price data for two stocks The return values
    (d1,p1,d2,p2) are the trade time (in days) and prices for stocks 1
    and 2 (intc and aapl)
    """
    ticker1, ticker2 = 'INTC', 'AAPL'
    M1 = fromstring( file('data/%s.dat' % ticker1, 'rb').read(), 'd')

    M1 = M1.resize( (M1.shape[0]/2,2) )

    M2 = fromstring( file('data/%s.dat' % ticker2, 'rb').read(), 'd')
    M2 = M2.resize( (M2.shape[0]/2,2) )

    d1, p1 = M1[:,0], M1[:,1]
    d2, p2 = M2[:,0], M2[:,1]
    return (d1,p1,d2,p2)

