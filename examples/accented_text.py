"""
matplotlib supports accented characters via TeX mathtext

The following accents are provided: \hat, \breve, \grave, \bar,
\acute, \tilde, \vec, \dot, \ddot.  All of them have the same syntax,
eg to make an overbar you do \bar{o} or to make an o umlaut you do
\ddot{o}.  The shortcuts are also provided, eg: \"o \'e \`e \~n \.x
\^y

"""
from pylab import *

plot(range(10))

title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{i}\bar{A}\tilde{n}\vec{q}$', fontsize=20)
# shorthad is also supported
xlabel(r"""$\"o\'e\`e\~n\.x\^y$""", fontsize=20)


show()
