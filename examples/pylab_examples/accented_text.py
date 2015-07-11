#!/usr/bin/env python
"""
matplotlib supports accented characters via TeX mathtext

The following accents are provided: \hat, \breve, \grave, \bar,
\acute, \tilde, \vec, \dot, \ddot.  All of them have the same syntax,
e.g., to make an overbar you do \bar{o} or to make an o umlaut you do
\ddot{o}.  The shortcuts are also provided, e.g.,: \"o \'e \`e \~n \.x
\^y

"""
import matplotlib.pyplot as plt

plt.axes([0.1, 0.15, 0.8, 0.75])
plt.plot(range(10))

plt.title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{i}\bar{A}\tilde{n}\vec{q}$', fontsize=20)
# shorthand is also supported and curly's are optional
plt.xlabel(r"""$\"o\ddot o \'e\`e\~n\.x\^y$""", fontsize=20)


plt.show()
