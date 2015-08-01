import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import OldScalarFormatter, ScalarFormatter

## Example 1 ## 
x = np.arange(0, 1, .01)
f = plt.figure(figsize=(6, 6))
f.text(0.5, 0.975, 'The old formatter',
       horizontalalignment='center', verticalalignment='top')

plt.subplot(221)
plt.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(OldScalarFormatter())
ax1.yaxis.set_major_formatter(OldScalarFormatter())

plt.subplot(222)
plt.plot(x * 1e5, x * 1e-4)
ax2 = plt.gca()
ax2.xaxis.set_major_formatter(OldScalarFormatter())
ax2.yaxis.set_major_formatter(OldScalarFormatter())

plt.subplot(223)
plt.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3 = plt.gca()
ax3.xaxis.set_major_formatter(OldScalarFormatter())
ax3.yaxis.set_major_formatter(OldScalarFormatter())

plt.subplot(224)
plt.plot(-x * 1e5, -x * 1e-4)
ax4 = plt.gca()
ax4.xaxis.set_major_formatter(OldScalarFormatter())
ax4.yaxis.set_major_formatter(OldScalarFormatter())

## Example 2 ##
x = np.arange(0, 1, .01)
f = plt.figure(figsize=(6, 6))
f.text(0.5, 0.975, 'The new formatter, default settings',
       horizontalalignment='center',
       verticalalignment='top')

plt.subplot(221)
plt.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.set_major_formatter(ScalarFormatter())

plt.subplot(222)
plt.plot(x * 1e5, x * 1e-4)
ax2 = plt.gca()
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_formatter(ScalarFormatter())

plt.subplot(223)
plt.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3 = plt.gca()
ax3.xaxis.set_major_formatter(ScalarFormatter())
ax3.yaxis.set_major_formatter(ScalarFormatter())

plt.subplot(224)
plt.plot(-x * 1e5, -x * 1e-4)
ax4 = plt.gca()
ax4.xaxis.set_major_formatter(ScalarFormatter())
ax4.yaxis.set_major_formatter(ScalarFormatter())

## Example 3 ##
x = np.arange(0, 1, .01)
f = plt.figure(figsize=(6, 6))
f.text(0.5, 0.975, 'The new formatter, no numerical offset',
       horizontalalignment='center',
       verticalalignment='top')

plt.subplot(221)
plt.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.subplot(222)
plt.plot(x * 1e5, x * 1e-4)
ax2 = plt.gca()
ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.subplot(223)
plt.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3 = plt.gca()
ax3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.subplot(224)
plt.plot(-x * 1e5, -x * 1e-4)
ax4 = plt.gca()
ax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax4.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

## Example 4 ##
x = np.arange(0, 1, .01)
f = plt.figure(figsize=(6, 6))
f.text(0.5, 0.975, 'The new formatter, with mathtext',
       horizontalalignment='center',
       verticalalignment='top')
plt.subplot(221)
plt.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

plt.subplot(222)
plt.plot(x * 1e5, x * 1e-4)
ax2 = plt.gca()
ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

plt.subplot(223)
plt.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3 = plt.gca()
ax3.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

plt.subplot(224)
plt.plot(-x * 1e5, -x * 1e-4)
ax4 = plt.gca()
ax4.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.show()
