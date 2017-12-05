Added support for QT in new ToolManager
=======================================

Now it is possible to use the ToolManager with Qt5
For example

  import matplotlib

  matplotlib.use('QT5AGG')
  matplotlib.rcParams['toolbar'] = 'toolmanager'
  import matplotlib.pyplot as plt

  plt.plot([1,2,3])
  plt.show()

The main example `examples/user_interfaces/toolmanager_sgskip.py` shows more
details, just adjust the header to use QT instead of GTK3
