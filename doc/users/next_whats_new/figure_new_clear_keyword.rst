New parameter `clear` for :func:`~matplotlib.pyplot.figure`
-----------------------------------------------------------

When the pyplot's function :func:`~matplotlib.pyplot.figure` is called
with a ``num`` parameter, a new window is only created if no existing
window with the same value exists. A new bool parameter `clear` was
added for explicitly clearing its existing contents. This is particularly
useful when utilized in interactive sessions. Since
:func:`~matplotlib.pyplot.subplots` also accepts keyword arguments
from :func:`~matplotlib.pyplot.figure`, it can also be used there::

   import matplotlib.pyplot as plt
   
   fig0 = plt.figure(num=1)
   fig0.suptitle("A fancy plot")
   print("fig0.texts: ", [t.get_text() for t in fig0.texts])
   
   fig1 = plt.figure(num=1, clear=False)  # do not clear contents of window
   fig1.text(0.5, 0.5, "Really fancy!")
   print("fig0 is fig1: ",  fig0 is fig1)
   print("fig1.texts: ", [t.get_text() for t in fig1.texts]) 
  
   fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)  # clear contents
   print("fig0 is fig2: ",  fig0 is fig2)  
   print("fig2.texts: ", [t.get_text() for t in fig2.texts])

   # The output:
   # fig0.texts:  ['A fancy plot']
   # fig0 is fig1:  True
   # fig1.texts:  ['A fancy plot', 'Really fancy!']
   # fig0 is fig2:  True
   # fig2.texts:  []