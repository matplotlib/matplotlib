New parameter `clear` for :func:`~matplotlib.pyplot.figure`
-----------------------------------------------------------

When the pyplot's function :func:`~matplotlib.pyplot.figure` is called
with a ``num`` parameter, a new window is only created if no existing
window with the same number exists. A new bool parameter `clear` was
added for explicitly clearing its existing contents. This is particularly
useful when utilized in interactive sessions. Since
:func:`~matplotlib.pyplot.subplots` also accepts keyword arguments
from :func:`~matplotlib.pyplot.figure`, it can also be used there::

   import matplotlib.pyplot as plt
   
   fg0 = plt.figure(num=1);
   fg0.suptitle("A fancy plot");
   print("fg0.texts: ", [t.get_text() for t in fg0.texts])
   
   fg1 = plt.figure(num=1, clear=False);  # do not clear contents of window
   fg1.text(0.5, 0.5, "Really fancy!")
   print("fg0 is fg1: ",  fg0 is fg1)
   print("fg1.texts: ", [t.get_text() for t in fg1.texts]) 
  
   fg2, axx = plt.subplots(2, 1, num=1, clear=True)  # clear contents
   print("fg0 is fg2: ",  fg0 is fg2)  
   print("fg2.texts: ", [t.get_text() for t in fg2.texts])

   # The output:
   # fg0.texts:  ['A fancy plot']
   # fg0 is fg1:  True
   # fg1.texts:  ['A fancy plot', 'Really fancy!']
   # fg0 is fg2:  True
   # fg2.texts:  []


   

