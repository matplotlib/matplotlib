GridSpec items can now add subplots to their parent Figure directly
```````````````````````````````````````````````````````````````````

`SubplotSpec` gained an ``add_subplot`` method, which allows one to write ::

   fig = plt.figure()
   gs = fig.add_gridspec(2, 2)
   gs[0, 0].add_subplot()  # instead of `fig.add_subplot(gs[0, 0])`
