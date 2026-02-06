New Wilkinson tick locator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new tick locator `.WilkinsonLocator` based on the extended Wilkinson algorithm
by Talbot, Lin and Hanrahan [1] has been added to the `matplotlib.ticker`
module. This locator takes into consideration four primary critera:
- Simplicity: how simple the tick values are (e.g. round numbers)
- Coverage: how well the ticks cover the data range
- Density: how well the number of ticks matches the target number of ticks
- Legibility: how legible the tick labels are (e.g. including zero, avoiding long labels)

[1] Talbot, J., Lin, S., & Hanrahan, P. (2010). An Extension of Wilkinson's Algorithm for Positioning Tick Labels on Axes. 
    IEEE Transactions on Visualization and Computer Graphics, 16(6), 1036-1047. https://doi.org/10.1109/TVCG.2010.115
