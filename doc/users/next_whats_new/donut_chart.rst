Doughnut Charts 
-----------------------------------

A doughnut chart is essentially a pie chart with an area of the center cut out. It is one of the most aesthetically pleasing to look at charts that there is. In Matplotlib, it is apparent there there is no consolidated method of drawing a doughnut chart. With our changes we solve this problem.

A parameter, donut, is added to the pie class. Donut is a dictionary with three possible parameters: breaks, width, and callout. Breaks specify where to break the dataset into multiple layers, width specifies how much of each slice to show and callout specifies the text in the middle of the donut chart.

… code block:: python
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    # Set values for the chart
    radius = 1
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']
    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Set the breaks
    callout = 'Comparison \n 50%'
    breaks = [1, 3]
    width = 0.3

    # Draw the chart
    pie, _ = ax.pie([75, 25, 15, 85, 45, 55], radius=radius, colors=colors, labels=labels,
                    donut={'breaks': breaks, 'width': width, 'callout': callout})
    ax.axis('equal')


