Pie charts are now circular by default
--------------------------------------
We acknowledge that the majority of people do not like egg-shaped pies.
Therefore, an axes to which a pie chart is plotted will be set to have 
equal aspect ratio by default. This ensures that the pie appears circular
independent on the axes size or units. To revert to the previous behaviour
you may set the axes' aspect to automatic, ax.set_aspect("auto") or 
plt.axis("auto").