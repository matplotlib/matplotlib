Axes.violinplot and cbook.violin_stats safely handle empty datasets
-------------------------------------------------------------------

`~matplotlib.axes.Axes.violinplot` and `matplotlib.cbook.violin_stats` now safely handle empty datasets without crashing and ignore masked and non-finite values.
