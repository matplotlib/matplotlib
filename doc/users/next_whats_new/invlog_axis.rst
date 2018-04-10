Inverted/reverse logarithmic axis
---------------------------------
When plotting a quantity on a logarithmic scale for which the reciprocal value
`1/x` is the actual quantity of interest, it makes more sense to reverse the
minor ticks on the logarithmic axis. One area where this is of immediate use is
when doing Fourier analysis if one is interested in (time/length-)scales
instead of frequencies. This is shown in the new `inv_log_demo.py` example in
the gallery.