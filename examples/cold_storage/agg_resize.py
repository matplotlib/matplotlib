# test the resizing methods in the agg wrapper

import matplotlib.agg as agg


imMatrix = agg.trans_affine(1,0,0,1,0,0)
interp = agg.span_interpolator_linear(imMatrix);
