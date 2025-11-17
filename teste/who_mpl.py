import sys, inspect, matplotlib as mpl
import matplotlib.backends.backend_svg as bsvg
import matplotlib.backends.backend_pdf as bpdf
print("PY =", sys.executable)
print("mpl =", mpl.__file__)
print("svg =", inspect.getsourcefile(bsvg))
print("pdf =", inspect.getsourcefile(bpdf))
