import matplotlib
matplotlib.use('WX')
	
from matplotlib.backends import Figure, Toolbar
from matplotlib.axes import Subplot
import Numeric as numpy
	
from wxPython.wx import *
	
class PlotFigure(wxFrame):
    def __init__(self):
        wxFrame.__init__(self, None, -1, "Test embedded wxFigure")
        self.fig = Figure(self, -1, (5,4), 72)
        self.toolbar = Toolbar(self.fig)
        sizer = wxBoxSizer(wxVERTICAL)
        sizer.Add(self.fig, 0, wxTOP)
        sizer.Add(self.toolbar, 1, wxGROW)
        self.SetSizer(sizer)
        self.Fit()
            
    def plot_data(self):
        a = Subplot(self.fig, 111)
        t = numpy.arange(0.0,3.0,0.01)
        s = numpy.sin(2*numpy.pi*t)
        a.plot(t,s)
        self.fig.add_axis(a)
        self.toolbar.update()
        self.fig.draw()
    
if __name__ == '__main__':
    app = wxPySimpleApp()
    frame = PlotFigure()
    frame.plot_data()
    frame.Show(true)
    app.MainLoop()
