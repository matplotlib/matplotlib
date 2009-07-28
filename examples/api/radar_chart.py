import numpy as np

import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes 
from matplotlib.projections import register_projection 

def radar_factory(num_vars, frame='circle'): 
    """Create a radar chart with `num_vars` axes.""" 
    # calculate evenly-spaced axis angles 
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars) 
    # rotate theta such that the first axis is at the top 
    theta += np.pi/2 

    def draw_poly_frame(self, x0, y0, r): 
        # TODO: use transforms to convert (x, y) to (r, theta)
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta] 
        return plt.Polygon(verts, closed=True, edgecolor='k') 
        
    def draw_circle_frame(self, x0, y0, r): 
        return plt.Circle((x0, y0), r) 

    frame_dict = {'polygon': draw_poly_frame, 'circle': draw_circle_frame} 
    if frame not in frame_dict: 
        raise ValueError, 'unknown value for `frame`: %s' % frame 

    class RadarAxes(PolarAxes): 
        """Class for creating a radar chart (a.k.a. a spider or star chart) 
        
        http://en.wikipedia.org/wiki/Radar_chart 
        """ 
        name = 'radar' 
        # use 1 line segment to connect specified points 
        RESOLUTION = 1 
        # define draw_frame method 
        draw_frame = frame_dict[frame] 
        
        def fill(self, *args, **kwargs): 
            """Override fill so that line is closed by default""" 
            closed = kwargs.pop('closed', True) 
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs) 
        
        def plot(self, *args, **kwargs): 
            """Override plot so that line is closed by default""" 
            lines = super(RadarAxes, self).plot(*args, **kwargs) 
            for line in lines: 
                self._close_line(line) 
        
        def _close_line(self, line): 
            x, y = line.get_data() 
            # FIXME: markers at x[0], y[0] get doubled-up 
            if x[0] != x[-1]: 
                x = np.concatenate((x, [x[0]])) 
                y = np.concatenate((y, [y[0]])) 
                line.set_data(x, y) 
        
        def set_varlabels(self, labels): 
            self.set_thetagrids(theta * 180/np.pi, labels) 
        
        def _gen_axes_patch(self): 
            x0, y0 = (0.5, 0.5) 
            r = 0.5 
            return self.draw_frame(x0, y0, r)
        
    register_projection(RadarAxes) 
    return theta 


if __name__ == '__main__': 
    #The following data is from the Denver Aerosol Sources and Health study. 
    #See  doi:10.1016/j.atmosenv.2008.12.017    
    #
    #The data are pollution source profile estimates for five modeled pollution
    #sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical species.
    #The radar charts are experimented with here to see if we can nicely 
    #visualize how the modeled source profiles change across four scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #     Sulfate
    #     Nitrate
    #     Elemental Carbon (EC)
    #     Organic Carbon fraction 1 (OC)
    #     Organic Carbon fraction 2 (OC2)
    #     Organic Carbon fraction 3 (OC3)
    #     Pyrolized Organic Carbon (OP)
    #  2)Inclusion of gas-phase specie carbon monoxide (CO) 
    #  3)Inclusion of gas-phase specie ozone (O3). 
    #  4)Inclusion of both gas-phase speciesis present...
    N = 9
    theta = radar_factory(N)
    spoke_labels = ['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 
                    'O3']
    f1_base = [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00]
    f1_CO =   [0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00] 
    f1_O3 =   [0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03] 
    f1_both = [0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01] 

    f2_base = [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00]
    f2_CO =   [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00] 
    f2_O3 =   [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00] 
    f2_both = [0.09, 0.95, 0.02, 0.03, 0.00, 0.01, 0.13, 0.06, 0.00] 

    f3_base = [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00]
    f3_CO =   [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00] 
    f3_O3 =   [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00] 
    f3_both = [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00] 
    
    f4_base = [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00]
    f4_CO =   [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00] 
    f4_O3 =   [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95] 
    f4_both = [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88] 

    f5_base = [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]
    f5_CO =   [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00] 
    f5_O3 =   [0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00] 
    f5_both = [0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16] 

    fig = plt.figure(figsize=(9,9))
    # adjust spacing around the subplots
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    title_list = ['Basecase', 'With CO', 'With O3', 'CO & O3']
    data = {'Basecase': [f1_base, f2_base, f3_base, f4_base, f5_base],
            'With CO': [f1_CO, f2_CO, f3_CO, f4_CO, f5_CO],
            'With O3': [f1_O3, f2_O3, f3_O3, f4_O3, f5_O3], 
            'CO & O3': [f1_both, f2_both, f3_both, f4_both, f5_both]}
    colors = ['b', 'r', 'g', 'm', 'y']
    # chemicals range from 0 to 1
    radial_grid = [0.2, 0.4, 0.6, 0.8]
    # If you don't care about the order, you can loop over data_dict.items()
    for n, title in enumerate(title_list):
        ax = fig.add_subplot(2, 2, n+1, projection='radar')
        plt.rgrids(radial_grid)
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(data[title], colors):
            ax.plot(theta, d, color=color) 
            ax.fill(theta, d, facecolor=color, alpha=0.25)  
        ax.set_varlabels(spoke_labels)
    # add legend relative to top-left plot
    plt.subplot(2,2,1)
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')
    plt.figtext(0.5, 0.965,  '5-Factor Solution Profiles Across Four Scenarios', 
               ha='center', color='black', weight='bold', size='large')        
    plt.show()
