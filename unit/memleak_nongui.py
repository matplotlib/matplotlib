import os,matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.cbook import report_memory

def plot():
    fig = Figure()
    i = 0
    while True:
        print i,report_memory(i)
        fig.clf()
        ax = fig.add_axes([0.1,0.1,0.7,0.7])
        ax.plot([1,2,3])
        i += 1

if __name__ == '__main__': plot()
