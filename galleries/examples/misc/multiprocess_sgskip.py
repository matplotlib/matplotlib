"""
===============
Multiprocessing
===============

Demo of using multiprocessing for generating data in one process and
plotting in another.

Written by Robert Cimrman
"""

# sphinx_gallery_thumbnail_path = "_static/multiprocess.png"
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# %%
#
# Processing Class
# ================
#
# This class plots data it receives from a pipe.
#


class ProcessPlotter:
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()


class NBPlot:
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, x, y, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send((x, y))


if __name__ == '__main__':
    np.random.seed(19680801)
    pl = NBPlot()
    for ii in range(20):
        x = ii
        y = np.random.random()
        pl.plot(x, y)
        time.sleep(0.5)
    pl.plot(0, 0, True)
