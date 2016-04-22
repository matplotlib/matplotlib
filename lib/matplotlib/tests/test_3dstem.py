import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['3dstemplot_default'],
                  extensions=['png'])
def test_3dstemplot_default():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.power(x, 2)
    markerline, stemlines, baseline = ax.stem(x, y, z)


@image_comparison(baseline_images=['3dstemplot_rotate_along_x'],
                  extensions=['png'])
def test_3dstemplot_rotate_along_x():
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.power(x, 2)
    markerline, stemlines, baseline = ax2.stem(x, y, z, zdir='-x')


@image_comparison(baseline_images=['3dstemplot_rotate_along_y'],
                  extensions=['png'])
def test_3dstemplot_rotate_along_y():
    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d')
    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.power(x, 2)
    markerline, stemlines, baseline = ax3.stem(x, y, z, zdir='-y')


@image_comparison(baseline_images=['2plane_default'],
                  extensions=['png'])
def test_2plane_default():
    fig4 = plt.figure()
    ax4 = fig4.gca(projection='3d')
    x = np.linspace(-np.pi/2, np.pi/2, 40)
    y = [1]*len(x)
    z = np.cos(x)
    markerline, stemlines, baseline = ax4.stem(x, y, z, '-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 1)
    
@image_comparison(baseline_images=['2plane_zdir_x'],
                  extensions=['png'])
def test_2plane_zdir_x():
    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    x = np.linspace(-np.pi/2, np.pi/2, 40)
    y = [1]*len(x)
    z = np.cos(x)
    markerline, stemlines, baseline = ax5.stem(x, y, z, '-.', zdir='x')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 1)
    
@image_comparison(baseline_images=['2plane_zdir_y'],
                  extensions=['png'])
def test_3dstemplot_zdir_y():
    fig6 = plt.figure()
    ax6 = fig6.gca(projection='3d')
    x = np.linspace(-np.pi/2, np.pi/2, 40)
    y = [1]*len(x)
    z = np.cos(x)
    markerline, stemlines, baseline = ax6.stem(x, y, z, '-.', zdir='y')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 1)
    
@image_comparison(baseline_images=['2plane_zdir_z'],
                  extensions=['png'])
def test_3dstemplot_zdir_z():
    fig7 = plt.figure()
    ax7 = fig7.gca(projection='3d')
    x = np.linspace(-np.pi/2, np.pi/2, 40)
    y = [1]*len(x)
    z = np.cos(x)
    markerline, stemlines, baseline = ax7.stem(x, y, z, '-.', zdir='z')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 1)
