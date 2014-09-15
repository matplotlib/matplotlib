def bulleye_plot(data, ax=None, figsize=(12,8), vlim=None, segBold=[]):
    """
    Left Ventricle bull eye for the Left Ventricle according to the 
    American Heart Association (AHA)
    Use Example:
        data = range(17)
        bulleye_plot(data)
    """
    data = np.array(data).ravel()

    if vlim is None:
        vlim = [data.min(), data.max()]

    axnone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.canvas.set_window_title('Left Ventricle Bull Eyes (AHA)')
        axnone = True


    theta = np.linspace(0, 2*np.pi, 768)
    r = np.linspace(0.2, 1, 4)


    # Create the bound for the segment 17
    linewidth = 2
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

    # Create the bounds for the segments  1-12
    for i in range(6):
        theta_i = i * 60 * np.pi/180
        ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

    # Create the bounds for the segmentss 13-16
    for i in range(4):
        theta_i = i * 90 * np.pi/180 - 45*np.pi/180
        ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)





    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:,np.newaxis], 128, axis=1).T
    for i in range(6):
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180 # First segment start at 60 degrees
        theta0 = np.repeat(theta0[:,np.newaxis], 2, axis=1)
        z = np.ones((128,2)) * data[i]
        ax.pcolormesh(theta0, r0, z, vmin=vlim[0], vmax=vlim[1])
        if i+1 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[2],r[3]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[2],r[3]], '-k', lw=linewidth+1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:,np.newaxis], 128, axis=1).T
    for i in range(6):
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180 # First segment start at 60 degrees
        theta0 = np.repeat(theta0[:,np.newaxis], 2, axis=1)
        z = np.ones((128,2)) * data[i+6]
        ax.pcolormesh(theta0, r0, z, vmin=vlim[0], vmax=vlim[1])
        if i+7 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[1],r[2]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[1],r[2]], '-k', lw=linewidth+1)


    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:,np.newaxis], 192, axis=1).T
    for i in range(4):
        theta0 = theta[i*192:i*192+192] + 45*np.pi/180 # First segment start at 45 degrees
        theta0 = np.repeat(theta0[:,np.newaxis], 2, axis=1)
        z = np.ones((192,2)) * data[i+12]
        ax.pcolormesh(theta0, r0, z, vmin=vlim[0], vmax=vlim[1])
        if i+13 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[0],r[1]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[0],r[1]], '-k', lw=linewidth+1)

    #Fill the segments 17
    if data.size == 17:
        r0 = np.array([0, r[0]])
        r0 = np.repeat(r0[:,np.newaxis], theta.size, axis=1).T
        theta0 = np.repeat(theta[:,np.newaxis], 2, axis=1)
        z = np.ones((theta.size,2)) * data[16]
        ax.pcolormesh(theta0, r0, z, vmin=vlim[0], vmax=vlim[1])
        if 17 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)


    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


    #Add legend
    if axnone:
        cm = plt.cm.jet

        #define the bins and normalize
        cNorm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

        ax = fig.add_axes([0.3, 0.04, 0.45, 0.05])
        ticks = [vlim[0], 0, vlim[1]]
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cm, norm=cNorm,
                                       orientation='horizontal', ticks=ticks)

    plt.show()

    if axnone:
        return fig, ax
