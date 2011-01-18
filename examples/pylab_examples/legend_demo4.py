import matplotlib.pyplot as plt

if 1:

    ax = plt.subplot(111)
    
    b1 = ax.bar([0, 1, 2], [0.2, 0.3, 0.1], width=0.4,
                label="Bar 1", align="center")

    b2 = ax.bar([0.5, 1.5, 2.5], [0.3, 0.2, 0.2], color="red", width=0.4,
                label="Bar 2", align="center")

    err1 = ax.errorbar([0, 1, 2], [2, 3, 1], xerr=0.4, fmt="s",
                       label="test 1")
    err2 = ax.errorbar([0, 1, 2], [3, 2, 4], yerr=0.3, fmt="o",
                       label="test 2")
    err3 = ax.errorbar([0, 1, 2], [1, 1, 3], xerr=0.4, yerr=0.3, fmt="^",
                       label="test 3")

    # legend
    leg1 = plt.legend(loc=1)

    # legend of selected artists
    artists = [b1, err2]
    leg2 = plt.legend(artists, [a.get_label() for a in artists], loc=2)

    # custome handler
    import matplotlib.legend_handler as mlegend_handler
    myhandler = mlegend_handler.HandlerErrorbar(npoints=1)

    leg3 = plt.legend([err1, err3], ["T1", "T2"], loc=3,
                      handler_map={err3:myhandler})

    plt.gca().add_artist(leg1) 
    plt.gca().add_artist(leg2)
   
    plt.show()
    
