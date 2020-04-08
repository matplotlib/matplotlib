A `.QuiverKey` can now be used used as an entry in the legend
-----------------------------------------------------------

eg.::

    qk = ax.quiverkey(Q, 0.9, 0.8, U=10, label='QK length = 10', labelpos='E',
                  color='red')
    ax.legend([qk])
