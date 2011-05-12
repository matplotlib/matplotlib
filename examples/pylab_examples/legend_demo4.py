import matplotlib.pyplot as plt

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

plt.show()
    
