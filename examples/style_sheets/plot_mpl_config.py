import matplotlib.pyplot as plt
from matplotlib.config.mpl_config import MPLConfig

user_config = {'lines.linewidth': 10}

mplrc = MPLConfig.from_user_config(user_config)
mplrc.set_defaults()

plt.plot([1, 2, 3])
plt.show()
