"""
===================
Drawstyle Parameter
===================

This example demonstrates the drawstyle parameter that can be used to 
modify how the points are connected. 

"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,14,0.1)
y = np.sin(x/2)


fig, ax = plt.subplots(figsize=(15,5))
_ = ax.plot(x, y, drawstyle='steps-post')
_ = ax.set_aspect('equal')
plt.show()
