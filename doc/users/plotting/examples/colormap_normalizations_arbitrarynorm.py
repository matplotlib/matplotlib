"""
Demonstration of using the ArbitraryNorm classes for normalization.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x=np.linspace(0,16*np.pi,1024)
y=np.linspace(-1,1,512)
X,Y=np.meshgrid(x,y)

data=np.zeros(X.shape)

data[Y>0]=np.cos(X[Y>0])*Y[Y>0]**2

for i,val in enumerate(np.arange(-1,1.1,0.2)):
    if val<0:data[(X>(i*(50./11)))*(Y<0)]=val
    if val>0:data[(X>(i*(50./11)))*(Y<0)]=val*2
            
figsize=(16,10)
cmap=cm.gist_rainbow

plt.figure(figsize=figsize)
plt.pcolormesh(x,y,data,cmap=cmap)
plt.title('Linear Scale')
plt.colorbar(format='%.3g')
plt.xlim(0,16*np.pi)
plt.ylim(-1,1)

plt.figure(figsize=figsize)
norm=colors.ArbitraryNorm(fpos=(lambda x: x**0.2),
                          fposinv=(lambda x: x**5),
                          fneg=(lambda x: x**0.5),
                          fneginv=(lambda x: x**2),
                          center=0.4)
plt.pcolormesh(x,y,data,cmap=cmap,norm=norm)
plt.title('Arbitrary norm')
plt.colorbar(ticks=norm.ticks(),format='%.3g')
plt.xlim(0,16*np.pi)
plt.ylim(-1,1)

plt.figure(figsize=figsize)
norm=colors.PositiveArbitraryNorm(vmin=0,
                                  fpos=(lambda x: x**0.5),
                                  fposinv=(lambda x: x**2))
plt.pcolormesh(x,y,data,cmap=cmap,norm=norm)
plt.title('Positive arbitrary norm')
plt.colorbar(ticks=norm.ticks(),format='%.3g')
plt.xlim(0,16*np.pi)
plt.ylim(-1,1)

plt.figure(figsize=figsize)
norm=colors.NegativeArbitraryNorm(vmax=0,
                                  fneg=(lambda x: x**0.5),
                                  fneginv=(lambda x: x**2))
plt.pcolormesh(x,y,data,cmap=cmap,norm=norm)
plt.title('Negative arbitrary norm')
plt.colorbar(ticks=norm.ticks(),format='%.3g')
plt.xlim(0,16*np.pi)
plt.ylim(-1,1)