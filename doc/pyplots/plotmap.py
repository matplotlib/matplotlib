# make plot of etopo bathymetry/topography data on
# lambert conformal conic map projection, drawing coastlines, state and
# country boundaries, and parallels/meridians.

# the data is interpolated to the native projection grid.
import os
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np

from pylab import title, colorbar, show, axes, cm, arange, figure, \
                  text
from matplotlib.cbook import get_sample_data

# read in topo data (on a regular lat/lon grid)
# longitudes go from 20 to 380.
topoin = np.loadtxt(get_sample_data('screenshots/etopo20data.gz').name)
lons = np.loadtxt(get_sample_data('screenshots/etopo20lons.gz').name)
lats = np.loadtxt(get_sample_data('screenshots/etopo20lats.gz').name)
# shift data so lons go from -180 to 180 instead of 20 to 380.
topoin,lons = shiftgrid(180.,topoin,lons,start=False)

# setup of basemap ('lcc' = lambert conformal conic).
# use major and minor sphere radii from WGS84 ellipsoid.
m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.,urcrnrlon=-2.566,urcrnrlat=46.352,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=50.,lon_0=-107.)
# transform to nx x ny regularly spaced native projection grid
nx = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
topodat,x,y = m.transform_scalar(topoin,lons,lats,nx,ny,returnxy=True)
# create the figure.
fig=figure(figsize=(6,6))
# add an axes, leaving room for colorbar on the right.
ax = fig.add_axes([0.1,0.1,0.7,0.7])
# plot image over map with imshow.
im = m.imshow(topodat,cm.jet)
# setup colorbar axes instance.
# for matplotlib 0.91 and earlier, could do l,b,w,h = ax.get_position()
# for post 0.91, pos = ax.get_position(); l,b,w,h = pos.bounds
# this works for both.
pos = ax.get_position()
l, b, w, h = getattr(pos, 'bounds', pos)
cax = axes([l+w+0.075, b, 0.05, h])
colorbar(cax=cax) # draw colorbar
axes(ax)  # make the original axes current again
# plot blue dot on boulder, colorado and label it as such.
xpt,ypt = m(-104.237,40.125)
m.plot([xpt],[ypt],'bo')
text(xpt+100000,ypt+100000,'Boulder')
# draw coastlines and political boundaries.
m.drawcoastlines()
m.drawcountries()
m.drawstates()
# draw parallels and meridians.
# label on left, right and bottom of map.
parallels = arange(0.,80,20.)
m.drawparallels(parallels,labels=[1,1,0,1])
meridians = arange(10.,360.,30.)
m.drawmeridians(meridians,labels=[1,1,0,1])
# set title.
title('ETOPO Topography - Lambert Conformal Conic')
show()
