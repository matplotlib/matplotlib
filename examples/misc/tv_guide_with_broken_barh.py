"""
Illustrates how the TV Guide heatmap can be rendered with the use of broken_barh, ScalarMappable & clip_box
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import TransformedBbox, Bbox

data = [
    [424, 1420053300, 1420066200, "Metus donec nulla", "Channel 1", 1466],
    [585, 1420054200, 1420059600, "Morbi purus praesent massa ad morbi penatibus", "Channel 10", 658],
    [585, 1420059600, 1420063200, "Lacus purus duis a neque dapibus est at libero dapibus enim", "Channel 10", 295],
    [589, 1420063200, 1420067400, "Porta metus", "Channel 10", 3598],
    [585, 1420067400, 1420070400, "Purus donec mauris adipiscing fusce cubilia", "Channel 10", 295],
    [585, 1420070400, 1420079400, "Purus lacus", "Channel 10", 295],
    [454, 1420058700, 1420069500, "Fames lacus mauris nibh urna enim tristique in scelerisque odio", "Channel 11", 1474],
    [454, 1420069500, 1420074000, "Lorem morbi natoque lectus erat duis class ut nulla tempor platea", "Channel 11", 12],
    [454, 1420074000, 1420081200, "Fames lorem sit ante et sem condimentum", "Channel 11", 156],
    [99,  1420057800, 1420064100, "Curae nulla ultrices eget pretium", "Channel 2", 1860],
    [541, 1420054200, 1420058700, "Dolor neque proin sapien pharetra orci", "Channel 3", 664],
    [543, 1420058700, 1420061580, "Nulla fusce diam phasellus facilisis felis porttitor", "Channel 3", 596],
    [547, 1420061580, 1420062000, "Ipsum fusce conubia ornare nam adipiscing lacus quis", "Channel 3", 1137],
    [100, 1420062000, 1420066200, "Donec felis nullam consequat id primis neque quisque nibh", "Channel 3", 1860],
    [548, 1420066200, 1420067700, "Velit lorem augue metus aliquet ligula", "Channel 3", 1137],
    [544, 1420067700, 1420070700, "Velit curae turpis ac dapibus at velit duis condimentum", "Channel 3", 596],
    [101, 1420070700, 1420075560, "Vitae augue sapien eu", "Channel 3", 1860],
    [795, 1420054200, 1420059300, "Proin dolor magna amet posuere venenatis", "Channel 4", 645],
    [799, 1420059300, 1420065600, "Massa lorem vivamus", "Channel 4", 784],
    [796, 1420072800, 1420077600, "Porta netus eget mus donec", "Channel 4", 645],
    [553, 1420054200, 1420066500, "Lorem morbi", "Channel 5", 787],
    [554, 1420066500, 1420067100, "Lorem magna diam praesent lacinia egestas congue", "Channel 5", 787],
    [554, 1420067100, 1420070400, "Magna nulla netus a tellus proin ullamcorper lobortis purus a", "Channel 5", 787],
    [554, 1420070400, 1420074000, "Curae etiam fames", "Channel 5", 587],
    [554, 1420074000, 1420077600, "Lorem proin", "Channel 5", 100],
    [554, 1420077600, 1420081200, "Justo curae magna class neque", "Channel 5", 50],
    [637, 1420054200, 1420059600, "Felis nulla leo fusce netus a ante", "Channel 6", 2465],
    [639, 1420059600, 1420068600, "Justo netus", "Channel 6", 6220],
    [639, 1420068600, 1420072200, "Risus massa natoque rutrum ac", "Channel 6", 5251],
    [639, 1420072200, 1420075800, "Class massa per parturient penatibus", "Channel 6", 4632],
    [639, 1420075800, 1420078200, "Massa felis sit tincidunt dolor", "Channel 6", 3232],
    [647, 1420054200, 1420059600, "Etiam risus senectus magna ultrices ipsum", "Channel 7", 650],
    [649, 1420059600, 1420066200, "Morbi proin dictum", "Channel 7", 580],
    [650, 1420066200, 1420067400, "Lorem metus vivamus posuere velit sit purus ligula hymenaeos", "Channel 7", 465],
    [650, 1420067400, 1420071300, "Netus justo nunc metus condimentum", "Channel 7", 580],
    [718, 1420071300, 1420080600, "Nulla donec fames vitae magna", "Channel 7", 123],
    [715, 1420054200, 1420060500, "Proin massa aliquam vel parturient proin urna sit neque lectus auctor leo erat sed suscipit", "Channel 8", 1257],
    [716, 1420060500, 1420065000, "Neque lorem sociis mi ac hendrerit adipiscing natoque", "Channel 8", 1257],
    [717, 1420065000, 1420066740, "Lacus vitae donec quam elementum praesent rutrum aenean", "Channel 8", 1257],
    [718, 1420066740, 1420066860, "Donec risus nam nisi", "Channel 8", 300],
    [718, 1420066860, 1420069200, "Nulla porta nostra in sem fusce lectus nisl dignissim rutrum", "Channel 8", 1257],
    [719, 1420069200, 1420075200, "Justo purus erat id risus id a", "Channel 8", 1257],
    [718, 1420075200, 1420079400, "Ipsum augue enim a risus", "Channel 8", 321],
    [466, 1420053300, 1420066800, "Fames neque porta mollis elit mus nec consectetuer eu", "Channel 9", 1741],
    [466, 1420066800, 1420067400, "Donec velit lacinia blandit cras", "Channel 9", 3000],
    [467, 1420067400, 1420070700, "Ipsum velit nulla dui justo montes eu", "Channel 9", 1741],
    [468, 1420070700, 1420083000, "Massa porta commodo quam eu consectetuer cras commodo", "Channel 9", 1]    
]

# load dataframe
df = DataFrame(data, columns=["id","start_time","end_time","program_title","station_title","view_count"])
# df = pd.read_csv("g.csv")
df['duration'] = df['end_time'] - df['start_time']

# get set of station names
st = df['station_title'].unique() 

fig, ax = plt.subplots()

# xaxis ticks formatting
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: datetime.fromtimestamp(x).strftime('%H:%M')))
ax.xaxis.set_major_locator(MultipleLocator(3600))
# ax.set_xlim(int(datetime(2014,12,31,21,0).strftime('%s')),int(datetime(201,12,31,21,0).strftime('%s'))

# yaxis ticks formatting
ax.set_yticks(np.arange(st.size))
ax.set_yticks(np.arange(st.size)+0.5,minor=True)
ax.set_yticklabels(st,minor=True)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False) 

# colorbar
m = ScalarMappable(cmap = 'YlOrRd') 
m.set_array(df['view_count'])
plt.colorbar(m, ticks=np.linspace(df['view_count'].min(), df['view_count'].max(), num=10), label='Views', pad = 0.01)

def intersect(*args):
    return (max([x[0] for x in args]),min([x[1] for x in args]))

c = 0
for t in st:
    dfs = df[df['station_title']==t]

    bar = map(lambda (i,r): (r['start_time'],r['duration']), dfs.iterrows())
    colors = map(lambda (i,r): m.to_rgba(r['view_count']), dfs.iterrows())
    z = ax.broken_barh(bar, (c,1), facecolor=colors)
    for i,r in dfs.iterrows():
        xargs = intersect([r['start_time'],r['end_time']], ax.get_xlim())
        box = TransformedBbox(Bbox([[xargs[0],c],[xargs[1],c+1]]), ax.transData)
        ax.annotate(r['program_title'], xy=(r['start_time']+200,c+0.45), clip_box=box) # 200, 0.45 simple text alingments in the cell
    c += 1
plt.show()