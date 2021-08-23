import matplotlib
matplotlib.use('pdf')

import matplotlib.font_manager as fm
from matplotlib import pyplot as plt

fm.fontManager.ttflist.append(fm.FontEntry(fname='fonts/NotoSansCJKsc-Medium.otf', name='Noto Sans SC Medium'))
matplotlib.rc('font', family=['DejaVu Sans', 'Noto Sans SC Medium'])

for ft in (3, 42):
    matplotlib.rc('pdf', fonttype=ft)
    plt.text(0.5, 0.5, 'Hello, Привет, 你好')
    plt.savefig(f'hello-{ft}.pdf')

    
