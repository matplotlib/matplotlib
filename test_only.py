from distutils.core import setup

import os

baseline_images = [
    'baseline_images/%s/*' % x
    for x in os.listdir('lib/matplotlib/tests/baseline_images')]

baseline_images += [
        'mpltest.ttf',
        'test_rcparams.rc'
    ]

setup(name='matplotlib.tests',
      packages=['matplotlib.tests'],
      package_dir={'matplotlib.tests': 'lib/matplotlib/tests'},
      package_data={'matplotlib.tests': baseline_images}
)
