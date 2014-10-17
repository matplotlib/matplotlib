# -*- coding: utf-8 -*-
"""
Demo of unicode support in text and labels.
"""
from __future__ import unicode_literals

import matplotlib.pyplot as plt


plt.title('Développés et fabriqués')
plt.xlabel("réactivité nous permettent d'être sélectionnés et adoptés")
plt.ylabel('André was here!')
plt.text(0.2, 0.8, 'Institut für Festkörperphysik', rotation=45)
plt.text(0.4, 0.2, 'AVA (check kerning)')

plt.show()
