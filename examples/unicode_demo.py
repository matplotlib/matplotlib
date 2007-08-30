#!/usr/bin/python
# -*- coding: utf-8 -*-

from pylab import *
plot([1,2,4])
title( unicode('Développés et fabriqués', 'latin-1') )
xlabel( unicode("réactivité nous permettent d'être sélectionnés et adoptés", 'latin-1') )
ylabel( unicode('Andr\xc3\xa9 was here!', 'utf-8') )
text( 0.5, 2.5, unicode('Institut für Festkörperphysik', 'latin-1'), rotation=45)
text( 1, 1.5, u'AVA (check kerning)')

savefig('test')
show()
