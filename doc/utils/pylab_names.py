from __future__ import print_function
"""
autogenerate some tables for pylab namespace
"""
from pylab import *
d = locals()

modd = dict()
for k in sorted(d):
    o = d[k]
    if not callable(o):
        continue
    doc = getattr(o, '__doc__', None)
    if doc is not None:
        doc = ' - '.join([line for line in doc.split('\n') if line.strip()][:2])

    mod = getattr(o, '__module__', None)
    if mod is None:
        mod = 'unknown'

    if mod is not None:
        if mod.startswith('matplotlib'):
            if k[0].isupper():
                k = ':class:`~%s.%s`'%(mod, k)
            else:
                k = ':func:`~%s.%s`'%(mod, k)
            mod = ':mod:`%s`'%mod
        elif mod.startswith('numpy'):
            #k = '`%s <%s>`_'%(k, 'http://scipy.org/Numpy_Example_List_With_Doc#%s'%k)
            k = '`%s <%s>`_'%(k, 'http://sd-2116.dedibox.fr/pydocweb/doc/%s.%s'%(mod, k))


    if doc is None: doc = 'TODO'

    mod, k, doc = mod.strip(), k.strip(), doc.strip()[:80]
    modd.setdefault(mod, []).append((k, doc))

for mod in sorted(modd):
    border = '*' * len(mod)
    print(mod)
    print(border)

    print()
    funcs, docs = zip(*modd[mod])
    maxfunc = max(len(f) for f in funcs)
    maxdoc = max(40, max(len(d) for d in docs))
    border = '=' * maxfunc + ' ' + '=' * maxdoc
    print(border)
    print('{:<{}} {:<{}}'.format('symbol', maxfunc, 'description', maxdoc))
    print(border)
    for func, doc in modd[mod]:
        print('{:<{}} {:<{}}'.format(func, maxfunc, doc, maxdoc))
    print(border)
    print()
