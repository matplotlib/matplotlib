python_sources = [
  '__init__.py',
  'backend_agg.py',
  'backend_cairo.py',
  '_backend_gtk.py',
  'backend_gtk3.py',
  'backend_gtk3agg.py',
  'backend_gtk3cairo.py',
  'backend_gtk4.py',
  'backend_gtk4agg.py',
  'backend_gtk4cairo.py',
  'backend_macosx.py',
  'backend_mixed.py',
  'backend_nbagg.py',
  '_backend_pdf_ps.py',
  'backend_pdf.py',
  'backend_pgf.py',
  'backend_ps.py',
  'backend_qt.py',
  'backend_qtagg.py',
  'backend_qtcairo.py',
  'backend_qt5.py',
  'backend_qt5agg.py',
  'backend_qt5cairo.py',
  'backend_svg.py',
  'backend_template.py',
  '_backend_tk.py',
  'backend_tkagg.py',
  'backend_tkcairo.py',
  'backend_webagg.py',
  'backend_webagg_core.py',
  'backend_wx.py',
  'backend_wxagg.py',
  'backend_wxcairo.py',
  'qt_compat.py',
]

typing_sources = [
  # Compiled extension types.
  '_backend_agg.pyi',
  '_macosx.pyi',
  '_tkagg.pyi',
]

py3.install_sources(python_sources, typing_sources,
                    subdir: 'matplotlib/backends')

subdir('qt_editor')
subdir('web_backend')
