
add_languages('objc', native: false)
py3.extension_module(
  '_macos',
  sources: files(
    '_macos.m',
    'MPLAppDelegate.m',
    'MPLFigureCanvas.m',
    'MPLFigureManager.m',
    'MPLNavigationToolbar2.m',
    'MPLUtils.m',
  ),
  dependencies: dependency('appleframeworks', modules: 'Cocoa'),
  override_options: ['werror=true'],
  objc_args: ['-fobjc-arc'],
  install: true,
  subdir: 'matplotlib/backends',
)
