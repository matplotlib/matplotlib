from . import test_mathtext
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup


@pytest.mark.parametrize('tests',
[
    (test_mathtext.math_tests),
    (test_mathtext.font_tests)
])
@pytest.mark.parametrize('basename,fontset,extensions',
[
    ('mathtext', 'cm', None),
    ('mathtext', 'stix', None),
    ('mathtext', 'stixsans', None),
    ('mathtext', 'dejavusans', None),
    ('mathtext', 'dejavuserif', None),
    ('mathfont', 'cm', ['png']),
    ('mathfont', 'stix', ['png']),
    ('mathfont', 'stixsans', ['png']),
    ('mathfont', 'dejavusans', ['png']),
    ('mathfont', 'dejavuserif', ['png']),
])
def test_make_set(basename, fontset, tests, extensions):
    def make_test(filename, test):
        @image_comparison(baseline_images=[filename], extensions=extensions)
        def single_test():
            matplotlib.rcParams['mathtext.fontset'] = fontset
            fig = plt.figure(figsize=(5.25, 0.75))
            fig.text(0.5, 0.5, test, horizontalalignment='center',
                     verticalalignment='center')
        func = single_test
        func.__name__ = str("test_" + filename)
        return func

    # We inject test functions into the global namespace, rather than
    # using a generator, so that individual tests can be run more
    # easily from the commandline and so each test will have its own
    # result.
    for i, test in enumerate(tests):
        filename = '%s_%s_%02d' % (basename, fontset, i)
        globals()['test_%s' % filename] = make_test(filename, test)



# def pytest_generate_tests(metafunc):
#     if 'stringinput' in metafunc.fixturenames:
#         metafunc.parametrize("stringinput",
#                              metafunc.config.option.stringinput)
#
import pytest

def pytest_collect_file(parent, path):
    if path.ext == ".py" and path.basename.contains("test_mathtext"):
        return test_mathtextFile(path, parent)

class test_mathtextFile(pytest.File):
    def collect(self):

        make_set('mathtext', )
        math_tests = test_mathtext.math_tests
         fonts = ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif',]

        make_set('mathfont', 'cm', font_tests, ['png'])
        make_set('mathfont', 'stix', font_tests, ['png'])
        make_set('mathfont', 'stixsans', font_tests, ['png'])
        make_set('mathfont', 'dejavusans', font_tests, ['png'])
        make_set('mathfont', 'dejavuserif', font_tests, ['png'])

    for i, test in enumerate(tests):
        filename = '%s_%s_%02d' % (basename, fontset, i)
        globals()['test_%s' % filename] = make_test(filename, test)

        raw = yaml.safe_load(self.fspath.open())
        for name, spec in raw.items():
            yield test_mathtextItem(filename, test, extensions)

class test_mathtextItem(pytest.Item):
    def __init__(self, filename, test, extensions):
        super(test_mathtextItem, self).__init__(name, parent)
        self.filename = filename
        self.test = test
        self.extensions = extensions
    @image_comparison(baseline_images=[self.filename],
                      extensions=self.extensions)
    def runtest(self):
        matplotlib.rcParams['mathtext.fontset'] = fontset
        fig = plt.figure(figsize=(5.25, 0.75))
        fig.text(0.5, 0.5, test, horizontalalignment='center',
                 verticalalignment='center')

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        pass
        # if isinstance(excinfo.value, YamlException):
        #     return "\n".join([
        #         "usecase execution failed",
        #         "   spec failed: %r: %r" % excinfo.value.args[1:3],
        #         "   no further details known at this point."
        #     ])

    def reportinfo(self):
        return self.fspath, 0, "usecase: %s" % self.name

class YamlException(Exception):
    """ custom exception for error reporting. """