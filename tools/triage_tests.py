"""
This is a developer utility to help analyze and triage image
comparison failures.

It allows the failures to be quickly compared against the expected
results, and the new results to be either accepted (by copying the new
results to the source tree) or rejected (by copying the original
expected result to the source tree).

To start:

    If you ran the tests from the top-level of a source checkout, simply run:

        python tools/triage_tests.py

    Otherwise, you can manually select the location of `result_images`
    on the commandline.

Keys:

    left/right: Move between test, expected and diff images
    up/down:    Move between tests
    A:          Accept test.  Copy the test result to the source tree.
    R:          Reject test.  Copy the expected result to the source tree.
"""

from __future__ import unicode_literals

import os
import shutil
import sys

from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets


# matplotlib stores the baseline images under two separate subtrees,
# but these are all flattened in the result_images directory.  In
# order to find the source, we need to search for a match in one of
# these two places.

BASELINE_IMAGES = [
    os.path.join('lib', 'matplotlib', 'tests', 'baseline_images'),
    os.path.join('lib', 'mpl_toolkits', 'tests', 'baseline_images')
    ]


# Non-png image extensions

exts = ['pdf', 'svg']


class Thumbnail(QtWidgets.QFrame):
    """
    Represents one of the three thumbnails at the top of the window.
    """
    def __init__(self, parent, index, name):
        super(Thumbnail, self).__init__()

        self.parent = parent
        self.index = index

        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel(name)
        label.setAlignment(QtCore.Qt.AlignHCenter |
                           QtCore.Qt.AlignVCenter)
        layout.addWidget(label, 0)

        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignHCenter |
                                QtCore.Qt.AlignVCenter)
        self.image.setMinimumSize(800/3, 500/3)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def mousePressEvent(self, ev):
        self.parent.set_large_image(self.index)


class ListWidget(QtWidgets.QListWidget):
    """
    The list of files on the left-hand side
    """
    def __init__(self, parent):
        super(ListWidget, self).__init__()
        self.parent = parent
        self.currentRowChanged.connect(self.change_row)

    def change_row(self, i):
        self.parent.set_entry(i)


class EventFilter(QtCore.QObject):
    # A hack keypresses can be handled globally and aren't swallowed
    # by the individual widgets

    def __init__(self, window):
        super(EventFilter, self).__init__()
        self.window = window

    def eventFilter(self, receiver, event):
        if event.type() == QtCore.QEvent.KeyPress:
            self.window.keyPressEvent(event)
            return True
        else:
            return False
            return super(EventFilter, self).eventFilter(receiver, event)


class Dialog(QtWidgets.QDialog):
    """
    The main dialog window.
    """
    def __init__(self, entries):
        super(Dialog, self).__init__()

        self.entries = entries
        self.current_entry = -1
        self.current_thumbnail = -1

        event_filter = EventFilter(self)
        self.installEventFilter(event_filter)

        self.filelist = ListWidget(self)
        self.filelist.setMinimumWidth(400)
        for entry in entries:
            self.filelist.addItem(entry.display)

        images_box = QtWidgets.QWidget()
        images_layout = QtWidgets.QVBoxLayout()
        thumbnails_box = QtWidgets.QWidget()
        thumbnails_layout = QtWidgets.QHBoxLayout()
        self.thumbnails = []
        for i, name in enumerate(('test', 'expected', 'diff')):
            thumbnail = Thumbnail(self, i, name)
            thumbnails_layout.addWidget(thumbnail)
            self.thumbnails.append(thumbnail)
        thumbnails_box.setLayout(thumbnails_layout)
        self.image_display = QtWidgets.QLabel()
        self.image_display.setAlignment(QtCore.Qt.AlignHCenter |
                                        QtCore.Qt.AlignVCenter)
        self.image_display.setMinimumSize(800, 500)
        images_layout.addWidget(thumbnails_box, 3)
        images_layout.addWidget(self.image_display, 6)
        images_box.setLayout(images_layout)

        buttons_box = QtWidgets.QWidget()
        buttons_layout = QtWidgets.QHBoxLayout()
        accept_button = QtWidgets.QPushButton("Accept (A)")
        accept_button.clicked.connect(self.accept_test)
        buttons_layout.addWidget(accept_button)
        reject_button = QtWidgets.QPushButton("Reject (R)")
        reject_button.clicked.connect(self.reject_test)
        buttons_layout.addWidget(reject_button)
        buttons_box.setLayout(buttons_layout)
        images_layout.addWidget(buttons_box)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.filelist, 3)
        main_layout.addWidget(images_box, 6)

        self.setLayout(main_layout)

        self.setWindowTitle("matplotlib test triager")

        self.set_entry(0)

    def set_entry(self, index):
        if self.current_entry == index:
            return

        self.current_entry = index
        entry = self.entries[index]

        self.pixmaps = []
        for fname, thumbnail in zip(entry.thumbnails, self.thumbnails):
            pixmap = QtGui.QPixmap(fname)
            scaled_pixmap = pixmap.scaled(
                thumbnail.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation)
            thumbnail.image.setPixmap(scaled_pixmap)
            self.pixmaps.append(scaled_pixmap)

        self.set_large_image(0)
        self.filelist.setCurrentRow(self.current_entry)

    def set_large_image(self, index):
        self.thumbnails[self.current_thumbnail].setFrameShape(0)
        self.current_thumbnail = index
        pixmap = QtGui.QPixmap(self.entries[self.current_entry]
                                   .thumbnails[self.current_thumbnail])
        self.image_display.setPixmap(pixmap)
        self.thumbnails[self.current_thumbnail].setFrameShape(1)

    def accept_test(self):
        self.entries[self.current_entry].accept()
        self.filelist.currentItem().setText(
            self.entries[self.current_entry].display)
        # Auto-move to the next entry
        self.set_entry(min((self.current_entry + 1), len(self.entries) - 1))

    def reject_test(self):
        self.entries[self.current_entry].reject()
        self.filelist.currentItem().setText(
            self.entries[self.current_entry].display)
        # Auto-move to the next entry
        self.set_entry(min((self.current_entry + 1), len(self.entries) - 1))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Left:
            self.set_large_image((self.current_thumbnail - 1) % 3)
        elif e.key() == QtCore.Qt.Key_Right:
            self.set_large_image((self.current_thumbnail + 1) % 3)
        elif e.key() == QtCore.Qt.Key_Up:
            self.set_entry(max(self.current_entry - 1, 0))
        elif e.key() == QtCore.Qt.Key_Down:
            self.set_entry(min(self.current_entry + 1, len(self.entries) - 1))
        elif e.key() == QtCore.Qt.Key_A:
            self.accept_test()
        elif e.key() == QtCore.Qt.Key_R:
            self.reject_test()
        else:
            super(Dialog, self).keyPressEvent(e)


class Entry(object):
    """
    A model for a single image comparison test.
    """
    def __init__(self, path, root, source):
        self.source = source
        self.root = root
        self.dir, fname = os.path.split(path)
        self.reldir = os.path.relpath(self.dir, self.root)
        self.diff = fname

        basename = fname[:-len('-failed-diff.png')]
        for ext in exts:
            if basename.endswith('_' + ext):
                display_extension = '_' + ext
                extension = ext
                basename = basename[:-4]
                break
        else:
            display_extension = ''
            extension = 'png'

        self.basename = basename
        self.extension = extension
        self.generated = basename + '.' + extension
        self.expected = basename + '-expected.' + extension
        self.expected_display = (basename + '-expected' + display_extension +
                                 '.png')
        self.generated_display = basename + display_extension + '.png'
        self.name = os.path.join(self.reldir, self.basename)
        self.destdir = self.get_dest_dir(self.reldir)

        self.thumbnails = [
            self.generated_display,
            self.expected_display,
            self.diff
            ]
        self.thumbnails = [os.path.join(self.dir, x) for x in self.thumbnails]

        self.status = 'unknown'

        if self.same(os.path.join(self.dir, self.generated),
                     os.path.join(self.destdir, self.generated)):
            self.status = 'accept'

    def same(self, a, b):
        """
        Returns True if two files have the same content.
        """
        with open(a, 'rb') as fd:
            a_content = fd.read()
        with open(b, 'rb') as fd:
            b_content = fd.read()
        return a_content == b_content

    def copy_file(self, a, b):
        """
        Copy file from a to b.
        """
        print("copying: {} to {}".format(a, b))
        shutil.copyfile(a, b)

    def get_dest_dir(self, reldir):
        """
        Find the source tree directory corresponding to the given
        result_images subdirectory.
        """
        for baseline_dir in BASELINE_IMAGES:
            path = os.path.join(self.source, baseline_dir, reldir)
            if os.path.isdir(path):
                return path
        raise ValueError("Can't find baseline dir for {}".format(reldir))

    @property
    def display(self):
        """
        Get the display string for this entry.  This is the text that
        appears in the list widget.
        """
        status_map = {'unknown': '\N{BALLOT BOX}',
                      'accept':  '\N{BALLOT BOX WITH CHECK}',
                      'reject':  '\N{BALLOT BOX WITH X}'}
        box = status_map[self.status]
        return '{} {} [{}]'.format(box, self.name, self.extension)

    def accept(self):
        """
        Accept this test by copying the generated result to the
        source tree.
        """
        a = os.path.join(self.dir, self.generated)
        b = os.path.join(self.destdir, self.generated)
        self.copy_file(a, b)
        self.status = 'accept'

    def reject(self):
        """
        Reject this test by copying the expected result to the
        source tree.
        """
        a = os.path.join(self.dir, self.expected)
        b = os.path.join(self.destdir, self.generated)
        self.copy_file(a, b)
        self.status = 'reject'


def find_failing_tests(result_images, source):
    """
    Find all of the failing tests by looking for files with
    `-failed-diff` at the end of the basename.
    """
    entries = []
    for root, dirs, files in os.walk(result_images):
        for fname in files:
            basename, ext = os.path.splitext(fname)
            if basename.endswith('-failed-diff'):
                path = os.path.join(root, fname)
                entry = Entry(path, result_images, source)
                entries.append(entry)
    entries.sort(key=lambda x: x.name)
    return entries


def launch(result_images, source):
    """
    Launch the GUI.
    """
    entries = find_failing_tests(result_images, source)

    if len(entries) == 0:
        print("No failed tests")
        sys.exit(0)

    app = QtWidgets.QApplication(sys.argv)
    dialog = Dialog(entries)
    dialog.show()
    filter = EventFilter(dialog)
    app.installEventFilter(filter)
    sys.exit(app.exec_())


if __name__ == '__main__':
    import argparse

    source_dir = os.path.join(os.path.dirname(__file__), '..')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Triage image comparison test failures.

If no arguments are provided, it assumes you ran the tests at the
top-level of a source checkout as `python tests.py`.

Keys:
    left/right: Move between test, expected and diff images
    up/down:    Move between tests
    A:          Accept test.  Copy the test result to the source tree.
    R:          Reject test.  Copy the expected result to the source tree.
""")
    parser.add_argument("result_images", type=str, nargs='?',
                        default=os.path.join(source_dir, 'result_images'),
                        help="The location of the result_images directory")
    parser.add_argument("source", type=str, nargs='?', default=source_dir,
                        help="The location of the matplotlib source tree")
    args = parser.parse_args()

    launch(args.result_images, args.source)
