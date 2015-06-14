from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from nose.tools import assert_equal, assert_in
import matplotlib.type1font as t1f
import os.path
import difflib
import hashlib


def sha1(data):
    hash = hashlib.sha1()
    hash.update(data)
    return hash.hexdigest()


def test_Type1Font():
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    font = t1f.Type1Font(filename)
    slanted = font.transform({'slant': 1})
    condensed = font.transform({'extend': 0.5})
    assert_equal(map(sha1, font.parts),
                 ['f4ce890d648e67d413a91b3109fe67a732ced96f',
                  'af46adb6c528956580c125c6abf3b5eb9983bbc1',
                  'e2538a88a810bc207cfa1194b658ee8967042db8'])
    assert_equal(font.parts[1:], slanted.parts[1:])
    assert_equal(font.parts[1:], condensed.parts[1:])

    differ = difflib.Differ()
    diff = set(differ.compare(font.parts[0].splitlines(),
                              slanted.parts[0].splitlines()))
    for line in (
         # Removes UniqueID
         '- FontDirectory/CMR10 known{/CMR10 findfont dup/UniqueID known{dup',
         '+ FontDirectory/CMR10 known{/CMR10 findfont dup',
         # Alters FontMatrix
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         '+ /FontMatrix [0.001 0.0 0.001 0.001 0.0 0.0]readonly def',
         # Alters ItalicAngle
         '-  /ItalicAngle 0 def',
         '+  /ItalicAngle -45.0 def'):
        assert_in(line, diff, 'diff to slanted font must contain %s' % line)

    diff = set(differ.compare(font.parts[0].splitlines(),
                              condensed.parts[0].splitlines()))
    for line in (
         # Removes UniqueID
         '- FontDirectory/CMR10 known{/CMR10 findfont dup/UniqueID known{dup',
         '+ FontDirectory/CMR10 known{/CMR10 findfont dup',
         # Alters FontMatrix
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         '+ /FontMatrix [0.0005 0.0 0.0 0.001 0.0 0.0]readonly def'):
        assert_in(line, diff, 'diff to condensed font must contain %s' % line)
