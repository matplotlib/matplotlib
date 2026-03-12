"""
A module for reading dvi files output by TeX. Several limitations make
this not (currently) useful as a general-purpose dvi preprocessor, but
it is currently used by the pdf backend for processing usetex text.

Interface::

  with Dvi(filename, 72) as dvi:
      # iterate over pages:
      for page in dvi:
          w, h, d = page.width, page.height, page.descent
          for x, y, font, glyph, width in page.text:
              fontname = font.texname
              pointsize = font.size
              ...
          for x, y, height, width in page.boxes:
              ...
"""

import dataclasses
import enum
import io
import logging
import os
import re
import struct
import subprocess
import sys
import typing
from collections import namedtuple
from functools import cache, cached_property, lru_cache, partial
from pathlib import Path

import fontTools.agl
import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, font_manager
from matplotlib.ft2font import LoadFlags

_log = logging.getLogger(__name__)

# Many dvi related files are looked for by external processes, require
# additional parsing, and are used many times per rendering, which is why they
# are cached using lru_cache().

# Dvi is a bytecode format documented in
# https://ctan.org/pkg/dvitype
# https://texdoc.org/serve/dvitype.pdf/0
#
# The file consists of a preamble, some number of pages, a postamble,
# and a finale. Different opcodes are allowed in different contexts,
# so the Dvi object has a parser state:
#
#   pre:       expecting the preamble
#   outer:     between pages (followed by a page or the postamble,
#              also e.g. font definitions are allowed)
#   inpage:    processing a page
#   post:      within the postamble
#   post_post: state after the postamble (our current implementation
#              just stops reading)
#   finale:    the finale (unimplemented in our current implementation)

_dvistate = enum.Enum('DviState', 'pre outer inpage post post_post finale')


def _read_num(f, nbytes: int, signed: bool, strict=True):
    """Read N bytes from a file as an big-endian number."""
    b = f.read(nbytes)
    if strict:
        assert len(b) == nbytes
    return int.from_bytes(b, "big", signed=signed)


class Ops:
    """
    Low-level tools for reading a DVI file as a sequence of ops.

    This is just using a class for namespacing purposes, don't make instances of it.
    Rather, you want to use functions like Ops.read_file, Ops.read_io, etc.
    """
    Op = namedtuple('Op', 'code name args')

    @dataclasses.dataclass(slots=True)
    class DispatchTable:
        """
        Storage for how to interpret different bytes as operations, and unpack
        their arguments. A table is naturally 256 entries long, covering every
        possible single-byte value. It starts with every entry being a placeholder,
        and the convention is to replace these placeholders with the .op() method.

        The existing provided tables are:
         - Ops.tbl_dvi, which is used for normal DVI files.
         - Ops.tbl_vf_outer, which is used for VF files outside of packets.
         - Ops.tbl_vf_inner, which is used for VF files inside of packets.

        The ability to create other tables is available to anybody, but would
        probably be a niche requirement in practice.
        """
        entries: list = dataclasses.field(
            default_factory=lambda: [('unknown', 0, ['delta'], ['delta'], None)] * 256)

        def op(self,
            bmin: int, bmax: int, opname: str,
            arg_types: str ='', arg_names: str ='', extra=None):
            """
            Can be used standalone, or as a decorator.

            Parameters
            ----------
            bmin : int
                Minimum byte for this op.

            bmax : int
                Maximum byte for this op. This creates an inclusive range.

            opname : str
                The reported symbolic name of the op. This is conventionally used
                for dispatch, like with `.dviread.VM`.

            arg_types : str
                Space-separated series of extraction specifiers (see below).

            arg_names : str
                Space-separated series of argument names, one per extraction specifier.

            extra : Fn[file, **args] -> dict, or None
                An optional callback which extracts additional arguments, based on the
                values of previously extracted arguments.

            Returns
            -------
            decorator : Fn[extra_fn] -> None
                A more ergonomic way to provide the `extra` param, if desired.

            Extraction Specifiers
            ---------------------
            delta : the difference between the current op's code and `bmin`.
            u1 : An unsigned 1-byte number.
            u2 : An unsigned 2-byte number.
            u3 : An unsigned 3-byte number.
            u4 : An unsigned 4-byte number.
            s1 : A signed 1-byte number.
            s2 : A signed 2-byte number.
            s3 : A signed 3-byte number.
            s4 : A signed 4-byte number.
            slen : A signed number `delta` bytes long, or if `delta` is 0, None.
            slen1 : A signed number `delta`+1 bytes long.
            ulen1 : An unsigned number `delta`+1 bytes long.
            olen1 : A number `delta+1` bytes long, which is signed if `delta` == 3.
            fin : Attempt to finish the file by reading up to 7 bytes.
            @x : a byte string `x` bytes long, where `x` is a previous argument.

            >>> from matplotlib.dviread import Ops
            >>> my_table = Ops.DispatchTable()
            >>> with my_table as t:
            ...    t.op(0, 22, 'my_op_name', 'u1 u2', 'arg_a arg_b')
            ...    @t.op(23, 23, 'my_op_2', 'u4', 'length')
            ...    def _extra(f, length: int) -> dict:
            ...        b: bytes = f.read(length)
            ...        return { 'payload': b }
            ...
            ...    # While extra arguments offer a comprehensive escape hatch,
            ...    # using a previous argument as a length is directly supported.
            ...    # So the more idiomatic version of the previous example would be:
            ...    t.op(23, 23, 'my_op_2',
            ...        'u4      @length',
            ...        'length  payload')
            """
            arg_types = (' ' + arg_types).split()
            arg_names = (' ' + arg_names).split()
            entry = (opname, bmin, arg_types, arg_names, extra)
            for i in range(bmin, bmax+1):
                self.entries[i] = entry

            # Optional decorator support
            def decorator(fn):
                entry = (opname, bmin, arg_types, arg_names, fn)
                for i in range(bmin, bmax+1):
                    self.entries[i] = entry
            return decorator

        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    @classmethod
    def read_op(cls, f, table: DispatchTable) -> Op | None:
        """Returns None if we've run out of file."""
        opcode = f.read(1)
        if not opcode:
            return None
        opcode = int(opcode[0])
        entry = table.entries[opcode]
        args = cls._parse_args(f, opcode, entry)
        opname = entry[0]
        return cls.Op(opcode, opname, args)

    @classmethod
    def read_io(cls, f, table=None) -> typing.Generator[Op, None, None]:
        """Read ops from a file-like object."""
        table = table or cls.tbl_dvi
        while True:
            op = cls.read_op(f, table)
            if op:
                yield op
            else:
                break

    @classmethod
    def read_file(cls, filename: str, **kwargs) -> typing.Generator[Op, None, None]:
        """Open a file and read ops from it."""
        with open(filename, "rb") as f:
            yield from cls.read_io(f, **kwargs)

    @classmethod
    def read_bytes(cls, b: bytes, **kwargs) -> typing.Generator[Op, None, None]:
        """Read ops from an in-memory byte sequence."""
        yield from cls.read_io(io.BytesIO(b), **kwargs)

    # Internals
    _parsers = {
        'delta': lambda f, delta: delta,
        'u1': lambda f, delta: _read_num(f, 1, False),
        'u2': lambda f, delta: _read_num(f, 2, False),
        'u3': lambda f, delta: _read_num(f, 3, False),
        'u4': lambda f, delta: _read_num(f, 4, False),
        's1': lambda f, delta: _read_num(f, 1, True),
        's2': lambda f, delta: _read_num(f, 2, True),
        's3': lambda f, delta: _read_num(f, 3, True),
        's4': lambda f, delta: _read_num(f, 4, True),
        'slen': lambda f, delta: _read_num(f, delta, True) if delta else None,
        'slen1': lambda f, delta: _read_num(f, delta + 1, True),
        'ulen1': lambda f, delta: _read_num(f, delta + 1, False),
        'olen1': lambda f, delta: _read_num(f, delta + 1, delta == 3),
        'fin': lambda f, delta: _read_num(f, 7, False, strict=False),
    }
    @classmethod
    def _parse_args(cls, f, opcode, entry) -> dict:
        opname, base, types, names, extra_fn = entry
        delta = opcode-base
        result = {}
        for t, n in zip(types, names):
            if t.startswith("@"):
                result[n] = f.read(result[t[1:]])
            else:
                result[n] = cls._parsers[t](f, delta)

        # Support arbitrary logic for extra params
        if extra_fn:
            extra = extra_fn(f, **result)
            result.update(extra)

        return result

    # Available dispatch tables
    tbl_dvi = DispatchTable()
    with tbl_dvi as t:
        t.op(0, 127, 'set_char', 'delta', 'c')
        t.op(128, 128, 'set_char', 'u1', 'c')
        t.op(129, 129, 'set_char', 'u2', 'c')
        t.op(130, 130, 'set_char', 'u3', 'c')
        t.op(131, 131, 'set_char', 's4', 'c')
        t.op(132, 132, 'set_rule', 's4 s4', 'height width')

        t.op(133, 133, 'put_char', 'u1', 'c')
        t.op(134, 134, 'put_char', 'u2', 'c')
        t.op(135, 135, 'put_char', 'u3', 'c')
        t.op(136, 136, 'put_char', 's4', 'c')
        t.op(137, 137, 'put_rule', 's4 s4', 'height width')

        t.op(138, 138, 'nop')
        t.op(139, 139, 'bop',
            "s4 s4 s4 s4 s4 s4 s4 s4 s4 s4 s4",
            "c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 p")
        t.op(140, 140, 'eop')

        t.op(141, 141, 'push')
        t.op(142, 142, 'pop')

        t.op(143, 146, 'right', 'slen1', 'amount')
        t.op(147, 147, 'w0')
        t.op(148, 151, 'w', 'slen1', 'new_w')
        t.op(152, 152, 'x0')
        t.op(153, 156, 'x', 'slen1', 'new_x')

        t.op(157, 160, 'down', 'slen1', 'amount')
        t.op(161, 161, 'y0')
        t.op(162, 165, 'y', 'slen1', 'new_y')
        t.op(166, 166, 'z0')
        t.op(167, 170, 'z', 'slen1', 'new_z')

        t.op(171, 234, 'fnt_num', 'delta', 'n')
        t.op(235, 238, 'fnt_num', 'slen1', 'n')

        t.op(239, 242, 'special', 'ulen1 @k', 'k text')

        t.op(243, 246, 'fnt_def',
            'olen1 u4 s4 u4 u1 u1 @a   @l',
            'k     c  s  d  a  l  area name')

        t.op(247, 247, 'pre',
            "u1 u4  u4  u4  u1 @k",
            "i  num den mag k  cmnt")
        t.op(248, 248, 'post',
            'u4 u4  u4  u4  u4 u4 u2 u2',
            'p  num den mag l  u  s  t')
        t.op(249, 249, 'post_post', 'u4 u1 fin', 'q i padding')

        t.op(250, 250, 'begin_reflect')
        t.op(251, 251, 'end_reflect')

        @t.op(252, 252, 'define_native_font',
            'u4 u4 u2    u1 @l u4',
            'k  s  flags l  n  i')
        def _extra(f, flags: int, **_) -> dict:
            read_arg = partial(_read_num, f)
            effects = {}
            if flags & 0x0200:
                effects["rgba"] = [read_arg(1, False) for _ in range(4)]
            if flags & 0x1000:
                effects["extend"] = read_arg(4, True) / 65536
            if flags & 0x2000:
                effects["slant"] = read_arg(4, True) / 65536
            if flags & 0x4000:
                effects["embolden"] = read_arg(4, True) / 65536
            return {'effects': effects}

        @t.op(253, 253, 'set_glyphs', 'u4 u2', 'w k')
        def _extra(f, w, k) -> dict:
            read_arg = partial(_read_num, f)
            xy = [read_arg(4, True) for _ in range(2 * k)]
            g = [read_arg(2, False) for _ in range(k)]
            return {'xy': xy, 'g': g}

        @t.op(254, 254, 'set_text_and_glyphs', 'u2', 'l')
        def _extra(f, l: int) -> dict:
            read_arg = partial(_read_num, f)
            t = f.read(2 * l)  # utf16
            w = read_arg(4, False)
            k = read_arg(2, False)
            xy = [read_arg(4, True) for _ in range(2 * k)]
            g = [read_arg(2, False) for _ in range(k)]
            return {'t': t, 'w': w, 'k': k, 'xy': xy, 'g': g}

        t.op(255, 255, 'malformed')

    # Operations that are valid inside a VF packet. This is a subset of DVI.
    tbl_vf_inner = DispatchTable()
    with tbl_vf_inner as t:
        t.op(0, 127, 'set_char', 'delta', 'c')
        t.op(128, 128, 'set_char', 'u1', 'c')
        t.op(129, 129, 'set_char', 'u2', 'c')
        t.op(130, 130, 'set_char', 'u3', 'c')
        t.op(131, 131, 'set_char', 's4', 'c')
        t.op(132, 132, 'set_rule', 's4 s4', 'height width')

        t.op(133, 133, 'put_char', 'u1', 'c')
        t.op(134, 134, 'put_char', 'u2', 'c')
        t.op(135, 135, 'put_char', 'u3', 'c')
        t.op(136, 136, 'put_char', 's4', 'c')
        t.op(137, 137, 'put_rule', 's4 s4', 'height width')

        t.op(139, 139, 'malformed')
        t.op(138, 138, 'nop')
        t.op(140, 140, 'malformed')

        t.op(141, 141, 'push')
        t.op(142, 142, 'pop')

        t.op(143, 146, 'right', 'slen1', 'amount')
        t.op(147, 147, 'w0')
        t.op(148, 151, 'w', 'slen1', 'new_w')
        t.op(152, 152, 'x0')
        t.op(153, 156, 'x', 'slen1', 'new_x')

        t.op(157, 160, 'down', 'slen1', 'amount')
        t.op(161, 161, 'y0')
        t.op(162, 165, 'y', 'slen1', 'new_y')
        t.op(166, 166, 'z0')
        t.op(167, 170, 'z', 'slen1', 'new_z')

        t.op(171, 234, 'fnt_num', 'delta', 'n')
        t.op(235, 238, 'fnt_num', 'slen1', 'n')

        t.op(239, 242, 'special', 'ulen1 @k', 'k text')

        t.op(243, 255, 'malformed')

    # Operations that are valid outside a VF packet.
    tbl_vf_outer = DispatchTable()
    with tbl_vf_outer as t:
        t.op(0, 241, 'char_packet',
            'delta u1 u3  @pl',
            'pl    cc tfm dvi')
        t.op(242, 242, 'char_packet',
            'u4 u4 u4  @pl',
            'pl cc tfm dvi')
        t.op(243, 246, 'fnt_def',
            'olen1 u4 s4 u4 u1 u1 @a   @l',
            'k     c  s  d  a  l  area name')
        t.op(247, 247, 'pre',
            'u1 u1 @k   u4 u4',
            'i  k  cmnt cs ds')
        t.op(248, 248, 'post', 'fin', 'padding')
        t.op(249, 255, 'malformed')

# The marks on a page consist of text and boxes. A page also has dimensions.
Page = namedtuple('Page', 'text boxes height width descent')


# Supports namedtuple interface with fields 'x y height width'
# for backwards compatibility, but is a dataclass.
@dataclasses.dataclass(slots=True, frozen=True)
class Box:
    x: int
    y: int
    height: int
    width: int

    # Format varies by backend, so we just provide it verbatim. Default is None.
    color: str | None = None

    def _as_legacy_tuple(self):
        # In the future, we should add a deprecation warning to this central location.
        # This will help us catch and clean up uses of the old API.
        return (self.x, self.y, self.height, self.width)

    def __iter__(self):
        return iter(self._as_legacy_tuple())

    def __getitem__(self, i):
        return self._as_legacy_tuple()[i]

    def replace(self, /, **kwargs):
        return dataclasses.replace(self, **kwargs)


# Supports namedtuple interface with fields 'x y font glyph width'
# for backwards compatibility, but is a dataclass.
@dataclasses.dataclass(slots=True, frozen=True)
class Text:
    """
    A glyph in the dvi file.

    In order to render the glyph, load the glyph at index ``text.index``
    from the font at ``text.font.resolve_path()`` with size ``text.font.size``,
    warped with ``text.font.effects``, then draw it at position
    ``(text.x, text.y)``.

    ``text.glyph`` is the glyph number actually stored in the dvi file (whose
    interpretation depends on the font).  ``text.width`` is the glyph width in
    dvi units.
    """
    x: int
    y: int
    font: 'DviFont'
    glyph: int
    width: int

    # Format varies by backend, so we just provide it verbatim. Default is None.
    color: str | None = None

    @property
    def index(self):
        """
        The FreeType index of this glyph (that can be passed to FT_Load_Glyph).
        """
        # See DviFont._index_dvi_to_freetype for details on the index mapping.
        return self.font._index_dvi_to_freetype(self.glyph)

    font_path = property(lambda self: self.font.resolve_path())
    font_size = property(lambda self: self.font.size)
    font_effects = property(lambda self: self.font.effects)

    def _as_legacy_tuple(self):
        # In the future, we should add a deprecation warning to this central location.
        # This will help us catch and clean up uses of the old API.
        return (self.x, self.y, self.font, self.glyph, self.width)

    def __iter__(self):
        return iter(self._as_legacy_tuple())

    def __getitem__(self, i):
        return self._as_legacy_tuple()[i]

    def replace(self, /, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @property  # To be deprecated together with font_path, font_size, font_effects.
    def glyph_name_or_index(self):
        """
        The glyph name, the native charmap glyph index, or the raw glyph index.

        If the font is a TrueType file (which can currently only happen for
        DVI files generated by xetex or luatex), then this number is the raw
        index of the glyph, which can be passed to FT_Load_Glyph/load_glyph.

        Otherwise, the font is a PostScript font.  For such fonts, if
        :file:`pdftex.map` specifies an encoding for this glyph's font,
        that is a mapping of glyph indices to Adobe glyph names; which
        is used by this property to convert dvi numbers to glyph names.
        Callers can then convert glyph names to glyph indices (with
        FT_Get_Name_Index/get_name_index), and load the glyph using
        FT_Load_Glyph/load_glyph.

        If :file:`pdftex.map` specifies no encoding for a PostScript font,
        this number is an index to the font's "native" charmap; glyphs should
        directly load using FT_Load_Char/load_char after selecting the native
        charmap.
        """
        # The last section is only true on luatex since luaotfload 3.23; this
        # must be checked by the code generated by texmanager. (luaotfload's
        # docs states "No one should rely on the mapping between DVI character
        # codes and font glyphs [prior to v3.15] unless they tightly
        # control all involved versions and are deeply familiar with the
        # implementation", but a further mapping bug was fixed in luaotfload
        # commit 8f2dca4, first included in v3.23).
        entry = self._get_pdftexmap_entry()
        return (_parse_enc(entry.encoding)[self.glyph]
                if entry.encoding is not None else self.glyph)

    def _as_unicode_or_name(self):
        if self.font.subfont:
            raise NotImplementedError("Indexing TTC fonts is not supported yet")
        path = self.font.resolve_path()
        if path.name.lower().endswith("pk"):
            # PK fonts have no encoding information; report glyphs as ASCII but
            # with a "?" to indicate that this is just a guess.
            return (f"{chr(self.glyph)}?" if chr(self.glyph).isprintable() else
                    f"pk{self.glyph:#02x}")
        face = font_manager.get_font(path)
        glyph_name = face.get_glyph_name(self.index)
        glyph_str = fontTools.agl.toUnicode(glyph_name)
        return glyph_str or glyph_name


@dataclasses.dataclass(slots=True)
class VM:
    """
    Tracks the state of a DVI document over a series of ops.
    """
    # Default fields that you usually shouldn't provide
    stack: list = dataclasses.field(default_factory=list)
    text: list = dataclasses.field(default_factory=list)
    boxes: list = dataclasses.field(default_factory=list)
    colors: list[str] = dataclasses.field(default_factory=list)
    down_stack: list = dataclasses.field(default_factory=list)
    fonts: dict = dataclasses.field(default_factory=dict)
    state: _dvistate = _dvistate.pre
    baseline_v: None = None  # TODO: type
    h: int = 0
    v: int = 0
    w: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    f: int = 0

    @property
    def color(self):
        """The current color according to color push/pop specials."""
        return self.colors[-1] if self.colors else None

    def _put_char(self, char):
        font = self.fonts[self.f]
        color = self.color
        if isinstance(font, cbook._ExceptionInfo):
            raise font.to_exception()
        elif font._vf is None:
            self.text.append(Text(self.h, self.v, font, char,
                                  font._width_of(char), color))
        else:
            scale = font._scale
            for x, y, f, g, w in font._vf[char].text:
                newf = DviFont(scale=_mul1220(scale, f._scale),
                               metrics=f._metrics, texname=f.texname, vf=f._vf)
                self.text.append(Text(self.h + _mul1220(x, scale),
                                      self.v + _mul1220(y, scale),
                                      newf, g, newf._width_of(g), color))
            self.boxes.extend([Box(self.h + _mul1220(x, scale),
                                   self.v + _mul1220(y, scale),
                                   _mul1220(a, scale), _mul1220(b, scale), color)
                               for x, y, a, b in font._vf[char].boxes])

    def _assert_state(self, opname, state):
        if self.state != state:
            raise ValueError(f"""state precondition failed:
                op {opname} must be used in state {state},
                but was used in state {self.state}""")

    def _reconsider_baseline_v(self):
        """Should be called in ops that modify self.stack or self.down_stack."""
        # Pages appear to start with the sequence
        #   bop (begin of page)
        #   xxx comment
        #   <push, ..., pop>  # if using chemformula
        #   down
        #   push
        #     down
        #     <push, push, xxx, right, xxx, pop, pop>  # if using xcolor
        #     down
        #     push
        #       down (possibly multiple)
        #       push  <=  here, v is the baseline position.
        #         etc.
        # (dviasm is useful to explore this structure.)
        # Thus, we use the vertical position at the first time the stack depth
        # reaches 3, while at least three "downs" have been executed (excluding
        # those popped out (corresponding to the chemformula preamble)), as the
        # baseline (the "down" count is necessary to handle xcolor).
        if (self.baseline_v is None
                and len(getattr(self, "stack", [])) == 3
                and self.down_stack[-1] >= 4):
            self.baseline_v = self.v

    def op_pre(self, _, i, num, den, mag, k, cmnt):
        self._assert_state("pre", _dvistate.pre)
        if i not in [2, 7]:  # 2: pdftex, luatex; 7: xetex
            raise ValueError(f"Unknown dvi format {i}")
        if num != 25400000 or den != 7227 * 2**16:
            raise ValueError("Nonstandard units in dvi file")
            # meaning: TeX always uses those exact values, so it
            # should be enough for us to support those
            # (There are 72.27 pt to an inch so 7227 pt =
            # 7227 * 2**16 sp to 100 in. The numerator is multiplied
            # by 10^5 to get units of 10**-7 meters.)
        if mag != 1000:
            raise ValueError("Nonstandard magnification in dvi file")
            # meaning: LaTeX seems to frown on setting \mag, so
            # I think we can assume this is constant
        self.state = _dvistate.outer

    def op_bop(self, _, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        self._assert_state("bop", _dvistate.outer)
        self.state = _dvistate.inpage
        self.h = self.v = self.w = self.x = self.y = self.z = 0
        self.stack = []
        self.text = []          # list of Text objects
        self.boxes = []         # list of Box objects
        self.baseline_v = None
        self.down_stack = [0]

    def op_eop(self, _):
        self._assert_state("eop", _dvistate.inpage)
        self.state = _dvistate.outer
        self.h = self.v = self.w = self.x = self.y = self.z = 0
        self.stack = []

    def op_post(self, _, **kwargs):
        self._assert_state("post", _dvistate.outer)
        self.state = _dvistate.post

    def op_post_post(self, _, **kwargs):
        self._assert_state("post_post", _dvistate.post)
        self.state = _dvistate.post_post

    def op_nop(self, _):
        pass

    def op_push(self, _):
        self.down_stack.append(self.down_stack[-1])
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))
        self._reconsider_baseline_v()

    def op_pop(self, _):
        self.down_stack.pop()
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()
        self._reconsider_baseline_v()

    def op_down(self, _, amount: int):
        self.down_stack[-1] += 1
        self.v += amount
        self._reconsider_baseline_v()

    def op_right(self, _, amount: int):
        self.h += amount

    def op_w0(self, _):
        self.h += self.w

    def op_w(self, _, new_w: int):
        self.w = new_w
        self.h += self.w

    def op_x0(self, _):
        self.h += self.x

    def op_x(self, _, new_x: int):
        self.x = new_x
        self.h += self.x

    def op_y0(self, _):
        self.v += self.y

    def op_y(self, _, new_y: int):
        self.y = new_y
        self.v += self.y

    def op_z0(self, _):
        self.v += self.z

    def op_z(self, _, new_z: int):
        self.z = new_z
        self.v += self.z

    def op_fnt_def(self, _, k, c, s, d, area: str, name: str, **kwargs):
        n = area + name
        fontname = name
        if fontname.startswith(b"[") and c == 0x4c756146:  # c == "LuaF"
            # See https://chat.stackexchange.com/rooms/106428 (and also
            # https://tug.org/pipermail/dvipdfmx/2021-January/000168.html).
            # AFAICT luatex's dvi drops info re: OpenType variation-axis values.
            self.fonts[k] = DviFont.from_luatex(s, n)
            return
        fontname = fontname.decode("ascii")
        try:
            tfm = _tfmfile(fontname)
        except FileNotFoundError as exc:
            if fontname.startswith("[") and fontname.endswith(";") and c == 0:
                exc.add_note(
                    "This dvi file was likely generated with a too-old "
                    "version of luaotfload; luaotfload 3.23 is required.")
            # Explicitly allow defining missing fonts for Vf support; we only
            # register an error when trying to load a glyph from a missing font
            # and throw that error in Dvi._read.  For Vf, _finalize_packet
            # checks whether a missing glyph has been used, and in that case
            # skips the glyph definition.
            self.fonts[k] = cbook._ExceptionInfo.from_exception(exc)
            return
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError(f'tfm checksum mismatch: {n}')
        try:
            vf = _vffile(fontname)
        except FileNotFoundError:
            vf = None
        self.fonts[k] = DviFont(scale=s, metrics=tfm, texname=n, vf=vf)

    def op_fnt_num(self, _, n: int):
        self.f = n

    def op_put_char(self, _, c):
        self._put_char(c)

    def op_set_char(self, _, c):
        self._put_char(c)
        if isinstance(self.fonts[self.f], cbook._ExceptionInfo):
            return
        self.h += self.fonts[self.f]._width_of(c)

    def op_set_rule(self, _, height, width):
        if height > 0 and width > 0:
            self.boxes.append(Box(self.h, self.v, height, width, self.color))
        self.h += width

    def op_put_rule(self, _, height, width):
        if height > 0 and width > 0:
            self.boxes.append(Box(self.h, self.v, height, width, self.color))

    def op_special(self, _, k: int, text: bytes):
        if text.startswith(b'color push'):
            color = text[len('color push'):].decode('utf-8').strip()
            self.colors.append(color)
        elif text == b'color pop':
            self.colors.pop()
        _log.debug('Dvi._xxx: encountered special: %r', text)

    def op_define_native_font(self, _, k, s, flags, l, n, i, effects):
        self.fonts[k] = DviFont.from_xetex(s, n, i, effects)

    def op_set_glyphs(self, _, w, k, xy, g):
        font = self.fonts[self.f]
        for i in range(k):
            self.text.append(Text(self.h + xy[2 * i], self.v + xy[2 * i + 1],
                                  font, g[i], font._width_of(g[i]), self.color))
        self.h += w

    def op_set_text_and_glyphs(self, _, l: int, t: bytes, w: int, k: int, xy, g):
        font = self.fonts[self.f]
        for i in range(k):
            self.text.append(Text(self.h + xy[2 * i], self.v + xy[2 * i + 1],
                                  font, g[i], font._width_of(g[i]), self.color))
        self.h += w

    def op_begin_reflect(self, _, **kwargs):
        raise NotImplementedError()

    def op_end_reflect(self, _, **kwargs):
        raise NotImplementedError()

    def op_malformed(self, _):
        raise ValueError("Malformed DVI data")


class Dvi:
    """
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
        ...     for page in dvi:
        ...         print(''.join(chr(t.glyph) for t in page.text))
    """

    def __init__(self, filename, dpi):
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
        _log.debug('Dvi: %s', filename)
        self.file = open(filename, 'rb')
        self.dpi = dpi

    def __enter__(self):
        """Context manager enter method, does nothing."""
        return self

    def __exit__(self, etype, evalue, etrace):
        """
        Context manager exit method, closes the underlying file if it is open.
        """
        self.close()

    def close(self):
        """Close the underlying file if it is open."""
        if not self.file.closed:
            self.file.close()

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Yields
        ------
        Page
            Details of all the text and box objects on the page.
            The Page tuple contains lists of Text and Box tuples and
            the page dimensions, and the Text and Box tuples contain
            coordinates transformed into a standard Cartesian
            coordinate system at the dpi value given when initializing.
            The coordinates are floating point numbers, but otherwise
            precision is not lost and coordinate values are not clipped to
            integers.
        """
        vm = VM()
        for opcode, opname, args in Ops.read_io(self.file):
            getattr(vm, f"op_{opname}")(opcode, **args)
            if opname == "eop":
                yield self._output_page(vm, self.dpi)

    def _output_page(self, vm: VM, dpi: int) -> Page:
        """Output the text and boxes belonging to the most recent page."""
        minx = miny = np.inf
        maxx = maxy = -np.inf
        maxy_pure = -np.inf
        for elt in vm.text + vm.boxes:
            if isinstance(elt, Box):
                x, y, h, w = elt
                e = 0  # zero depth
            else:  # glyph
                x, y, font, g, w = elt
                h, e = font._height_depth_of(g)
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)
        if vm.baseline_v is not None:
            maxy_pure = vm.baseline_v  # This should normally be the case.
            vm.baseline_v = None

        if not vm.text and not vm.boxes:  # Avoid infs/nans from inf+/-inf.
            return Page(text=[], boxes=[], width=0, height=0, descent=0)

        if dpi is None:
            # special case for ease of debugging: output raw dvi coordinates
            return Page(text=vm.text, boxes=vm.boxes,
                        width=maxx-minx, height=maxy_pure-miny,
                        descent=maxy-maxy_pure)

        # convert from TeX's "scaled points" to dpi units
        d = dpi / (72.27 * 2**16)
        descent = (maxy - maxy_pure) * d

        text = [
            t.replace(x=(t.x-minx)*d, y=(maxy-t.y)*d - descent, width=t.width * d)
            for t in vm.text
        ]
        boxes = [
            b.replace(
                x=(b.x-minx)*d, y=(maxy-b.y)*d - descent,
                height=b.height*d, width=b.width*d)
            for b in vm.boxes]

        return Page(text=text, boxes=boxes, width=(maxx-minx)*d,
                    height=(maxy_pure-miny)*d, descent=descent)


class DviFont:
    """
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    metrics : Tfm | TtfMetrics
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally in the DVI file, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
    fname : str
       Compatibility shim so that DviFont can be used with
       ``_backend_pdf_ps.CharacterTracker``; not a real filename.
    size : float
       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.
    """

    def __init__(self, scale, metrics, texname, vf):
        _api.check_isinstance(bytes, texname=texname)
        self._scale = scale
        self._metrics = metrics
        self.texname = texname
        self._vf = vf
        self._path = None
        self._encoding = None

    @classmethod
    def from_luatex(cls, scale, texname):
        path_b, sep, rest = texname[1:].rpartition(b"]")
        if not (texname.startswith(b"[") and sep and rest[:1] in [b"", b":"]):
            raise ValueError(f"Invalid modern font name: {texname}")
        # utf8 on Windows, not utf16!
        path = path_b.decode("utf8") if os.name == "nt" else os.fsdecode(path_b)
        subfont = 0
        effects = {}
        if rest[1:]:
            for kv in rest[1:].decode("ascii").split(";"):
                key, val = kv.split("=", 1)
                if key == "index":
                    subfont = val
                elif key in ["embolden", "slant", "extend"]:
                    effects[key] = int(val) / 65536
                else:
                    _log.warning("Ignoring invalid key-value pair: %r", kv)
        metrics = TtfMetrics(path)
        font = cls(scale, metrics, texname, vf=None)
        font._path = Path(path)
        font.subfont = subfont
        font.effects = effects
        return font

    @classmethod
    def from_xetex(cls, scale, texname, subfont, effects):
        # utf8 on Windows, not utf16!
        path = texname.decode("utf8") if os.name == "nt" else os.fsdecode(texname)
        metrics = TtfMetrics(path)
        font = cls(scale, metrics, b"[" + texname + b"]", vf=None)
        font._path = Path(path)
        font.subfont = subfont
        font.effects = effects
        return font

    size = property(lambda self: self._scale * (72.0 / (72.27 * 2**16)))

    widths = _api.deprecated("3.11")(property(lambda self: [
        (1000 * self._tfm.width.get(char, 0)) >> 20
        for char in range(max(self._tfm.width, default=-1) + 1)]))

    @property
    def fname(self):
        """A fake filename"""
        return self.texname.decode('latin-1')

    def _get_fontmap(self, string):
        """Get the mapping from characters to the font that includes them.

        Each value maps to self; there is no fallback mechanism for DviFont.
        """
        return {char: self for char in string}

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.texname == other.texname and self.size == other.size)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"<{type(self).__name__}: {self.texname}>"

    def _width_of(self, char):
        """Width of char in dvi units."""
        metrics = self._metrics.get_metrics(char)
        if metrics is None:
            _log.debug('No width for char %d in font %s.', char, self.texname)
            return 0
        return _mul1220(metrics.tex_width, self._scale)

    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""
        metrics = self._metrics.get_metrics(char)
        if metrics is None:
            _log.debug('No metrics for char %d in font %s', char, self.texname)
            return [0, 0]
        hd = [
            _mul1220(metrics.tex_height, self._scale),
            _mul1220(metrics.tex_depth, self._scale),
        ]
        # cmsyXX (symbols font) glyph 0 ("minus") has a nonzero descent
        # so that TeX aligns equations properly
        # (https://tex.stackexchange.com/q/526103/)
        # but we actually care about the rasterization depth to align
        # the dvipng-generated images.
        if re.match(br'^cmsy\d+$', self.texname) and char == 0:
            hd[-1] = 0
        return hd

    def resolve_path(self):
        if self._path is None:
            fontmap = PsfontsMap(find_tex_file("pdftex.map"))
            try:
                psfont = fontmap[self.texname]
            except LookupError as exc:
                try:
                    find_tex_file(f"{self.texname.decode('ascii')}.mf")
                except FileNotFoundError:
                    raise exc from None
                else:
                    self._path = Path(find_tex_file(
                        f"{self.texname.decode('ascii')}.600pk"))
            else:
                if psfont.filename is None:
                    raise ValueError("No usable font file found for {} ({}); "
                                     "the font may lack a Type-1 version"
                                     .format(psfont.psname.decode("ascii"),
                                             psfont.texname.decode("ascii")))
                self._path = Path(psfont.filename)
        return self._path

    @cached_property
    def subfont(self):
        return 0

    @cached_property
    def effects(self):
        if self.resolve_path().match("*.600pk"):
            return {}
        return PsfontsMap(find_tex_file("pdftex.map"))[self.texname].effects

    def _index_dvi_to_freetype(self, idx):
        """Convert dvi glyph indices to FreeType ones."""
        # Glyphs indices stored in the dvi file map to FreeType glyph indices
        # (i.e., which can be passed to FT_Load_Glyph) in various ways:
        # - for xetex & luatex "native fonts", dvi indices are directly equal
        #   to FreeType indices.
        # - if pdftex.map specifies an ".enc" file for the font, that file maps
        #   dvi indices to Adobe glyph names, which can then be converted to
        #   FreeType glyph indices with FT_Get_Name_Index.
        # - if no ".enc" file is specified, then the font must be a Type 1
        #   font, and dvi indices directly index into the font's CharStrings
        #   vector.
        if self.texname.startswith(b"["):
            return idx
        if self._encoding is None:
            face = font_manager.get_font(self.resolve_path())
            psfont = PsfontsMap(find_tex_file("pdftex.map"))[self.texname]
            if psfont.encoding:
                self._encoding = [face.get_name_index(name)
                                  for name in _parse_enc(psfont.encoding)]
            else:
                self._encoding = face._get_type1_encoding_vector()
        return self._encoding[idx]


class Vf:
    r"""
    A virtual font (\*.vf file) containing subroutines for dvi files.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    The virtual font format is a derivative of dvi:
    http://mirrors.ctan.org/info/knuth/virtual-fonts
    This class reuses some of the machinery of `Dvi`
    but replaces the `!_read` loop and dispatch mechanism.

    The format is:
     - `pre` op (247)
     - font definitions (243-246)
     - character packets (0-242)
     - postamble (248)

    Each character packet declares its payload length, and the payload is made
    of (a subset of) the normal DVI ops. This is exposed as a Page object
    and represents a single glyph, which is accessible via __getitem__.

    Examples
    --------
    ::

        vf = Vf(filename)
        glyph = vf[code]
        glyph.text, glyph.boxes, glyph.width
    """

    def __init__(self, filename):
        self._chars = {}

        self.inner_vm = VM(state=_dvistate.outer)
        self.state = _dvistate.pre
        for op in Ops.read_file(filename, table=Ops.tbl_vf_outer):
            opcode, opname, args = op
            getattr(self, f"op_{opname}")(opcode, **args)
        del self.inner_vm
        del self.state

    def __getitem__(self, code):
        return self._chars[code]

    def op_pre(self, _, i, k, cmnt, cs, ds):
        if self.state is not _dvistate.pre:
            raise ValueError("pre command in middle of vf file")
        if i != 202:
            raise ValueError(f"Unknown vf format {i}")
        if len(cmnt):
            _log.debug('vf file comment: %s', cmnt)
        self.state = _dvistate.outer
        # cs = checksum, ds = design size

    def op_fnt_def(self, code: int, **kwargs):
        if self.state is not _dvistate.outer:
            raise ValueError(f"fnt_def command cannot be used in state {self.state}")
        self.inner_vm.op_fnt_def(code, **kwargs)

    def op_char_packet(self, _, pl: int, cc: int, tfm: int, dvi: bytes):
        if self.state is not _dvistate.outer:
            raise ValueError(
                f"char_packet command cannot be used in state {self.state}")
        vm = self.inner_vm

        # Just feed these right on in to the inner VM, wrapping as a page
        vm.op_bop(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for op in Ops.read_bytes(dvi):
            opcode, opname, args = op
            getattr(vm, f"op_{opname}")(opcode, **args)
        vm.op_eop(0)

        # Create a Page object from that, and store it in self._chars.
        # Note, some prior logic was explicitly lenient about missing fonts here.
        # It's unclear if this still needs to be explicitly handled. Tests welcome!
        self._chars[cc] = Page(
            text=vm.text, boxes=vm.boxes, width=tfm,
            height=None, descent=None)

    def op_post(self, _, **kwargs):
        pass


def _mul1220(num1, num2):
    """Multiply two numbers in 12.20 fixed point format."""
    # Separated into a function because >> has surprising precedence
    return (num1*num2) >> 20


@dataclasses.dataclass(frozen=True, kw_only=True)
class TexMetrics:
    """
    Metrics of a glyph, with TeX semantics.

    TeX metrics have different semantics from FreeType metrics: tex_width
    corresponds to FreeType's ``advance`` (i.e., including whitespace padding);
    tex_height to ``bearingY`` (how much the glyph extends over the baseline);
    tex_depth to ``height - bearingY`` (how much the glyph extends under the
    baseline, as a positive number).
    """
    tex_width: int
    tex_height: int
    tex_depth: int


class Tfm:
    """
    A TeX Font Metric file.

    This implementation covers only the bare minimum needed by the Dvi class.

    Parameters
    ----------
    filename : str or path-like

    Attributes
    ----------
    checksum : int
       Used for verifying against the dvi file.
    design_size : int
       Design size of the font (in 12.20 TeX points); unused because it is
       overridden by the scale factor specified in the dvi file.
    """

    def __init__(self, filename):
        _log.debug('opening tfm file %s', filename)
        with open(filename, 'rb') as file:
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = struct.unpack('!6H', header1[2:14])
            _log.debug('lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d',
                       lh, bc, ec, nw, nh, nd)
            header2 = file.read(4*lh)
            self.checksum, self.design_size = struct.unpack('!2I', header2[:8])
            # there is also encoding information etc.
            char_info = file.read(4*(ec-bc+1))
            widths = struct.unpack(f'!{nw}i', file.read(4*nw))
            heights = struct.unpack(f'!{nh}i', file.read(4*nh))
            depths = struct.unpack(f'!{nd}i', file.read(4*nd))
        self._glyph_metrics = {}
        for idx, char in enumerate(range(bc, ec+1)):
            byte0 = char_info[4*idx]
            byte1 = char_info[4*idx+1]
            self._glyph_metrics[char] = TexMetrics(
                tex_width=widths[byte0],
                tex_height=heights[byte1 >> 4],
                tex_depth=depths[byte1 & 0xf],
            )

    def get_metrics(self, idx):
        """Return a glyph's TexMetrics, or None if unavailable."""
        return self._glyph_metrics.get(idx)

    width = _api.deprecated("3.11", alternative="get_metrics")(
        property(lambda self: {c: m.tex_width for c, m in self._glyph_metrics}))
    height = _api.deprecated("3.11", alternative="get_metrics")(
        property(lambda self: {c: m.tex_height for c, m in self._glyph_metrics}))
    depth = _api.deprecated("3.11", alternative="get_metrics")(
        property(lambda self: {c: m.tex_depth for c, m in self._glyph_metrics}))


class TtfMetrics:
    def __init__(self, filename):
        self._face = font_manager.get_font(filename, hinting_factor=1)

    def get_metrics(self, idx):
        # _mul1220 uses a truncating bitshift for compatibility with dvitype.
        # When upem is 2048 the conversion to 12.20 is exact, but when
        # upem is 1000 (e.g. lmroman10-regular.otf) the metrics themselves
        # are not exactly representable as 12.20 fp.  Manual testing via
        # \sbox0{x}\count0=\wd0\typeout{\the\count0} suggests that metrics
        # are rounded (not truncated) after conversion to 12.20 and before
        # multiplication by the scale.
        upem = self._face.units_per_EM  # Usually 2048 or 1000.
        g = self._face.load_glyph(idx, LoadFlags.NO_SCALE)
        return TexMetrics(
            tex_width=round(g.horiAdvance / upem * 2**20),
            tex_height=round(g.horiBearingY / upem * 2**20),
            tex_depth=round((g.height - g.horiBearingY) / upem * 2**20),
        )


PsFont = namedtuple('PsFont', 'texname psname effects encoding filename')


class PsfontsMap:
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX's native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
    the 35 PostScript fonts (i.e., have no filename for them, as in
    the Times-Bold example above), while the pdf-related files perhaps
    only avoid the "Base 14" pdf fonts. But the user may have
    configured these files differently.

    Examples
    --------
    >>> map = PsfontsMap(find_tex_file('pdftex.map'))
    >>> entry = map[b'ptmbo8r']
    >>> entry.texname
    b'ptmbo8r'
    >>> entry.psname
    b'Times-Bold'
    >>> entry.encoding
    '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
    >>> entry.effects
    {'slant': 0.16700000000000001}
    >>> entry.filename
    """
    __slots__ = ('_filename', '_unparsed', '_parsed')

    # Create a filename -> PsfontsMap cache, so that calling
    # `PsfontsMap(filename)` with the same filename a second time immediately
    # returns the same object.
    @lru_cache
    def __new__(cls, filename):
        self = object.__new__(cls)
        self._filename = os.fsdecode(filename)
        # Some TeX distributions have enormous pdftex.map files which would
        # take hundreds of milliseconds to parse, but it is easy enough to just
        # store the unparsed lines (keyed by the first word, which is the
        # texname) and parse them on-demand.
        with open(filename, 'rb') as file:
            self._unparsed = {}
            for line in file:
                tfmname = line.split(b' ', 1)[0]
                self._unparsed.setdefault(tfmname, []).append(line)
        self._parsed = {}
        return self

    def __getitem__(self, texname):
        assert isinstance(texname, bytes)
        if texname in self._unparsed:
            for line in self._unparsed.pop(texname):
                if self._parse_and_cache_line(line):
                    break
        try:
            return self._parsed[texname]
        except KeyError:
            raise LookupError(
                f"The font map {self._filename!r} is missing a PostScript font "
                f"associated to TeX font {texname.decode('ascii')!r}; this problem can "
                f"often be solved by installing a suitable PostScript font package in "
                f"your TeX package manager") from None

    def _parse_and_cache_line(self, line):
        """
        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        """
        # If the map file specifies multiple encodings for a font, we
        # follow pdfTeX in choosing the last one specified. Such
        # entries are probably mistakes but they have occurred.
        # https://tex.stackexchange.com/q/10826/

        if not line or line.startswith((b" ", b"%", b"*", b";", b"#")):
            return
        tfmname = basename = special = encodingfile = fontfile = None
        is_subsetted = is_t1 = is_truetype = False
        matches = re.finditer(br'"([^"]*)(?:"|$)|(\S+)', line)
        for match in matches:
            quoted, unquoted = match.groups()
            if unquoted:
                if unquoted.startswith(b"<<"):  # font
                    fontfile = unquoted[2:]
                elif unquoted.startswith(b"<["):  # encoding
                    encodingfile = unquoted[2:]
                elif unquoted.startswith(b"<"):  # font or encoding
                    word = (
                        # <foo => foo
                        unquoted[1:]
                        # < by itself => read the next word
                        or next(filter(None, next(matches).groups())))
                    if word.endswith(b".enc"):
                        encodingfile = word
                    else:
                        fontfile = word
                        is_subsetted = True
                elif tfmname is None:
                    tfmname = unquoted
                elif basename is None:
                    basename = unquoted
            elif quoted:
                special = quoted
        effects = {}
        if special:
            words = reversed(special.split())
            for word in words:
                if word == b"SlantFont":
                    effects["slant"] = float(next(words))
                elif word == b"ExtendFont":
                    effects["extend"] = float(next(words))

        # Verify some properties of the line that would cause it to be ignored
        # otherwise.
        if fontfile is not None:
            if fontfile.endswith((b".ttf", b".ttc")):
                is_truetype = True
            elif not fontfile.endswith(b".otf"):
                is_t1 = True
        elif basename is not None:
            is_t1 = True
        if is_truetype and is_subsetted and encodingfile is None:
            return
        if not is_t1 and ("slant" in effects or "extend" in effects):
            return
        if abs(effects.get("slant", 0)) > 1:
            return
        if abs(effects.get("extend", 0)) > 2:
            return

        if basename is None:
            basename = tfmname
        if encodingfile is not None:
            encodingfile = find_tex_file(encodingfile)
        if fontfile is not None:
            fontfile = find_tex_file(fontfile)
        self._parsed[tfmname] = PsFont(
            texname=tfmname, psname=basename, effects=effects,
            encoding=encodingfile, filename=fontfile)
        return True


def _parse_enc(path):
    r"""
    Parse a \*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : `os.PathLike`

    Returns
    -------
    list
        The nth list item is the PostScript glyph name of the nth glyph.
    """
    no_comments = re.sub("%.*", "", Path(path).read_text(encoding="ascii"))
    array = re.search(r"(?s)\[(.*)\]", no_comments).group(1)
    lines = [line for line in array.split() if line]
    if all(line.startswith("/") for line in lines):
        return [line[1:] for line in lines]
    else:
        raise ValueError(f"Failed to parse {path} as Postscript encoding")


class _LuatexKpsewhich:
    @cache  # A singleton.
    def __new__(cls):
        self = object.__new__(cls)
        self._proc = self._new_proc()
        return self

    def _new_proc(self):
        return subprocess.Popen(
            ["luatex", "--luaonly", str(cbook._get_data_path("kpsewhich.lua"))],
            # mktexpk logs to stderr; suppress that.
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            # Store generated pk fonts in our own cache.
            env={"MT_VARTEXFONTS": str(Path(mpl.get_cachedir(), "vartexfonts")),
                 **os.environ})

    def search(self, filename):
        if self._proc.poll() is not None:  # Dead, restart it.
            self._proc = self._new_proc()
        self._proc.stdin.write(os.fsencode(filename) + b"\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline().rstrip()
        return None if out == b"nil" else os.fsdecode(out)


@lru_cache
def find_tex_file(filename):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """

    # we expect these to always be ascii encoded, but use utf-8
    # out of caution
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='replace')

    try:
        lk = _LuatexKpsewhich()
    except FileNotFoundError:
        lk = None  # Fallback to directly calling kpsewhich, as below.

    if lk:
        path = lk.search(filename)
    else:
        if sys.platform == 'win32':
            # On Windows only, kpathsea can use utf-8 for cmd args and output.
            # The `command_line_encoding` environment variable is set to force
            # it to always use utf-8 encoding.  See Matplotlib issue #11848.
            kwargs = {'env': {**os.environ, 'command_line_encoding': 'utf-8'},
                      'encoding': 'utf-8'}
        else:  # On POSIX, run through the equivalent of os.fsdecode().
            kwargs = {'env': {**os.environ},
                      'encoding': sys.getfilesystemencoding(),
                      'errors': 'surrogateescape'}
        kwargs['env'].update(
            MT_VARTEXFONTS=str(Path(mpl.get_cachedir(), "vartexfonts")))

        try:
            path = cbook._check_and_log_subprocess(
                ['kpsewhich', '-mktex=pk', filename], _log, **kwargs,
            ).rstrip('\n')
        except (FileNotFoundError, RuntimeError):
            path = None

    if path:
        return path
    else:
        raise FileNotFoundError(
            f"Matplotlib's TeX implementation searched for a file named "
            f"{filename!r} in your texmf tree, but could not find it")


@lru_cache
def _fontfile(cls, suffix, texname):
    return cls(find_tex_file(texname + suffix))


_tfmfile = partial(_fontfile, Tfm, ".tfm")
_vffile = partial(_fontfile, Vf, ".vf")


if __name__ == '__main__':
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("dpi", nargs="?", type=float, default=None)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    def _print_fields(*args):
        print(" ".join(map("{:>11}".format, args)))

    with Dvi(args.filename, args.dpi) as dvi:
        for page in dvi:
            print(f"=== NEW PAGE === "
                  f"(w: {page.width}, h: {page.height}, d: {page.descent})")
            print("--- GLYPHS ---")
            for font, group in itertools.groupby(page.text, lambda text: text.font):
                font_name = (font.texname.decode("utf8") if os.name == "nt"
                             else os.fsdecode(font.texname))
                if isinstance(font._metrics, Tfm):
                    print(f"font: {font_name} at {font.resolve_path()}")
                else:
                    print(f"font: {font_name}")
                print(f"scale: {font._scale / 2 ** 20}")
                _print_fields("x", "y", "glyph", "chr", "w", "color")
                for text in group:
                    _print_fields(text.x, text.y, text.glyph,
                                  text._as_unicode_or_name(), text.width,
                                  text.color or "(default)")
            if page.boxes:
                print("--- BOXES ---")
                _print_fields("x", "y", "h", "w", "color")
                for box in page.boxes:
                    _print_fields(box.x, box.y, box.height, box.width, box.color)
