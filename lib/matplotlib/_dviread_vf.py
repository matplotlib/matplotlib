from functools import partial
import logging

from matplotlib.dviread import Dvi
from matplotlib._dviread import _dvistate, _fontfile, Page


_log = logging.getLogger(__name__)


class Vf(Dvi):
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
    but replaces the `_read` loop and dispatch mechanism.

    Examples
    --------
    ::

        vf = Vf(filename)
        glyph = vf[code]
        glyph.text, glyph.boxes, glyph.width
    """

    def __init__(self, filename):
        super().__init__(filename, 0)
        try:
            self._first_font = None
            self._chars = {}
            self._read()
        finally:
            self.close()

    def __getitem__(self, code):
        return self._chars[code]

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        packet_char, packet_ends = None, None
        packet_len, packet_width = None, None
        while True:
            byte = self.file.read(1)[0]
            # If we are in a packet, execute the dvi instructions
            if self.state is _dvistate.inpage:
                byte_at = self.file.tell()-1
                if byte_at == packet_ends:
                    self._finalize_packet(packet_char, packet_width)
                    packet_len, packet_char, packet_width = None, None, None
                    # fall through to out-of-packet code
                elif byte_at > packet_ends:
                    raise ValueError("Packet length mismatch in vf file")
                else:
                    if byte in (139, 140) or byte >= 243:
                        raise ValueError(
                            "Inappropriate opcode %d in vf file" % byte)
                    Dvi._dtable[byte](self, byte)
                    continue

            # We are outside a packet
            if byte < 242:          # a short packet (length given by byte)
                packet_len = byte
                packet_char, packet_width = self._arg(1), self._arg(3)
                packet_ends = self._init_packet(byte)
                self.state = _dvistate.inpage
            elif byte == 242:       # a long packet
                packet_len, packet_char, packet_width = \
                            [self._arg(x) for x in (4, 4, 4)]
                self._init_packet(packet_len)
            elif 243 <= byte <= 246:
                k = self._arg(byte - 242, byte == 246)
                c, s, d, a, l = [self._arg(x) for x in (4, 4, 4, 1, 1)]
                self._fnt_def_real(k, c, s, d, a, l)
                if self._first_font is None:
                    self._first_font = k
            elif byte == 247:       # preamble
                i, k = self._arg(1), self._arg(1)
                x = self.file.read(k)
                cs, ds = self._arg(4), self._arg(4)
                self._pre(i, x, cs, ds)
            elif byte == 248:       # postamble (just some number of 248s)
                break
            else:
                raise ValueError("Unknown vf opcode %d" % byte)

    def _init_packet(self, pl):
        if self.state != _dvistate.outer:
            raise ValueError("Misplaced packet in vf file")
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack, self.text, self.boxes = [], [], []
        self.f = self._first_font
        return self.file.tell() + pl

    def _finalize_packet(self, packet_char, packet_width):
        self._chars[packet_char] = Page(
            text=self.text, boxes=self.boxes, width=packet_width,
            height=None, descent=None)
        self.state = _dvistate.outer

    def _pre(self, i, x, cs, ds):
        if self.state is not _dvistate.pre:
            raise ValueError("pre command in middle of vf file")
        if i != 202:
            raise ValueError("Unknown vf format %d" % i)
        if len(x):
            _log.debug('vf file comment: %s', x)
        self.state = _dvistate.outer
        # cs = checksum, ds = design size


_vffile = partial(_fontfile, Vf, ".vf")
