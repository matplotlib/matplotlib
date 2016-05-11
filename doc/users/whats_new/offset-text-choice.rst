Improved offset text choice
---------------------------
The default offset-text choice was changed to only use significant digits that
are common to all ticks (e.g. 1231..1239 -> 1230, instead of 1231), except when
they straddle a relatively large multiple of a power of ten, in which case that
multiple is chosen (e.g. 1999..2001->2000).
