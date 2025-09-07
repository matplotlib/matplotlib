Missing glyphs use Last Resort font
-----------------------------------

Most fonts do not have 100% character coverage, and will fall back to a "not found"
glyph for characters that are not provided. Often, this glyph will be minimal (e.g., the
default DejaVu Sans "not found" glyph is just a rectangle.) Such minimal glyphs provide
no context as to the characters that are missing.

Now, missing glyphs will fall back to the `Last Resort font
<https://github.com/unicode-org/last-resort-font>`__ produced by the Unicode Consortium.
This special-purpose font provides glyphs that represent types of Unicode characters.
These glyphs show a representative character from the missing Unicode block, and at
larger sizes, more context to help determine which character and font are needed.

To disable this fallback behaviour, set :rc:`font.enable_last_resort` to ``False``.

.. plot::
    :alt: An example of missing glyph behaviour, the first glyph from Bengali script,
        second glyph from Hiragana, and the last glyph from the Unicode Private Use
        Area. Multiple lines repeat the text with increasing font size from top to
        bottom.

    text_raw = r"'\N{Bengali Digit Zero}\N{Hiragana Letter A}\ufdd0'"
    text = eval(text_raw)
    sizes = [
        (0.85, 8),
        (0.80, 10),
        (0.75, 12),
        (0.70, 16),
        (0.63, 20),
        (0.55, 24),
        (0.45, 32),
        (0.30, 48),
        (0.10, 64),
    ]

    fig = plt.figure()
    fig.text(0.01, 0.90, f'Input: {text_raw}')
    for y, size in sizes:
        fig.text(0.01, y, f'{size}pt:{text}', fontsize=size)
