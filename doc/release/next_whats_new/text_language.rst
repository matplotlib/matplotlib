Specifying text language
------------------------

OpenType fonts may support language systems which can be used to select different
typographic conventions, e.g., localized variants of letters that share a single Unicode
code point, or different default font features. The text API now supports setting a
language to be used and may be set/get with:

- `matplotlib.text.Text.set_language` / `matplotlib.text.Text.get_language`
- Any API that creates a `.Text` object by passing the *language* argument (e.g.,
  ``plt.xlabel(..., language=...)``)

The language of the text must be in a format accepted by libraqm, namely `a BCP47
language code <https://www.w3.org/International/articles/language-tags/>`_. If None or
unset, then no particular language will be implied, and default font settings will be
used.

For example, Matplotlib's default font ``DejaVu Sans`` supports language-specific glyphs
in the Serbian and Macedonian languages in the Cyrillic alphabet, or the Sámi family of
languages in the Latin alphabet.

.. plot::
    :include-source:

    fig = plt.figure(figsize=(7, 3))

    char = '\U00000431'
    fig.text(0.5, 0.8, f'\\U{ord(char):08x}', fontsize=40, horizontalalignment='center')
    fig.text(0, 0.6, f'Serbian: {char}', fontsize=40, language='sr')
    fig.text(1, 0.6, f'Russian: {char}', fontsize=40, language='ru',
             horizontalalignment='right')

    char = '\U0000014a'
    fig.text(0.5, 0.3, f'\\U{ord(char):08x}', fontsize=40, horizontalalignment='center')
    fig.text(0, 0.1, f'English: {char}', fontsize=40, language='en')
    fig.text(1, 0.1, f'Inari Sámi: {char}', fontsize=40, language='smn',
             horizontalalignment='right')
