Complex text layout with libraqm
--------------------------------

Text support has been extended to include complex text layout. This support includes:

1. Languages that require advanced layout, such as Arabic or Hebrew.
2. Text that mixes left-to-right and right-to-left languages.

   .. plot::
       :show-source-link: False

       text = 'Here is some رَقْم in اَلْعَرَبِيَّةُ'
       fig = plt.figure(figsize=(6, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center')

3. Ligatures that combine several adjacent characters for improved legibility.

   .. plot::
       :show-source-link: False

       text = 'f\N{Hair Space}f\N{Hair Space}i \N{Rightwards Arrow} ffi'
       fig = plt.figure(figsize=(3, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center')

4. Combining multiple or double-width diacritics.

   .. plot::
       :show-source-link: False

       text = (
           'a\N{Combining Circumflex Accent}\N{Combining Double Tilde}'
           'c\N{Combining Diaeresis}')
       text = ' + '.join(
           c if c in 'ac' else f'\N{Dotted Circle}{c}'
           for c in text) + f' \N{Rightwards Arrow} {text}'
       fig = plt.figure(figsize=(6, 1))
       fig.text(0.5, 0.5, text, size=32, ha='center', va='center',
                # Builtin DejaVu Sans doesn't support multiple diacritics.
                family=['Noto Sans', 'DejaVu Sans'])

Note, all advanced features require corresponding font support, and may require
additional fonts over the builtin DejaVu Sans.
