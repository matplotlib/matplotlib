`EngFormatter` now accepts `usetex`, `useMathText` as keyword only arguments
``````````````````````````````````````````````````````````````````````````````````````````````

A public API has been added to `EngFormatter` to control how the numbers in the ticklabels will be rendered.
By default, `useMathText` evaluates to `rcParams['axes.formatter.use_mathtext']` and
`usetex` evaluates to `rcParams['text.usetex']`.

If either is `True` then  the numbers will be encapsulated by `$` signs. When using `TeX` this implies
that the numbers will be shown in TeX's math font. When using mathtext, the `$` signs around numbers will
ensure unicode rendering (as implied by mathtext). This will make sure that the minus signs in the ticks
are rendered as the unicode-minus (U+2212) when using mathtext (without relying on the `fix_minus` method).
