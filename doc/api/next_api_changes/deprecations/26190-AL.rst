Artists explicitly passed in will no longer be filtered by legend() based on their label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, artists explicitly passed to ``legend(handles=[...])`` are filtered
out if their label starts with an underscore.  This behavior is deprecated;
explicitly filter out such artists
(``[art for art in artists if not art.get_label().startswith('_')]``) if
necessary.
