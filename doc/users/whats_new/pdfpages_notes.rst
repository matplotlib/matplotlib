Per-page pdf notes in multi-page pdfs (PdfPages)
------------------------------------------------

Add a new method attach_note to the PdfPages class, allowing the
attachment of simple text notes to pages in a multi-page pdf of
figures. The new note is visible in the list of pdf annotations in a
viewer that has this facility (Adobe Reader, OSX Preview, Skim,
etc.). Per default the note itself is kept off-page to prevent it to
appear in print-outs.

`PdfPages.attach_note` needs to be called before savefig in order to be
added to the correct figure.
