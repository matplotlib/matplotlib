Looping GIFs with PillowWriter
------------------------------

We acknowledge that most people want to watch a gif more than once.  Saving an animatation
as a gif with PillowWriter now defaults to producing an endless looping gif. Gifs can be set
to loop a finite number of times by passing the ``loop`` keyword argument to PillowWriter. 
To restore the old behaviour of not looping use ``PillowWriter(loop=1)``
