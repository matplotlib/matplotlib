.. _outline:

************
Docs outline
************

Proposed chapters for the docs, who has responsibility for them, and
who reviews them.  The "unit" doesn't have to be a full chapter
(though in some cases it will be), it may be a chapter or a section in
a chapter.

===============================  ==================== ===========  ===================
User's guide unit                Author               Status       Reviewer
===============================  ==================== ===========  ===================
plotting 2-D arrays              Eric                 has author   Perry ? Darren
colormapping                     Eric                 has author   ?
quiver plots                     Eric                 has author   ?
histograms                       Manuel ?             no author    Erik Tollerud ?
bar / errorbar                   ?                    no author    ?
x-y plots                        ?                    no author    Darren
time series plots                ?                    no author    ?
date plots                       John                 has author   ?
working with data                John                 has author   Darren
custom ticking                   ?                    no author    ?
masked data                      Eric                 has author   ?
patches                          ?                    no author    ?
legends                          ?                    no author    ?
animation                        John                 has author   ?
collections                      ?                    no author    ?
text - mathtext                  Michael              in review    John
text - usetex                    Darren               submitted    ?
text - annotations               John                 submitted    ?
fonts et al                      Michael ?            no author    Darren
pyplot tut                       John                 submitted    Eric
configuration                    Darren               preliminary  ?
win32 install                    Charlie ?            no author    Darren
os x install                     Charlie ?            no author    ?
linux install                    Darren               has author   ?
artist api                       John                 submitted    ?
event handling                   John                 submitted    ?
navigation                       John                 submitted    ?
interactive usage                ?                    no author    ?
widgets                          ?                    no author    ?
ui - gtk                         ?                    no author    ?
ui - wx                          ?                    no author    ?
ui - tk                          ?                    no author    ?
ui - qt                          Darren               has author   ?
backend - pdf                    Jouni ?              no author    ?
backend - ps                     Darren               has author   ?
backend - svg                    ?                    no author    ?
backend - agg                    ?                    no author    ?
backend - cairo                  ?                    no author    ?
===============================  ==================== ===========  ===================

Here is the ouline for the dev guide, much less fleshed out

===============================  ==================== ===========  ===================
Developer's guide unit           Author               Status       Reviewer
===============================  ==================== ===========  ===================
the renderer                     John                 has author   Michael ?
the canvas                       John                 has author   ?
the artist                       John                 has author   ?
transforms                       Michael              submitted    John
documenting mpl                  Darren               submitted    ?
coding guide                     John                 submitted    Eric
and_much_more                    ?                    ?            ?
===============================  ==================== ===========  ===================

And we might want to do a similar table for the FAQ, but that may also be overkill...

If you agree to author a unit, remove the question mark by your name
(or add your name if there is no candidate), and change the status to
"has author".  Once you have completed draft and checked it in, you
can change the status to "submitted" and try to find a reviewer if you
don't have one.  The reviewer should read your chapter, test it for
correctness (eg try your examples) and change the status to "complete"
when done.

You are free to lift and convert as much material from the web site or
the existing latex user's guide as you see fit.  The more the better.

The UI chapters should give an example or two of using mpl with your
GUI and any relevant info, such as version, installation, config,
etc...  The backend chapters should cover backend specific
configuration (eg PS only options), what features are missing, etc...

Please feel free to add units, volunteer to review or author a
chapter, etc...

It is probably easiest to be an editor. Once you have signed up to be
an editor, if you have an author pester the author for a submission
every so often. If you don't have an author, find one, and then pester
them!  Your only two responsibilities are getting your author to
produce and checking their work, so don't be shy.  You *do not* need
to be an expert in the subject you are editing -- you should know
something about it and be willing to read, test, give feedback and
pester!

Reviewer notes
==============

If you want to make notes for the authorwhen you have reviewed a
submission, you can put them here.  As the author cleans them up or
addresses them, they should be removed.

mathtext user's guide (reviewd by JDH)
--------------------------------------

This looks good -- there are a few minor things to close the book on
this chapter.

#. The main thing to wrap this up is getting the mathtext module
ported over to rest and included in the API so the links from the
user's guide tutorial work.

#. This section might also benefit from a little more detail on the
customizations that are possible (eg an example fleshing out the rc
options a little bit).  Admittedly, this is pretty clear from readin
ghte rc file, but it might be helpful to a newbie.

#. There is still a TODO in the file to include a complete list of symbols

