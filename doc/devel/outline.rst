.. _outline:

************
Docs outline
************

Proposed chapters for the docs, who has responsibility for them, and
who reviews them.  The "unit" doesn't have to be a full chapter
(though in some cases it will be), it may be a chapter or a section in
a chapter.

========================   ==================   ==========   ===================
User's guide unit          Author               Status       Reviewer
========================   ==================   ==========   ===================
plotting 2-D arrays        Eric                 has author   Perry ? Darren
colormapping               Eric                 has author   ?
quiver plots               Eric                 has author   ?
histograms                 Manuel ?             no author    Erik Tollerud ?
bar / errorbar             ?                    no author    ?
x-y plots                  ?                    no author    Darren
time series plots          ?                    no author    ?
date plots                 John                 has author   ?
working with data          John                 has author   Darren
custom ticking             ?                    no author    ?
masked data                Eric                 has author   ?
patches                    ?                    no author    ?
legends                    ?                    no author    ?
animation                  John                 has author   ?
collections                ?                    no author    ?
text - mathtext            Michael              accepted     John
text - usetex              Darren               accepted     John
text - annotations         John                 submitted    ?
fonts et al                Michael ?            no author    Darren
pyplot tut                 John                 submitted    Eric
configuration              Darren               submitted    ?
win32 install              Charlie ?            no author    Darren
os x install               Charlie ?            no author    ?
linux install              Darren               has author   ?
artist api                 John                 submitted    ?
event handling             John                 submitted    ?
navigation                 John                 submitted    ?
interactive usage          ?                    no author    ?
widgets                    ?                    no author    ?
ui - gtk                   ?                    no author    ?
ui - wx                    ?                    no author    ?
ui - tk                    ?                    no author    ?
ui - qt                    Darren               has author   ?
backend - pdf              Jouni ?              no author    ?
backend - ps               Darren               has author   ?
backend - svg              ?                    no author    ?
backend - agg              ?                    no author    ?
backend - cairo            ?                    no author    ?
========================   ==================   ==========   ===================

Here is the ouline for the dev guide, much less fleshed out

==========================   ===============   ===========   ==================
Developer's guide unit       Author            Status        Reviewer
==========================   ===============   ===========   ==================
the renderer                 John              has author    Michael ?
the canvas                   John              has author    ?
the artist                   John              has author    ?
transforms                   Michael           submitted     John
documenting mpl              Darren            submitted     John, Eric, Mike?
coding guide                 John              complete      Eric
and_much_more                ?                 ?             ?
==========================   ===============   ===========   ==================

We also have some work to do converting docstrings to ReST for the API
Reference. Please be sure to follow the few guidelines described in
:ref:`formatting-mpl-docs`. Once it is converted, please include the module in
the API documentation and update the status in the table to "converted". Once
docstring conversion is complete and all the modules are available in the docs,
we can figure out how best to organize the API Reference and continue from
there.

====================   ===========   ===================
Module                 Author        Status
====================   ===========   ===================
backend_agg                          needs conversion
backend_cairo                        needs conversion
backend_cocoa                        needs conversion
backend_emf                          needs conversion
backend_fltkagg                      needs conversion
backend_gdk                          needs conversion
backend_gtk                          needs conversion
backend_gtkagg                       needs conversion
backend_gtkcairo                     needs conversion
backend_mixed                        needs conversion
backend_pdf                          needs conversion
backend_ps             Darren        needs conversion
backend_qt             Darren        needs conversion
backend_qtagg          Darren        needs conversion
backend_qt4            Darren        needs conversion
backend_qt4agg         Darren        needs conversion
backend_svg                          needs conversion
backend_template                     needs conversion
backend_tkagg                        needs conversion
backend_wx                           needs conversion
backend_wxagg                        needs conversion
backends/tkagg                       needs conversion
config/checkdep        Darren        needs conversion
config/cutils          Darren        needs conversion
config/mplconfig       Darren        needs conversion
config/mpltraits       Darren        needs conversion
config/rcparams        Darren        needs conversion
config/rcsetup         Darren        needs conversion
config/tconfig         Darren        needs conversion
config/verbose         Darren        needs conversion
projections/__init__   Mike          converted
projections/geo        Mike          converted (not included--experimental)
projections/polar      Mike          converted
afm                                  converted
artist                               converted
axes                                 converted
axis                                 converted
backend_bases                        converted
cbook                                converted
cm                                   converted
collections                          converted
colorbar                             converted
colors                               converted
contour                              needs conversion
dates                  Darren        needs conversion
dviread                Darren        needs conversion
figure                 Darren        needs conversion
finance                Darren        needs conversion
font_manager           Mike          converted
fontconfig_pattern     Mike          converted
image                                needs conversion
legend                               needs conversion
lines                  Mike & ???    converted
mathtext               Mike          converted
mlab                   John/Mike     converted
mpl                                  N/A
patches                Mike          converted
path                   Mike          converted
pylab                                N/A
pyplot                               converted
quiver                               needs conversion
rcsetup                              needs conversion
scale                  Mike          converted
table                                needs conversion
texmanager             Darren        needs conversion
text                   Mike          converted
ticker                 John          converted
transforms             Mike          converted
type1font                            needs conversion
units                                needs conversion
widgets                              needs conversion
====================   ===========   ===================

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

mathtext user's guide-- reviewed by JDH
---------------------------------------

This looks good (see :ref:`mathtext-tutorial`) -- there are a few
minor things to close the book on this chapter:

#. The main thing to wrap this up is getting the mathtext module
    ported over to rest and included in the API so the links from the
    user's guide tutorial work.

   - There's nothing in the mathtext module that I really consider a
     "public" API (i.e. that would be useful to people just doing
     plots).  If mathtext.py were to be documented, I would put it in
     the developer's docs.  Maybe I should just take the link in the
     user's guide out. - MGD

#. This section might also benefit from a little more detail on the
   customizations that are possible (eg an example fleshing out the rc
   options a little bit).  Admittedly, this is pretty clear from
   readin ghte rc file, but it might be helpful to a newbie.

   - The only rcParam that is currently useful is mathtext.fontset,
     which is documented here.  The others only apply when
     mathtext.fontset == 'custom', which I'd like to declare
     "unsupported".  It's really hard to get a good set of math fonts
     working that way, though it might be useful in a bind when
     someone has to use a specific wacky font for mathtext and only
     needs basics, like sub/superscripts. - MGD

#. There is still a TODO in the file to include a complete list of symbols

   - Done.  It's pretty extensive, thanks to STIX... - MGD

