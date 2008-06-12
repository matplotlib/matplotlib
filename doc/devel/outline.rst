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
text - mathtext                  Michael              accepted     John
text - usetex                    Darren               accepted     John
text - annotations               John                 submitted    ?
fonts et al                      Michael ?            no author    Darren
pyplot tut                       John                 submitted    Eric
configuration                    Darren               submitted    ?
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
coding guide                     John                 in review    Eric
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

coding guide (reviewed by EF)
-----------------------------

Mostly fine (see :ref:`coding-guide`), just a few comments.
Also, there are a couple of typos, but I would rather just edit those
directly in another pass (if you don't happen to find them) than
include them as formal review notes.

#. DONE - Import recommendation for ma: given that the trunk is
   requiring numpy 1.1, perhaps we should be consistent and
   recommend the form::

     import numpy.ma as ma

   for use in the trunk.
   A note about the difference between the two forms and the
   history can stay in place, and the alternative form would
   still be required for the maintenance branch, I presume.

#. This is peripheral, but regarding the example::

      mpl.rcParams['xtick.major.pad'] = 6

   At least at the application level, I think we should move
   towards using validation routinely when setting rcParams,
   to reduce a source of hard-to-find bugs.  I don't know to
   what extent Darren's traits-based system takes care of
   this, but if it does, that is a big point in its favor.
   There are alternatives (e.g. building validation into the
   rc() function and using that instead of setting the
   dictionary entries directly), if necessary.

   Darren notes:

   Validation is actually built into RcParams. This was done
   just prior to development of the traited config, validation is done using
   the mechanisms developed in rcsetup. For example::

     >>> rcParams['a.b']=1
     ---------------------------------------------------------------------------
     KeyError                                  Traceback (most recent call last)

     /home/darren/<ipython console> in <module>()

     /usr/lib64/python2.5/site-packages/matplotlib/__init__.pyc in __setitem__(self, key, val)
         555         except KeyError:
         556             raise KeyError('%s is not a valid rc parameter.\
     --> 557 See rcParams.keys() for a list of valid parameters.'%key)
         558
         559

     KeyError: 'a.b is not a valid rc parameter.See rcParams.keys() for a list of valid parameters.'

   also::

     rcParams['text.usetex']=''
     ---------------------------------------------------------------------------
     ValueError                                Traceback (most recent call last)

     /home/darren/<ipython console> in <module>()

     /usr/lib64/python2.5/site-packages/matplotlib/__init__.pyc in __setitem__(self, key, val)
         551 instead.'% (key, alt))
         552                 key = alt
     --> 553             cval = self.validate[key](val)
         554             dict.__setitem__(self, key, cval)
         555         except KeyError:

     /usr/lib64/python2.5/site-packages/matplotlib/rcsetup.pyc in validate_bool(b)
          56     elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False): return False
          57     else:
     ---> 58         raise ValueError('Could not convert "%s" to boolean' % b)
          59
          60 def validate_bool_maybe_none(b):

     ValueError: Could not convert "" to boolean



#. DONE - You give the example::

        import matplotlib.cbook as cbook

   Should there also be a list of the standard variants like
   ``mtransforms``?  (And, again peripherally, I would
   shorten that one to ``mtrans``.)

#. DONE - The treatment of whitespace is split into two parts
   separated by paragraphs on docstrings and line length;
   this can be consolidated.  It might be worth mentioning
   the ``reindent.py`` and ``tabnanny.py`` utilities here.

#. DONE - (removed first person usage) - Minor question of literary
   style: should use of the first person be avoided in most places?
   It is used, for example, in the discussion of the automatic kwarg
   doc generation.  I don't mind leaving the first person in, with the
   general understanding that it means you.

#. DONE - Licenses: you might want to add a link to your
   explanation of your BSD choice.  Peripheral question: is
   there any problem with basemap's inclusion of
   sub-packages with the gamut of licenses, GPL to MIT?


usetex user's guide-- reviewed by JDH
-------------------------------------

Review of :ref:`usetex-tutorial`:

#. DONE - In the section on the ps distiller, you might mention that it is the
   rasterization which some users find objectionable, and the distiller pram
   (eg 6000) is a dpi setting for the rasterizer.  Not everyone will
   immediately grasp the controversy surrounding dumping high res bitmaps into
   a ps file.

#. DONE - ``= Possible Hangups =`` - this is moin, not rest.  I have
    fixed this already, just wanted to point it out.  Also, for everything but
    top level chapters, I refer ``Upper lower`` for section titles, eg
    ``Possible hangups``.

#. DONE - in the troubleshooting section, could you add a FAQ showing how to
   find their .matplotlib dir (matplotlib.get_data_path) and link to
   it.

   I think you mean mpl.get_configdir. I added this to the faq/HOWTO, and
   linked to it from usetex.rst and customizing.rst. I also added the
   MPLCONFIGDIR environment variable to environment_variables_faq.rst.

   DONE - Also link to the PATH var and properly format
   ``text.latex.preample``.

   DONE - For the dirs, do we want `tex.cache` or
   ``tex.cache``?  I've been using the latter.  We could use rest
   specific markup for files and dirs, but I've been resisting goin
   whle hog onthe markup...  But we may eventually decide that is the
   better choice.

   I think we should use the directive provided by sphinx::

     :file:`tex.cache`

   I don't think that looks too ugly in ascii, its clear what it means. If you
   don't like it, I think we should use::

     ``tex.cache``

   which is formatted most
   similarly to the :file: directive in html. Let me know if you don't want to
   use the file directive.

   - JDH: let's go with the file directive.

#. DONE - try and use internal reference links for every section and be
   generous with the sections.  Eg, I added a section headers and
   internal linka for the postscript and unicode sections (we will
   want to be able to refer people to these easily from FAQs and mail
   list posts, etc)::

    .. _usetex-postscript:

    Postscript options
    ==================

Looks good!

Thanks!
