
.. _saved_replies:

Saved replies
=============

To facilitate DRY (do not repeat yourself) communication principles, reduce potential bias and errors in
communication, and consistently maintain respectful and welcoming communication we decided to develop and collect a set
of reusable replies for recurring project needs.

Our goal is to write welcoming and friendly communication which encourage participation and contribution in various
ways and at different levels of technical experience, as well as clarifies what we are able, and sometimes are not able,
to support our community with.

As a maintainer, triage team member, or any other active community member, it may be helpful to store some of these in
your GitHub account’s `saved replies`_ for reviewing issues and PRs, and supporting other community members, as they participate
and contribute to Matplotlib.

Our saved replies can be long and detailed. This is because we aim to offer support without assuming the level of
experience, or the intent the community member we are addressing might have. There is still room for one to tweak and
personalize the message, if based on the ongoing conversation some parts are redundant, require further information, or
if they want to add a personal note to suggest that they will be available to offer support, for example.

Our intent with providing a baseline is to make it easier to not miss the minimum amount of information, and
consistently facilitate the respectful and welcoming interactions we aspire to maintain, not to set rigid communication
constraints. We trust our maintainers, triage team members, and all community members who continue to abide by our
`Code of Conduct`_, to communicate freely. It is based on their ongoing experience that we are able to provide these saved
replies to begin with.

.. _Code of Conduct: https://github.com/matplotlib/matplotlib/blob/master/CODE_OF_CONDUCT.md
.. _saved replies: https://github.com/settings/replies/

Issues
------

Coming soon 😉

Pull Requests
-------------

First Pull Request Merged
^^^^^^^^^^^^^^^^^^^^^^^^^
Hi-five ✋ on merging your first pull request to Matplotlib, @username! We hope you stick around and invite you to continue to take an active part in Matplotlib!

Your choices aren’t limited to programming 😉 – you can review pull requests, help us stay on top of new and old issues, develop educational material, refresh our documentation, work on our website, create marketing materials, translate website content, write grant proposals, and help with other fundraising initiatives. For more info, check out: https://matplotlib.org/stable/devel/index

Also, consider joining our `mailing list`_ ✉️. This is a great way to connect with other people in our community and be part of important conversations that affect the development of Matplotlib.

If you haven’t yet, do join us on `gitter`_ and `discourse`_ 🗣. The former is a chat platform, which is great when you have any questions on process or how to fix something, the latter is a forum which is useful for longer questions and discussions.

Last but not least, we have a monthly meeting 👥 for new contributors and a weekly meeting for the maintainers, everyone is welcome to join both! You can find out more about our regular project meetings in this `calendar page`_.

.. _mailing list: https://mail.python.org/mailman/listinfo/matplotlib-devel
.. _discourse: https://discourse.matplotlib.org/
.. _calendar page: https://scientific-python.org/calendars/

Rebase
^^^^^^

Hi, it looks like we’d need to rebase your code to make sure it includes changes that have since occurred in the main
repository. Would you like to do this yourself, or would you like us to do this for you? I’m asking because a rebase can
get a bit fiddly and not everyone likes doing them 😉

If you want to rebase, the first thing to do is to squash all your commits into one, which will make the job easier.
Make sure you are in the PR branch, then to rebase, for eleven commits do::

$ git rebase --interactive HEAD~11


And follow the instructions, basically replace 'pick' by 'f' in all but the first commit. Then update your main branch
from upstream, change back to the PR branch and do::

$ git rebase main


and if there are problems, do ``$ git status`` to see which files need fixing, then edit the files to fix up any conflicts
(sections marked by "<<<") . When you are done with that::

$ git add <the fixed files>
$ git rebase --continue
$ git push --force-with-lease origin HEAD


If you have any problems, feel free to ask questions.


PS.
If at any point anything goes wrong, and you don't know what to do, just do::

$ git rebase --abort

and everything will go back to the way it was in the before ``$ git rebase`` times, and you can come back here or to
`gitter`_, and ask us for help, or that we do the rebase after all 😉.

.. _gitter: https://gitter.im/matplotlib/matplotlib