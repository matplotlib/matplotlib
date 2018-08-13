Removed ``pytz`` as a dependency
--------------------------------

Since ``dateutil`` and ``pytz`` both provide time zones, and matplotlib already depends on ``dateutil``, matplotlib will now use ``dateutil`` time zones internally and drop the redundant dependency on ``pytz``. While ``dateutil`` time zones are preferred (and currently recommended in the Python documentation), the explicit use of ``pytz`` zones is still supported.
