Add ``cache_frame_data`` keyword-only argument into ``matplotlib.animation.FuncAnimation``
------------------------------------------------------------------------------------------

| ``matplotlib.animation.FuncAnimation`` has been caching frame data by default.

| However, this caching is not ideal in certain cases.
| e.g. When ``FuncAnimation`` needs to be only drawn(not saved) interactively and memory required by frame data is quite large.

| By adding ``cache_frame_data`` keyword-only argument, users can disable this caching now if necessary.
| Thereby, this new argument provides a fix for issue #8528.
