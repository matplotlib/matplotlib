Animations now default to video rich display in Jupyter notebooks
-----------------------------------------------------------------

When rich display of an Animation object is requested (i.e., by placing the variable as
the last line in a notebook cell), it will now default to the best available video
output. That is, if FFmpeg is available, a video file will be output, otherwise a
JavaScript animation will be output.
