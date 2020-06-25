# Check for the mpl_toolkits_baseline_images.
try:
    import mpl_toolkits_baseline_images
except:
    raise ImportError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install mpl_toolkits_baseline_images to get the '
        'test data.')
