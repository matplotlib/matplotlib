
define([], function() {
    if (window.require) {
        window.require.config({
            map: {
                "*" : {
                    "matplotlib": "nbextensions/matplotlib/nbagg_mpl",
                    "jupyter-js-widgets": "nbextensions/jupyter-js-widgets/extension"
                }
            }
        });
    }

    // Export the required load_ipython_extention
    return {
        load_ipython_extension: function() {}
    };
});
