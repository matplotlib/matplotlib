/* This .js file contains functions for matplotlib's built-in
   tornado-based server, that are not relevant when embedding WebAgg
   in another web application. */

// eslint-disable-next-line no-unused-vars
function mpl_ondownload(figure, format) {
    window.open(figure.id + '/download.' + format, '_blank');
}
