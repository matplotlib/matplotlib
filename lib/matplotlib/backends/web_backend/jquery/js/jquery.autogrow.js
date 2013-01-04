/*
 * Auto Grow Textarea Plugin
 * by Jevin 5/11/2010
 * http://www.technoreply.com/autogrow-textarea-plugin/
 *
 * Modified by Rob G (aka Fudgey/Mottie)
 *  - Converted into a plugin
 *  - Added ability to calculate approximate # cols when textarea is set to 100%
 *
 * Simplified by Brian Granger on 5/2/2011
 */

(function($) {
    $.fn.autogrow = function() {

    var grow = function(d) {
        var linesCount = 0;
        // modified split rule from
        // http://stackoverflow.com/questions/2035910/how-to-get-the-number-of-lines-in-a-textarea/2036424#2036424
        var lines = d.txt.value.split(/\r|\r\n|\n/);
        linesCount = lines.length;
        if (linesCount >= d.rowsDefault) {
            d.txt.rows = linesCount;
        } else {
            d.txt.rows = d.rowsDefault;
        }
    };

    return this.each(function() {
        var d = {
            colsDefault : 0,
            rowsDefault : 1,
            txt         : this,
            $txt        : $(this)
        };
        d.txt.onkeyup = function() {
            grow(d);
        };
        grow(d);
    });
};
})(jQuery);
