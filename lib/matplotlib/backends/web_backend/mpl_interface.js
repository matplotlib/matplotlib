var toolbar_items = [{% for name, tooltip, image, method in toolitems %}
    [{% if name is None %}'', '', '', ''{% else %}'{{ name }}', '{{ tooltip }}', '{{ image }}', '{{ method }}'{% end %}], {% end %}];


var extensions = [{% for filetype, extensions in sorted(canvas.get_supported_filetypes_grouped().items()) %}'{{ extensions[0] }}', {% end %}];
var default_extension = '{{ canvas.get_default_filetype() }}';


function init_mpl_canvas(fig) {
    var canvas_div = $('<div/>');
    canvas_div.attr('style', 'position: relative; clear: both;');
    fig.root.append(canvas_div);

    var canvas = $('<canvas/>');
    canvas.addClass('mpl-canvas');
    canvas.attr('style', "left: 0; top: 0; z-index: 0;")
    canvas.attr('width', '800');
    canvas.attr('height', '800');

    function canvas_keyboard_event(event) { return fig.key_event(event, event['data']); }
    canvas.keydown('key_press', canvas_keyboard_event);
    canvas.keyup('key_release', canvas_keyboard_event);

    canvas_div.append(canvas);

    fig.canvas = canvas[0];
    fig.context = canvas[0].getContext("2d");
    // Let the top level document handle key events.
    canvas.unbind('keydown');
    canvas.unbind('keyup');

    // create a second canvas which floats on top of the first.
    var rubberband = $('<canvas/>');
    rubberband.attr('style', "position: absolute; left: 0; top: 0; z-index: 1;")
    rubberband.attr('width', '800');
    rubberband.attr('height', '800');
    function mouse_event_fn(event) {
        return fig.mouse_event(event, event['data']);
    }
    rubberband.mousedown('button_press', mouse_event_fn);
    rubberband.mouseup('button_release', mouse_event_fn);
    rubberband.mousemove('motion_notify', mouse_event_fn);
    canvas_div.append(rubberband);

    fig.rubberband_canvas = rubberband[0];
    fig.rubberband_context = rubberband[0].getContext("2d");
    fig.rubberband_context.strokeStyle = "#000000";
};


function init_mpl_toolbar(fig) {
    var nav_element = $('<div/>')
    nav_element.attr('style', 'width: 600px');
    fig.root.append(nav_element);

    // Define a callback function for later on.
    function toolbar_event(event) {
        return fig.toolbar_button_onclick(event['data']); }
    function toolbar_mouse_event(event) {
        return fig.toolbar_button_onmouseover(event['data']); }

    for(var toolbar_ind in toolbar_items){
        var name = toolbar_items[toolbar_ind][0];
        var tooltip = toolbar_items[toolbar_ind][1];
        var image = toolbar_items[toolbar_ind][2];
        var method_name = toolbar_items[toolbar_ind][3];

        if (!name) {
            // put a spacer in here.
            continue;
        }

        var button = $('<button/>');
        button.addClass('ui-button ui-widget ui-state-default ui-corner-all ui-button-icon-only');
        button.attr('role', 'button');
        button.attr('aria-disabled', 'false');
        button.click(method_name, toolbar_event);
        button.mouseover(tooltip, toolbar_mouse_event);

        var icon_img = $('<span/>');
        icon_img.addClass('ui-button-icon-primary ui-icon');
        icon_img.addClass(image);
        icon_img.addClass('ui-corner-all');

        var tooltip_span = $('<span/>');
        tooltip_span.addClass('ui-button-text');
        tooltip_span.html(tooltip);

        button.append(icon_img);
        button.append(tooltip_span);

        nav_element.append(button);
    }

    var fmt_picker_span = $('<span/>');

    var fmt_picker = $('<select/>');
    fmt_picker.addClass('mpl-toolbar-option ui-widget ui-widget-content');
    fmt_picker_span.append(fmt_picker);
    nav_element.append(fmt_picker_span);
    fig.format_dropdown = fmt_picker;

    for (var ind in extensions) {
        var fmt = extensions[ind];
        var option = $('<option/>', {selected: fmt === default_extension}).html(fmt);
        fmt_picker.append(option)
    }

    // Add hover states to the ui-buttons
    $( ".ui-button" ).hover(
        function() { $(this).addClass("ui-state-hover");},
        function() { $(this).removeClass("ui-state-hover");}
    );

    var status_bar = $('<div class="mpl-message"/>');
    nav_element.append(status_bar);
    fig.message = status_bar[0];
};
