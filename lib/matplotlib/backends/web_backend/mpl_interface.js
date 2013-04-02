var toolbar_items = [{% for name, tooltip, image, method in toolitems %}
    [{% if name is None %}'', '', '', ''{% else %}'{{ name }}', '{{ tooltip }}', '{{ image }}', '{{ method }}'{% end %}], {% end %}];


var extensions = [{% for filetype, extensions in sorted(canvas.get_supported_filetypes_grouped().items()) %}'{{ extensions[0] }}', {% end %}];
var default_extension = '{{ canvas.get_default_filetype() }}';


function init_mpl_canvas(fig, canvas_div_id, id_prefix) {

    var canvas_div = $(document.getElementById(canvas_div_id));
    canvas_div.attr('style', 'position: relative; clear: both;');
    
    var canvas = $('<canvas/>', {id: id_prefix + '-canvas'});
    canvas.attr('id', id_prefix + '-canvas');
    canvas.addClass('mpl-canvas');
    canvas.attr('style', "left: 0; top: 0; z-index: 0;")
    canvas.attr('width', '800');
    canvas.attr('height', '800');
    
    function canvas_keyboard_event(event) { return fig.key_event(event, event['data']); }
    canvas.keydown('key_press', canvas_keyboard_event); 
    canvas.keyup('key_release', canvas_keyboard_event); 
    
    canvas_div.append(canvas);
    
    // create a second canvas which floats on top of the first.
    var rubberband = $('<canvas/>', {id: id_prefix + '-rubberband-canvas'});
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
};


function init_mpl_statusbar(container_id, id_prefix) {
    var status_bar = $('<span class="mpl-message"/>');
    var status_id = id_prefix + '-message';
    status_bar.attr('id', status_id);
    $(document.getElementById(container_id)).append(status_bar);
    return status_id
};

function init_mpl_toolbar(fig, nav_container_id, nav_elem_id_prefix) {
        // Adds a navigation toolbar to the object found with the given jquery query string
        
        if (nav_elem_id_prefix === undefined) {
            nav_elem_id_prefix = nav_container_id;
        }
        
        // Define a callback function for later on.
        function toolbar_event(event) { return fig.toolbar_button_onclick(event['data']); }
        function toolbar_mouse_event(event) { return fig.toolbar_button_onmouseover(event['data']); }

        var nav_element = $(document.getElementById(nav_container_id));
        
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
            button.attr("id", nav_elem_id_prefix + name);
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

        var fmt_picker = $('<select/>', {id: nav_elem_id_prefix + '-format_picker'});
        fmt_picker.addClass('mpl-toolbar-option ui-widget ui-widget-content');
        fmt_picker_span.append(fmt_picker);
        nav_element.append(fmt_picker_span);

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
};