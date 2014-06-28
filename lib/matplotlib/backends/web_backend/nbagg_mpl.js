var comm_websocket_adapter = function(comm) {
    // Create a "websocket"-like object which calls the given IPython comm
    // object with the appropriate methods. Currently this is a non binary
    // socket, so there is still some room for performance tuning.
    var ws = {};

    ws.close = function() {
        comm.close()
    };
    ws.send = function(m) {
        //console.log('sending', m);
        comm.send(m);
    };
    // Register the callback with on_msg.
    comm.on_msg(function(msg) {
        //console.log('receiving', msg['content']['data'], msg);
        // Pass the mpl event to the overriden (by mpl) onmessage function.
        ws.onmessage(msg['content']['data'])
    });
    return ws;
}

mpl.mpl_figure_comm = function(comm, msg) {
    // This is the function which gets called when the mpl process
    // starts-up an IPython Comm through the "matplotlib" channel.

    var id = msg.content.data.id;
    var element = $("#" + id);
    var ws_proxy = comm_websocket_adapter(comm)

    var fig = new mpl.figure(id, ws_proxy,
                           function() { },
                           element.get(0));

    // Call onopen now - mpl needs it, as it is assuming we've passed it a real
    // web socket which is closed, not our websocket->open comm proxy.
    ws_proxy.onopen();

    fig.parent_element = element.get(0);
    fig.cell_info = mpl.find_output_cell("<div id='" + id + "'></div>");

    var output_index = fig.cell_info[2]
    var cell = fig.cell_info[0];

    // Disable right mouse context menu.
    $(fig.rubberband_canvas).bind("contextmenu",function(e){
              return false;
    });

};

mpl.figure.prototype.handle_close = function(fig, msg) {
    // Update the output cell to use the data from the current canvas.
    fig.push_to_output();
    var dataURL = fig.canvas.toDataURL();
    $(fig.parent_element).html('<img src="' + dataURL + '">');
    fig.send_message('closing', {});
    fig.ws.close()
}

mpl.figure.prototype.push_to_output = function(remove_interactive) {
    var dataURL = this.canvas.toDataURL();
    this.cell_info[1]['text/html'] = '<img src="' + dataURL + '">';
}

mpl.figure.prototype.updated_canvas_event = function() {
    // Tell IPython that the notebook contents must change.
    IPython.notebook.set_dirty(true);
    this.send_message("ack", {});
    var fig = this;
    // Wait a second, then push the new image to the DOM so
    // that it is saved nicely (might be nice to debounce this).
    setTimeout(function () { fig.push_to_output() }, 1000);
}

mpl.figure.prototype._init_toolbar = function() {
    var fig = this;

    var nav_element = $('<div/>')
    nav_element.attr('style', 'width: 100%');
    this.root.append(nav_element);

    // Define a callback function for later on.
    function toolbar_event(event) {
        return fig.toolbar_button_onclick(event['data']);
    }
    function toolbar_mouse_event(event) {
        return fig.toolbar_button_onmouseover(event['data']);
    }

    for(var toolbar_ind in mpl.toolbar_items){
        var name = mpl.toolbar_items[toolbar_ind][0];
        var tooltip = mpl.toolbar_items[toolbar_ind][1];
        var image = mpl.toolbar_items[toolbar_ind][2];
        var method_name = mpl.toolbar_items[toolbar_ind][3];

        if (!name) { continue; };

        var button = $('<button class="btn btn-default" href="#" title="' + name + '"><i class="fa ' + image + ' fa-lg"></i></button>');
        button.click(method_name, toolbar_event);
        button.mouseover(tooltip, toolbar_mouse_event);
        nav_element.append(button);
    }

    // Add the status bar.
    var status_bar = $('<span class="mpl-message" style="text-align:right; float: right;"/>');
    nav_element.append(status_bar);
    this.message = status_bar[0];

    // Add the close button to the window.
    var buttongrp = $('<div class="btn-group inline pull-right"></div>');
    var button = $('<button class="btn btn-mini btn-danger" href="#" title="Close figure"><i class="fa icon-remove icon-large"></i></button>');
    button.click(function (evt) { fig.handle_close(fig, {}); } );
    button.mouseover('Close figure', toolbar_mouse_event);
    buttongrp.append(button);
    var titlebar = this.root.find($('.ui-dialog-titlebar'));
    titlebar.prepend(buttongrp);
}

mpl.find_output_cell = function(html_output) {
    // Return the cell and output element which can be found *uniquely* in the notebook.
    // Note - this is a bit hacky, but it is done because the "notebook_saving.Notebook"
    // IPython event is triggered only after the cells have been serialised, which for
    // our purposes (turning an active figure into a static one), is too late.
    var cells = IPython.notebook.get_cells();
    var ncells = cells.length;
    for (var i=0; i<ncells; i++) {
        var cell = cells[i];
        if (cell.cell_type == 'code'){
            for (var j=0; j<cell.output_area.outputs.length; j++) {
                var data = cell.output_area.outputs[j];
                if (cell.output_area.outputs[j]['text/html'] == html_output) {
                    var output = cell.output_area.outputs[j];
                    return [cell, output, j];
                }
            }
        }
    }
}

// Register the function which deals with the matplotlib target/channel.
// The kernel may be null if the page has been refreshed.
if (IPython.notebook.kernel != null) {
    IPython.notebook.kernel.comm_manager.register_target('matplotlib', mpl.mpl_figure_comm);
}
