function figure(fig_id, websocket_url_prefix) {
    this.id = fig_id;
    
    if (typeof(WebSocket) !== 'undefined') {
        this.WebSocket = WebSocket;
    } else if (typeof(MozWebSocket) !== 'undefined') {
        this.WebSocket = MozWebSocket;
    } else {
        alert('Your browser does not have WebSocket support.' +
              'Please try Chrome, Safari or Firefox ≥ 6. ' +
              'Firefox 4 and 5 are also supported but you ' +
              'have to enable WebSockets in about:config.');
    };
    
    
    this.ws = new this.WebSocket(websocket_url_prefix + fig_id + '/ws');
    
    this.supports_binary = (this.ws.binaryType != undefined);

    if (!this.supports_binary) {
        var warnings = document.getElementById("mpl-warnings");
        warnings.style.display = 'block';
        warnings.textContent = (
            "This browser does not support binary websocket messages. " +
            "Performance may be slow.");
    }
    
    this.imageObj = new Image();
    
    this.context = undefined;
    this.message = undefined;
    this.canvas = undefined;
    this.rubberband_canvas = undefined;
    this.rubberband_context = undefined;
    this.format_dropdown = undefined;
    
    this.focus_on_mousover = false;
    
}

figure.prototype.finalize = function (canvas_id_prefix, toolbar_id_prefix, message_id_prefix) {
    // resizing_div_id might be the canvas or a containing div for more control of display

    var canvas_id = canvas_id_prefix + '-canvas';
    var rubberband_id = canvas_id_prefix + '-rubberband-canvas';
    var message_id = message_id_prefix + '-message';

    this.message = document.getElementById(message_id);
    this.canvas = document.getElementById(canvas_id);
    this.context = this.canvas.getContext("2d");
    this.rubberband_canvas = document.getElementById(rubberband_id);
    this.rubberband_context = this.rubberband_canvas.getContext("2d");
    this.rubberband_context.strokeStyle = "#000000";
    
    this.format_dropdown = document.getElementById(toolbar_id_prefix + '-format_picker');
    
    this.ws.onopen = function () {
        this.ws.send(JSON.stringify(
            {type: 'supports_binary',
             value: this.supports_binary}));
    }
    
    // attach the onload function to the image object when an
    // image has been recieved via onmessage
    fig = this
    onload_creator = function(fig) {return function() {fig.context.drawImage(fig.imageObj, 0, 0);};};
    this.imageObj.onload = onload_creator(fig);
    

    this.imageObj.onunload = function() {
        this.ws.close();
    }

    this.ws.onmessage = gen_on_msg_fn(this);
};


function gen_on_msg_fn(fig)
{
    return function socket_on_message(evt) {
        if (fig.supports_binary) {
            if (evt.data instanceof Blob) {
                /* FIXME: We get "Resource interpreted as Image but
                 * transferred with MIME type text/plain:" errors on
                 * Chrome.  But how to set the MIME type?  It doesn't seem
                 * to be part of the websocket stream */
                evt.data.type = "image/png";
    
                /* Free the memory for the previous frames */
                if (fig.imageObj.src) {
                    (window.URL || window.webkitURL).revokeObjectURL(
                        fig.imageObj.src);
                }
                fig.imageObj.src = (window.URL || window.webkitURL).createObjectURL(
                    evt.data);
                fig.ws.send('{"type": "ack"}')
                return;
            }
        } else {
            if (evt.data.slice(0, 21) == "data:image/png;base64") {
                fig.imageObj.src = evt.data;
                fig.ws.send('{"type": "ack"}')
                return;
            }
        }
    
        var msg = JSON.parse(evt.data);
    
        switch(msg['type']) {
        case 'message':
            fig.message.textContent = msg['message'];
            break;
    
        case 'cursor':
            var cursor = msg['cursor'];
            switch(cursor)
            {
            case 0:
                cursor = 'pointer';
                break;
            case 1:
                cursor = 'default';
                break;
            case 2:
                cursor = 'crosshair';
                break;
            case 3:
                cursor = 'move';
                break;
            }
            fig.canvas.style.cursor = cursor;
            break;
    
        case 'resize':
            var size = msg['size'];
            if (size[0] != fig.canvas.width || size[1] != fig.canvas.height) {
                fig.canvas.width = size[0];
                fig.canvas.height = size[1];
                fig.rubberband_canvas.width = size[0];
                fig.rubberband_canvas.height = size[1];
                fig.ws.send(JSON.stringify({type: 'refresh'}));
                fig.ws.send(JSON.stringify(
                    {type: 'supports_binary',
                     value: fig.supports_binary}));
            }
            break;
    
        case 'rubberband':
            var x0 = msg['x0'];
            var y0 = fig.canvas.height - msg['y0'];
            var x1 = msg['x1'];
            var y1 = fig.canvas.height - msg['y1'];
            x0 = Math.floor(x0) + 0.5;
            y0 = Math.floor(y0) + 0.5;
            x1 = Math.floor(x1) + 0.5;
            y1 = Math.floor(y1) + 0.5;
            var min_x = Math.min(x0, x1);
            var min_y = Math.min(y0, y1);
            var width = Math.abs(x1 - x0);
            var height = Math.abs(y1 - y0);
    
            fig.rubberband_context.clearRect(
                0, 0, fig.canvas.width, fig.canvas.height);
            fig.rubberband_context.strokeRect(min_x, min_y, width, height);
            break;
        }
    };
};




function findPos(obj) {
    // Find the position of the given HTML node.
    
    var curleft = 0, curtop = 0;
    if (obj.offsetParent) {
        do {
            curleft += obj.offsetLeft;
            curtop += obj.offsetTop;
        } while (obj = obj.offsetParent);
        return { x: curleft, y: curtop };
    }
    return undefined;
}

figure.prototype.mouse_event = function(event, name) {
    var canvas_pos = findPos(this.canvas)
    
    if (this.focus_on_mouseover && name === 'motion_notify')
    {
        this.canvas.focus();
    }
    
    var x = event.pageX - canvas_pos.x;
    var y = event.pageY - canvas_pos.y;

    this.ws.send(JSON.stringify(
        {type: name,
         x: x, y: y,
         button: event.button}));

    /* This prevents the web browser from automatically changing to
     * the text insertion cursor when the button is pressed.  We want
     * to control all of the cursor setting manually through the
     * 'cursor' event from matplotlib */
    event.preventDefault();
    return false;   
}

figure.prototype.key_event = function(event, name) {
    /* Don't fire events just when a modifier is changed.  Modifiers are
       sent along with other keys. */
    if (event.keyCode >= 16 && event.keyCode <= 20) {
        return;
    }

    value = '';
    if (event.ctrlKey) {
        value += "ctrl+";
    }
    if (event.altKey) {
        value += "alt+";
    }
    value += String.fromCharCode(event.keyCode).toLowerCase();

    this.ws.send(JSON.stringify(
        {type: name,
         key: value}));
}

figure.prototype.toolbar_button_onclick = function(name) {
    if (name == 'download') {
        var format_dropdown = this.format_dropdown;
        var format = format_dropdown.options[format_dropdown.selectedIndex].value;
        window.open(this.id + '/download.' + format, '_blank');
    } else {
        this.ws.send(JSON.stringify(
            {type: "toolbar_button",
             "name": name}));
    }
};


figure.prototype.toolbar_button_onmouseover = function(tooltip) {
    this.message.textContent = tooltip;
};


