var ws;

function ws_url(path) {
    var loc = window.location
    var new_uri;

    new_uri = "ws://" + loc.host;
    new_uri += loc.pathname;
    new_uri += path;

    return new_uri;
}

window.onload = function() {
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

    var message = document.getElementById("mpl-message");
    var canvas_div = document.getElementById("mpl-canvas-div");
    var canvas = document.getElementById("mpl-canvas");
    var context = canvas.getContext("2d");
    var rubberband_canvas = document.getElementById("mpl-rubberband-canvas");
    var rubberband_context = rubberband_canvas.getContext("2d");
    rubberband_context.strokeStyle = "#000000";

    ws = new this.WebSocket(ws_url("ws"));

    var supports_binary = (ws.binaryType != undefined);

    if (!supports_binary) {
        var warnings = document.getElementById("mpl-warnings");
        warnings.style.display = 'block';
        warnings.textContent = (
            "This browser does not support binary websocket messages. " +
            "Performance may be slow.");
    }

    ws.onopen = function () {
        ws.send(JSON.stringify(
            {type: 'supports_binary',
             value: supports_binary}));
    }

    ws.onmessage = function (evt) {
        if (supports_binary) {
            if (evt.data instanceof Blob) {
                /* FIXME: We get "Resource interpreted as Image but
                 * transferred with MIME type text/plain:" errors on
                 * Chrome.  But how to set the MIME type?  It doesn't seem
                 * to be part of the websocket stream */
                evt.data.type = "image/png";

                /* Free the memory for the previous frames */
                if (imageObj.src) {
                    (window.URL || window.webkitURL).revokeObjectURL(
                        imageObj.src);
                }

                imageObj.src = (window.URL || window.webkitURL).createObjectURL(
                    evt.data);
                return;
            }
        } else {
            if (evt.data.slice(0, 21) == "data:image/png;base64") {
                imageObj.src = evt.data;
                return;
            }
        }

        var msg = JSON.parse(evt.data);

        switch(msg['type']) {
        case 'message':
            message.textContent = msg['message'];
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
            canvas.style.cursor = cursor;
            break;

        case 'resize':
            var size = msg['size'];
            if (size[0] != canvas.width || size[1] != canvas.height) {
                var div = document.getElementById("mpl-div");
                canvas.width = size[0];
                canvas.height = size[1];
                rubberband_canvas.width = size[0];
                rubberband_canvas.height = size[1];
                canvas_div.style.width = size[0];
                canvas_div.style.height = size[1];
                div.style.width = size[0];
                ws.send(JSON.stringify({type: 'refresh'}));
            }
            break;

        case 'rubberband':
            var x0 = msg['x0'];
            var y0 = rubberband_canvas.height - msg['y0'];
            var x1 = msg['x1'];
            var y1 = rubberband_canvas.height - msg['y1'];
            x0 = Math.floor(x0) + 0.5;
            y0 = Math.floor(y0) + 0.5;
            x1 = Math.floor(x1) + 0.5;
            y1 = Math.floor(y1) + 0.5;
            var min_x = Math.min(x0, x1);
            var min_y = Math.min(y0, y1);
            var width = Math.abs(x1 - x0);
            var height = Math.abs(y1 - y0);

            rubberband_context.clearRect(
                0, 0, rubberband_canvas.width, rubberband_canvas.height);
            rubberband_context.strokeRect(min_x, min_y, width, height);
            break;
        }
    };

    imageObj = new Image();
    imageObj.onload = function() {
        context.drawImage(imageObj, 0, 0);
    };
};

function mouse_event(event, name) {
    var canvas_div = document.getElementById("mpl-canvas-div");
    var x = event.pageX - canvas_div.offsetLeft;
    var y = event.pageY - canvas_div.offsetTop;

    ws.send(JSON.stringify(
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

function key_event(event, name) {
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

    ws.send(JSON.stringify(
        {type: name,
         key: value}));
}

function toolbar_button_onclick(name) {
    if (name == 'download') {
        var format_dropdown = document.getElementById("mpl-format");
        var format = format_dropdown.options[format_dropdown.selectedIndex].value;
        window.open('download.' + format, '_blank');
    } else {
        ws.send(JSON.stringify(
            {type: "toolbar_button",
             "name": name}));
    }
}

function toolbar_button_onmouseover(name) {
    var message = document.getElementById("mpl-message");
    message.textContent = name;
}
