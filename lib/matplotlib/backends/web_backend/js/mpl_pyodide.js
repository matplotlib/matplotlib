// mpl.js is imported before this, here we override functions from that and define new functions.

class MockJsWebSocket {
    binaryType = 'blob';

    constructor(fig_id) {
        this.fig_id = fig_id;
        this.readyState = 0;
        this.python_web_socket = null;
        this.python_onmessage_callback = null;
    }

    get onopen() {
        return this._onopen;
    }

    set onopen(listener) {
        this._onopen = listener;
        this.readyState = 1;
    }

    open(python_onmessage_callback) {
        this.python_onmessage_callback = python_onmessage_callback;
        this?.onopen();
    }

    receive_binary(content, _binary = true) {
        var buffer = content.getBuffer();
        content.destroy();
        try {
            const data = new Blob([buffer.data]);
            this.onmessage({ data });
        } finally {
            buffer.release(); // Release the memory when we're done
        }
    }

    receive_json(data) {
        this.onmessage({ data });
    }

    send(content) {
        this?.python_onmessage_callback(content);
    }

    _onopen = null;
}

mpl.get_websocket_type = function () {
    return MockJsWebSocket;
};

mpl.figure.prototype.handle_save = function (fig, _msg) {
    var format_dropdown = fig.format_dropdown;
    var format = format_dropdown.options[format_dropdown.selectedIndex].value;
    this.ws.send(
        JSON.stringify({ type: 'save', figure_id: this.id, format: format })
    );
};

mpl.figure.prototype._create_toolbar_image = function (image_name) {
    // Cache image blob URLs for reuse across multiple plots on the same page.
    if (mpl._toolbar_urls == undefined) {
        mpl._toolbar_urls = {};
    }
    let blob_url = mpl._toolbar_urls[image_name];
    if (!blob_url) {
        const image_bytes = mpl
            .toolbar_image_callback(image_name)
            .toJs({ create_pyproxies: false });
        const blob = new Blob([image_bytes], { type: 'image/png' });
        blob_url = URL.createObjectURL(blob);
        mpl._toolbar_urls[image_name] = blob_url;
    }

    const icon_img = new Image();
    icon_img.src = blob_url;
    return icon_img;
};
