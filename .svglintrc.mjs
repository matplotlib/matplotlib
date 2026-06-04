/** @type {import('svglint').Config} */
const config = {
    rules: {
        // Ensure all SVGs are valid XML.
        valid: true,

        // Block elements that can execute code or embed arbitrary content.
        // <script> can run arbitrary JavaScript; <foreignObject> and <iframe>
        // can embed arbitrary HTML.  Unlike event-handler attributes (which
        // are a legitimate tool for matplotlib's interactive SVG examples),
        // there is no use case in this repository for these elements outside
        // of a <script> already paired with its own exemption.
        elm: {
            "script": false,
            "foreignObject": false,
            "iframe": false,
        },

        custom: [
            // Block external URL references in href / xlink:href.
            // Internal fragment references (#id), data: URIs, and relative
            // paths are all fine.  http/https/ftp and protocol-relative URLs
            // are blocked because they cause the SVG renderer to make an
            // outbound network request, leaking the viewer's IP and UA to an
            // attacker-controlled server.
            (reporter, $, _ast) => {
                reporter.name = "no-external-references";
                const externalPattern = /^(https?:|ftp:|\/\/)/i;
                $("[href], [xlink\\:href]").each((_i, el) => {
                    if (!el.attribs) { return; }
                    const href =
                        el.attribs["href"] ?? el.attribs["xlink:href"];
                    if (href && externalPattern.test(href)) {
                        reporter.error(
                            `Found external reference '${href}' on <${el.name}>. ` +
                            "External URL references in SVGs cause the renderer " +
                            "to make an outbound request, leaking viewer IP/UA."
                        );
                    }
                });
            },
        ],
    },

    // These four files are intentional interactive SVG examples that
    // demonstrate matplotlib's SVG interactivity features.  They contain
    // embedded ECMAScript by design and are exempted from the <script> rule.
    ignore: [
        "doc/_static/svg_histogram.svg",
        "doc/_static/svg_tooltip.svg",
        "galleries/examples/user_interfaces/images/svg_histogram.svg",
        "galleries/examples/user_interfaces/images/svg_tooltip.svg",
    ],
};

export default config;
