// Import and setup functions to control Bootstrap's behavior.
import "@popperjs/core";
import { Tooltip } from "bootstrap";
import { documentReady } from "./mixin";

import "../styles/bootstrap.scss";

/*******************************************************************************
 * Trigger tooltips
 */

/**
 * Add tooltip to each element with the "tooltip" data-bs-toogle class
 */
function TriggerTooltip() {
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new Tooltip(tooltipTriggerEl, { delay: { show: 500, hide: 100 } });
  });
}

/*******************************************************************************
 * Call functions after document loading.
 */

documentReady(TriggerTooltip);
