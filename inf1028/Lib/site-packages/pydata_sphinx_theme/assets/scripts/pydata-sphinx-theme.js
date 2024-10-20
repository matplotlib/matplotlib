// Define the custom behavior of the page
import { documentReady } from "./mixin";
import { compare, validate } from "compare-versions";

import "../styles/pydata-sphinx-theme.scss";

/*******************************************************************************
 * Theme interaction
 */

var prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

/**
 * set the the body theme to the one specified by the user browser
 *
 * @param {event} e
 */
function autoTheme(e) {
  document.documentElement.dataset.theme = prefersDark.matches
    ? "dark"
    : "light";
}

/**
 * Set the theme using the specified mode.
 * It can be one of ["auto", "dark", "light"]
 *
 * @param {str} mode
 */
function setTheme(mode) {
  if (mode !== "light" && mode !== "dark" && mode !== "auto") {
    console.error(`Got invalid theme mode: ${mode}. Resetting to auto.`);
    mode = "auto";
  }

  // get the theme
  var colorScheme = prefersDark.matches ? "dark" : "light";
  document.documentElement.dataset.mode = mode;
  var theme = mode == "auto" ? colorScheme : mode;
  document.documentElement.dataset.theme = theme;
  // TODO: remove this line after Bootstrap upgrade
  // v5.3 has a colors mode: https://getbootstrap.com/docs/5.3/customize/color-modes/
  document.querySelectorAll(".dropdown-menu").forEach((el) => {
    if (theme === "dark") {
      el.classList.add("dropdown-menu-dark");
    } else {
      el.classList.remove("dropdown-menu-dark");
    }
  });

  // save mode and theme
  localStorage.setItem("mode", mode);
  localStorage.setItem("theme", theme);
  console.log(`[PST]: Changed to ${mode} mode using the ${theme} theme.`);

  // add a listener if set on auto
  prefersDark.onchange = mode == "auto" ? autoTheme : "";
}

/**
 * Change the theme option order so that clicking on the btn is always a change
 * from "auto"
 */
function cycleMode() {
  const defaultMode = document.documentElement.dataset.defaultMode || "auto";
  const currentMode = localStorage.getItem("mode") || defaultMode;

  var loopArray = (arr, current) => {
    var nextPosition = arr.indexOf(current) + 1;
    if (nextPosition === arr.length) {
      nextPosition = 0;
    }
    return arr[nextPosition];
  };

  // make sure the next theme after auto is always a change
  var modeList = prefersDark.matches
    ? ["auto", "light", "dark"]
    : ["auto", "dark", "light"];
  var newMode = loopArray(modeList, currentMode);
  setTheme(newMode);
}

/**
 * add the theme listener on the btns of the navbar
 */
function addModeListener() {
  // the theme was set a first time using the initial mini-script
  // running setMode will ensure the use of the dark mode if auto is selected
  setTheme(document.documentElement.dataset.mode);

  // Attach event handlers for toggling themes colors
  document.querySelectorAll(".theme-switch-button").forEach((el) => {
    el.addEventListener("click", cycleMode);
  });
}

/*******************************************************************************
 * TOC interactivity
 */

/**
 * TOC sidebar - add "active" class to parent list
 *
 * Bootstrap's scrollspy adds the active class to the <a> link,
 * but for the automatic collapsing we need this on the parent list item.
 *
 * The event is triggered on "window" (and not the nav item as documented),
 * see https://github.com/twbs/bootstrap/issues/20086
 */
function addTOCInteractivity() {
  window.addEventListener("activate.bs.scrollspy", function () {
    const navLinks = document.querySelectorAll(".bd-toc-nav a");

    navLinks.forEach((navLink) => {
      navLink.parentElement.classList.remove("active");
    });

    const activeNavLinks = document.querySelectorAll(".bd-toc-nav a.active");
    activeNavLinks.forEach((navLink) => {
      navLink.parentElement.classList.add("active");
    });
  });
}

/*******************************************************************************
 * Scroll
 */

/**
 * Navigation sidebar scrolling to active page
 */
function scrollToActive() {
  // If the docs nav doesn't exist, do nothing (e.g., on search page)
  if (!document.querySelector(".bd-docs-nav")) {
    return;
  }

  var sidebar = document.querySelector("div.bd-sidebar");

  // Remember the sidebar scroll position between page loads
  // Inspired on source of revealjs.com
  let storedScrollTop = parseInt(
    sessionStorage.getItem("sidebar-scroll-top"),
    10,
  );

  if (!isNaN(storedScrollTop)) {
    // If we've got a saved scroll position, just use that
    sidebar.scrollTop = storedScrollTop;
    console.log("[PST]: Scrolled sidebar using stored browser position...");
  } else {
    // Otherwise, calculate a position to scroll to based on the lowest `active` link
    var sidebarNav = document.querySelector(".bd-docs-nav");
    var active_pages = sidebarNav.querySelectorAll(".active");
    if (active_pages.length > 0) {
      // Use the last active page as the offset since it's the page we're on
      var latest_active = active_pages[active_pages.length - 1];
      var offset =
        latest_active.getBoundingClientRect().y -
        sidebar.getBoundingClientRect().y;
      // Only scroll the navbar if the active link is lower than 50% of the page
      if (latest_active.getBoundingClientRect().y > window.innerHeight * 0.5) {
        let buffer = 0.25; // Buffer so we have some space above the scrolled item
        sidebar.scrollTop = offset - sidebar.clientHeight * buffer;
        console.log("[PST]: Scrolled sidebar using last active link...");
      }
    }
  }

  // Store the sidebar scroll position
  window.addEventListener("beforeunload", () => {
    sessionStorage.setItem("sidebar-scroll-top", sidebar.scrollTop);
  });
}

/*******************************************************************************
 * Search
 */

/**
 * Find any search forms on the page and return their input element
 */
var findSearchInput = () => {
  let forms = document.querySelectorAll("form.bd-search");
  if (!forms.length) {
    // no search form found
    return;
  } else {
    var form;
    if (forms.length == 1) {
      // there is exactly one search form (persistent or hidden)
      form = forms[0];
    } else {
      // must be at least one persistent form, use the first persistent one
      form = document.querySelector(
        "div:not(.search-button__search-container) > form.bd-search",
      );
    }
    return form.querySelector("input");
  }
};

/**
 * Activate the search field on the page.
 * - If there is a search field already visible it will be activated.
 * - If not, then a search field will pop up.
 */
var toggleSearchField = () => {
  // Find the search input to highlight
  let input = findSearchInput();

  // if the input field is the hidden one (the one associated with the
  // search button) then toggle the button state (to show/hide the field)
  let searchPopupWrapper = document.querySelector(".search-button__wrapper");
  let hiddenInput = searchPopupWrapper.querySelector("input");
  if (input === hiddenInput) {
    searchPopupWrapper.classList.toggle("show");
  }
  // when toggling off the search field, remove its focus
  if (document.activeElement === input) {
    input.blur();
  } else {
    input.focus();
    input.select();
    input.scrollIntoView({ block: "center" });
  }
};

/**
 * Add an event listener for toggleSearchField() for Ctrl/Cmd + K
 */
var addEventListenerForSearchKeyboard = () => {
  window.addEventListener(
    "keydown",
    (event) => {
      let input = findSearchInput();
      // toggle on Ctrl+k or ⌘+k
      if (
        // Ignore if shift or alt are pressed
        !event.shiftKey &&
        !event.altKey &&
        // On Mac use ⌘, all other OS use Ctrl
        (useCommandKey
          ? event.metaKey && !event.ctrlKey
          : !event.metaKey && event.ctrlKey) &&
        // Case-insensitive so the shortcut still works with caps lock
        /^k$/i.test(event.key)
      ) {
        event.preventDefault();
        toggleSearchField();
      }
      // also allow Escape key to hide (but not show) the dynamic search field
      else if (document.activeElement === input && /Escape/i.test(event.key)) {
        toggleSearchField();
      }
    },
    true,
  );
};

/**
 * If the user is on a Mac, use command (⌘) instead of control (ctrl) key
 *
 * Note: `navigator.platform` is deprecated; however MDN still recommends using
 * it for the one specific use case of detecting whether a keyboard shortcut
 * should use control or command:
 * https://developer.mozilla.org/en-US/docs/Web/API/Navigator/platform#examples
 */
var useCommandKey =
  navigator.platform.indexOf("Mac") === 0 || navigator.platform === "iPhone";

/**
 * Change the search hint to `meta key` if we are a Mac
 */

var changeSearchShortcutKey = () => {
  let shortcuts = document.querySelectorAll(".search-button__kbd-shortcut");
  if (useCommandKey) {
    shortcuts.forEach(
      (f) => (f.querySelector("kbd.kbd-shortcut__modifier").innerText = "⌘"),
    );
  }
};

/**
 * Activate callbacks for search button popup
 */
var setupSearchButtons = () => {
  changeSearchShortcutKey();
  addEventListenerForSearchKeyboard();

  // Add the search button trigger event callback
  document.querySelectorAll(".search-button__button").forEach((btn) => {
    btn.onclick = toggleSearchField;
  });

  // Add the search button overlay event callback
  let overlay = document.querySelector(".search-button__overlay");
  if (overlay) {
    overlay.onclick = toggleSearchField;
  }
};

/*******************************************************************************
 * Version Switcher
 * Note that this depends on two variables existing that are defined in
 * and `html-page-context` hook:
 *
 * - DOCUMENTATION_OPTIONS.pagename
 * - DOCUMENTATION_OPTIONS.theme_switcher_url
 */

/**
 * path component of URL
 */
var getCurrentUrlPath = () => {
  if (DOCUMENTATION_OPTIONS.BUILDER == "dirhtml") {
    return DOCUMENTATION_OPTIONS.pagename == "index"
      ? `/`
      : `${DOCUMENTATION_OPTIONS.pagename}/`;
  }
  return `${DOCUMENTATION_OPTIONS.pagename}.html`;
};

/**
 * Allow user to dismiss the warning banner about the docs version being dev / old.
 * We store the dismissal date and version, to give us flexibility about making the
 * dismissal last for longer than one browser session, if we decide to do that.
 *
 * @param {event} event the event that trigger the check
 */
async function DismissBannerAndStorePref(event) {
  const banner = document.querySelector("#bd-header-version-warning");
  banner.remove();
  const version = DOCUMENTATION_OPTIONS.VERSION;
  const now = new Date();
  const banner_pref = JSON.parse(
    localStorage.getItem("pst_banner_pref") || "{}",
  );
  console.debug(
    `[PST] Dismissing the version warning banner on ${version} starting ${now}.`,
  );
  banner_pref[version] = now;
  localStorage.setItem("pst_banner_pref", JSON.stringify(banner_pref));
}

/**
 * Check if corresponding page path exists in other version of docs
 * and, if so, go there instead of the homepage of the other docs version
 *
 * @param {event} event the event that trigger the check
 */
async function checkPageExistsAndRedirect(event) {
  // ensure we don't follow the initial link
  event.preventDefault();
  const currentFilePath = getCurrentUrlPath();
  let tryUrl = event.currentTarget.getAttribute("href");
  let otherDocsHomepage = tryUrl.replace(currentFilePath, "");
  try {
    let head = await fetch(tryUrl, { method: "HEAD" });
    if (head.ok) {
      location.href = tryUrl; // the page exists, go there
    } else {
      location.href = otherDocsHomepage;
    }
  } catch (err) {
    // something went wrong, probably CORS restriction, fallback to other docs homepage
    location.href = otherDocsHomepage;
  }
}

/**
 * Load and parse the version switcher JSON file from an absolute or relative URL.
 *
 * @param {string} url The URL to load version switcher entries from.
 */
async function fetchVersionSwitcherJSON(url) {
  // first check if it's a valid URL
  try {
    var result = new URL(url);
  } catch (err) {
    if (err instanceof TypeError) {
      if (!window.location.origin) {
        // window.location.origin is null for local static sites
        // (ie. window.location.protocol == 'file:')
        //
        // TODO: Fix this to return the static version switcher by working out
        // how to get the correct path to the switcher JSON file on local static builds
        return null;
      }
      // assume we got a relative path, and fix accordingly. But first, we need to
      // use `fetch()` to follow redirects so we get the correct final base URL
      const origin = await fetch(window.location.origin, { method: "HEAD" });
      result = new URL(url, origin.url);
    } else {
      // something unexpected happened
      throw err;
    }
  }
  // load and return the JSON
  const response = await fetch(result);
  const data = await response.json();
  return data;
}

// Populate the version switcher from the JSON data
function populateVersionSwitcher(data, versionSwitcherBtns) {
  const currentFilePath = getCurrentUrlPath();
  versionSwitcherBtns.forEach((btn) => {
    // Set empty strings by default so that these attributes exist and can be used in CSS selectors
    btn.dataset["activeVersionName"] = "";
    btn.dataset["activeVersion"] = "";
  });
  // in case there are multiple entries with the same version string, this helps us
  // decide which entry's `name` to put on the button itself. Without this, it would
  // always be the *last* version-matching entry; now it will be either the
  // version-matching entry that is also marked as `"preferred": true`, or if that
  // doesn't exist: the *first* version-matching entry.
  data = data.map((entry) => {
    // does this entry match the version that we're currently building/viewing?
    entry.match =
      entry.version == DOCUMENTATION_OPTIONS.theme_switcher_version_match;
    entry.preferred = entry.preferred || false;
    // if no custom name specified (e.g., "latest"), use version string
    if (!("name" in entry)) {
      entry.name = entry.version;
    }
    return entry;
  });
  const hasMatchingPreferredEntry = data
    .map((entry) => entry.preferred && entry.match)
    .some(Boolean);
  var foundMatch = false;
  // create links to the corresponding page in the other docs versions
  data.forEach((entry) => {
    // create the node
    const anchor = document.createElement("a");
    anchor.setAttribute(
      "class",
      "dropdown-item list-group-item list-group-item-action py-1",
    );
    anchor.setAttribute("href", `${entry.url}${currentFilePath}`);
    anchor.setAttribute("role", "option");
    const span = document.createElement("span");
    span.textContent = `${entry.name}`;
    anchor.appendChild(span);
    // Add dataset values for the version and name in case people want
    // to apply CSS styling based on this information.
    anchor.dataset["versionName"] = entry.name;
    anchor.dataset["version"] = entry.version;
    // replace dropdown button text with the preferred display name of the
    // currently-viewed version, rather than using sphinx's {{ version }} variable.
    // also highlight the dropdown entry for the currently-viewed version's entry
    let matchesAndIsPreferred = hasMatchingPreferredEntry && entry.preferred;
    let matchesAndIsFirst =
      !hasMatchingPreferredEntry && !foundMatch && entry.match;
    if (matchesAndIsPreferred || matchesAndIsFirst) {
      anchor.classList.add("active");
      versionSwitcherBtns.forEach((btn) => {
        btn.innerText = entry.name;
        btn.dataset["activeVersionName"] = entry.name;
        btn.dataset["activeVersion"] = entry.version;
      });
      foundMatch = true;
    }
    // There may be multiple version-switcher elements, e.g. one
    // in a slide-over panel displayed on smaller screens.
    document.querySelectorAll(".version-switcher__menu").forEach((menu) => {
      // we need to clone the node for each menu, but onclick attributes are not
      // preserved by `.cloneNode()` so we add onclick here after cloning.
      let node = anchor.cloneNode(true);
      node.onclick = checkPageExistsAndRedirect;
      // on click, AJAX calls will check if the linked page exists before
      // trying to redirect, and if not, will redirect to the homepage
      // for that version of the docs.
      menu.append(node);
    });
  });
}

/*******************************************************************************
 * Warning banner when viewing non-stable version of the docs.
 */

/**
 * Show a warning banner when viewing a non-stable version of the docs.
 *
 * adapted 2023-06 from https://mne.tools/versionwarning.js, which was
 * originally adapted 2020-05 from https://scikit-learn.org/versionwarning.js
 *
 * @param {Array} data The version data used to populate the switcher menu.
 */
function showVersionWarningBanner(data) {
  var version = DOCUMENTATION_OPTIONS.VERSION;
  // figure out what latest stable version is
  var preferredEntries = data.filter((entry) => entry.preferred);
  if (preferredEntries.length !== 1) {
    const howMany = preferredEntries.length == 0 ? "No" : "Multiple";
    console.log(
      `[PST] ${howMany} versions marked "preferred" found in versions JSON, ignoring.`,
    );
    return;
  }
  const preferredVersion = preferredEntries[0].version;
  const preferredURL = preferredEntries[0].url;
  // if already on preferred version, nothing to do
  const versionsAreComparable = validate(version) && validate(preferredVersion);
  if (versionsAreComparable && compare(version, preferredVersion, "=")) {
    console.log(
      "This is the prefered version of the docs, not showing the warning banner.",
    );
    return;
  }
  // check if banner has been dismissed recently
  const dismiss_date_str = JSON.parse(
    localStorage.getItem("pst_banner_pref") || "{}",
  )[version];
  if (dismiss_date_str != null) {
    const dismiss_date = new Date(dismiss_date_str);
    const now = new Date();
    const milliseconds_in_a_day = 24 * 60 * 60 * 1000;
    const days_passed = (now - dismiss_date) / milliseconds_in_a_day;
    const timeout_in_days = 14;
    if (days_passed < timeout_in_days) {
      console.info(
        `[PST] Suppressing version warning banner; was dismissed ${Math.floor(days_passed)} day(s) ago`,
      );
      return;
    }
  }

  // now construct the warning banner
  const banner = document.querySelector("#bd-header-version-warning");
  const middle = document.createElement("div");
  const inner = document.createElement("div");
  const bold = document.createElement("strong");
  const button = document.createElement("a");
  const close_btn = document.createElement("a");
  // these classes exist since pydata-sphinx-theme v0.10.0
  // the init class is used for animation
  middle.classList = "bd-header-announcement__content  ms-auto me-auto";
  inner.classList = "sidebar-message";
  button.classList =
    "btn text-wrap font-weight-bold ms-3 my-1 align-baseline pst-button-link-to-stable-version";
  button.href = `${preferredURL}${getCurrentUrlPath()}`;
  button.innerText = "Switch to stable version";
  button.onclick = checkPageExistsAndRedirect;
  close_btn.classList = "ms-3 my-1 align-baseline";
  const close_x = document.createElement("i");
  close_btn.append(close_x);
  close_x.classList = "fa-solid fa-xmark";
  close_btn.onclick = DismissBannerAndStorePref;
  // add the version-dependent text
  inner.innerText = "This is documentation for ";
  const isDev =
    version.includes("dev") ||
    version.includes("rc") ||
    version.includes("pre");
  const newerThanPreferred =
    versionsAreComparable && compare(version, preferredVersion, ">");
  if (isDev || newerThanPreferred) {
    bold.innerText = "an unstable development version";
  } else if (versionsAreComparable && compare(version, preferredVersion, "<")) {
    bold.innerText = `an old version (${version})`;
  } else if (!version) {
    bold.innerText = "an unknown version"; // e.g., an empty string
  } else {
    bold.innerText = `version ${version}`;
  }
  banner.appendChild(middle);
  banner.append(close_btn);
  middle.appendChild(inner);
  inner.appendChild(bold);
  inner.appendChild(document.createTextNode("."));
  inner.appendChild(button);
  banner.classList.remove("d-none");
}

/*******************************************************************************
 * MutationObserver to move the ReadTheDocs button
 */

/**
 * intercept the RTD flyout and place it in the rtd-footer-container if existing
 * if not it stays where on top of the page
 */
function initRTDObserver() {
  const mutatedCallback = (mutationList, observer) => {
    mutationList.forEach((mutation) => {
      // Check whether the mutation is for RTD, which will have a specific structure
      if (mutation.addedNodes.length === 0) {
        return;
      }
      if (mutation.addedNodes[0].data === undefined) {
        return;
      }
      if (mutation.addedNodes[0].data.search("Inserted RTD Footer") != -1) {
        mutation.addedNodes.forEach((node) => {
          document.getElementById("rtd-footer-container").append(node);
        });
      }
    });
  };

  const observer = new MutationObserver(mutatedCallback);
  const config = { childList: true };
  observer.observe(document.body, config);
}

async function fetchAndUseVersions() {
  // fetch the JSON version data (only once), then use it to populate the version
  // switcher and maybe show the version warning bar
  var versionSwitcherBtns = document.querySelectorAll(
    ".version-switcher__button",
  );
  const hasSwitcherMenu = versionSwitcherBtns.length > 0;
  const hasVersionsJSON = DOCUMENTATION_OPTIONS.hasOwnProperty(
    "theme_switcher_json_url",
  );
  const wantsWarningBanner = DOCUMENTATION_OPTIONS.show_version_warning_banner;

  if (hasVersionsJSON && (hasSwitcherMenu || wantsWarningBanner)) {
    const data = await fetchVersionSwitcherJSON(
      DOCUMENTATION_OPTIONS.theme_switcher_json_url,
    );
    // TODO: remove the `if(data)` once the `return null` is fixed within fetchVersionSwitcherJSON.
    // We don't really want the switcher and warning bar to silently not work.
    if (data) {
      populateVersionSwitcher(data, versionSwitcherBtns);
      if (wantsWarningBanner) {
        showVersionWarningBanner(data);
      }
    }
  }
}

/*******************************************************************************
 * Add keyboard functionality to mobile sidebars.
 *
 * Wire up the hamburger-style buttons using the click event which (on buttons)
 * handles both mouse clicks and the space and enter keys.
 */
function setupMobileSidebarKeyboardHandlers() {
  // These are hidden checkboxes at the top of the page whose :checked property
  // allows the mobile sidebars to be hidden or revealed via CSS.
  const primaryToggle = document.getElementById("pst-primary-sidebar-checkbox");
  const secondaryToggle = document.getElementById(
    "pst-secondary-sidebar-checkbox",
  );
  const primarySidebar = document.querySelector(".bd-sidebar-primary");
  const secondarySidebar = document.querySelector(".bd-sidebar-secondary");

  // Toggle buttons -
  //
  // These are the hamburger-style buttons in the header nav bar. When the user
  // clicks, the button transmits the click to the hidden checkboxes used by the
  // CSS to control whether the sidebar is open or closed.
  const primaryClickTransmitter = document.querySelector(".primary-toggle");
  const secondaryClickTransmitter = document.querySelector(".secondary-toggle");
  [
    [primaryClickTransmitter, primaryToggle, primarySidebar],
    [secondaryClickTransmitter, secondaryToggle, secondarySidebar],
  ].forEach(([clickTransmitter, toggle, sidebar]) => {
    if (!clickTransmitter) {
      return;
    }
    clickTransmitter.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggle.checked = !toggle.checked;

      // If we are opening the sidebar, move focus to the first focusable item
      // in the sidebar
      if (toggle.checked) {
        // Note: this selector is not exhaustive, and we may need to update it
        // in the future
        const tabStop = sidebar.querySelector("a, button");
        // use setTimeout because you cannot move focus synchronously during a
        // click in the handler for the click event
        setTimeout(() => tabStop.focus(), 100);
      }
    });
  });

  // Escape key -
  //
  // When sidebar is open, user should be able to press escape key to close the
  // sidebar.
  [
    [primarySidebar, primaryToggle, primaryClickTransmitter],
    [secondarySidebar, secondaryToggle, secondaryClickTransmitter],
  ].forEach(([sidebar, toggle, transmitter]) => {
    if (!sidebar) {
      return;
    }
    sidebar.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        toggle.checked = false;
        transmitter.focus();
      }
    });
  });

  // When the <label> overlay is clicked to close the sidebar, return focus to
  // the opener button in the nav bar.
  [
    [primaryToggle, primaryClickTransmitter],
    [secondaryToggle, secondaryClickTransmitter],
  ].forEach(([toggle, transmitter]) => {
    toggle.addEventListener("change", (event) => {
      if (!event.currentTarget.checked) {
        transmitter.focus();
      }
    });
  });
}

/**
 * When the page loads, or the window resizes, or descendant nodes are added or
 * removed from the main element, check all code blocks and Jupyter notebook
 * outputs, and for each one that has scrollable overflow, set tabIndex = 0.
 */
function addTabStopsToScrollableElements() {
  const updateTabStops = () => {
    document
      .querySelectorAll(
        "pre, " + // code blocks
          ".nboutput > .output_area, " + // NBSphinx notebook output
          ".cell_output > .output, " + // Myst-NB
          ".jp-RenderedHTMLCommon", // ipywidgets
      )
      .forEach((el) => {
        el.tabIndex =
          el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight
            ? 0
            : -1;
      });
  };
  const debouncedUpdateTabStops = debounce(updateTabStops, 300);

  // On window resize
  window.addEventListener("resize", debouncedUpdateTabStops);

  // The following MutationObserver is for ipywidgets, which take some time to
  // finish loading and rendering on the page (so even after the "load" event is
  // fired, they still have not finished rendering). Would be nice to replace
  // the MutationObserver if there is a way to hook into the ipywidgets code to
  // know when it is done.
  const mainObserver = new MutationObserver(debouncedUpdateTabStops);

  // On descendant nodes added/removed from main element
  mainObserver.observe(document.getElementById("main-content"), {
    subtree: true,
    childList: true,
  });

  // On page load (when this function gets called)
  updateTabStops();
}
function debounce(callback, wait) {
  let timeoutId = null;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      callback(...args);
    }, wait);
  };
}

/*******************************************************************************
 * Announcement banner - fetch and load remote HTML
 */
async function setupAnnouncementBanner() {
  const banner = document.querySelector(".bd-header-announcement");
  const { pstAnnouncementUrl } = banner ? banner.dataset : null;

  if (!pstAnnouncementUrl) {
    return;
  }

  try {
    const response = await fetch(pstAnnouncementUrl);
    if (!response.ok) {
      throw new Error(
        `[PST]: HTTP response status not ok: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.text();
    if (data.length === 0) {
      console.log(`[PST]: Empty announcement at: ${pstAnnouncementUrl}`);
      return;
    }
    banner.innerHTML = `<div class="bd-header-announcement__content">${data}</div>`;
    banner.classList.remove("d-none");
  } catch (_error) {
    console.log(`[PST]: Failed to load announcement at: ${pstAnnouncementUrl}`);
    console.error(_error);
  }
}

/*******************************************************************************
 * Reveal (and animate) the banners (version warning, announcement) together
 */
async function fetchRevealBannersTogether() {
  // Wait until finished fetching and loading banners
  await Promise.allSettled([fetchAndUseVersions(), setupAnnouncementBanner()]);

  // The revealer element should have CSS rules that set height to 0, overflow
  // to hidden, and an animation transition on the height (unless the user has
  // turned off animations)
  const revealer = document.querySelector(".pst-async-banner-revealer");
  if (!revealer) {
    return;
  }

  // Remove the d-none (display-none) class to calculate the children heights.
  revealer.classList.remove("d-none");

  // Add together the heights of the element's children
  const height = Array.from(revealer.children).reduce(
    (height, el) => height + el.offsetHeight,
    0,
  );

  // Use the calculated height to give the revealer a non-zero height (if
  // animations allowed, the height change will animate)
  revealer.style.setProperty("height", `${height}px`);

  // Wait for a bit more than 300ms (the transition duration), then set height
  // to auto so the banner can resize if the window is resized.
  setTimeout(() => {
    revealer.style.setProperty("height", "auto");
  }, 320);
}

/*******************************************************************************
 * Call functions after document loading.
 */

// This one first to kick off the network request for the version warning
// and announcement banner data as early as possible.
documentReady(fetchRevealBannersTogether);

documentReady(addModeListener);
documentReady(scrollToActive);
documentReady(addTOCInteractivity);
documentReady(setupSearchButtons);
documentReady(initRTDObserver);
documentReady(setupMobileSidebarKeyboardHandlers);

// Determining whether an element has scrollable content depends on stylesheets,
// so we're checking for the "load" event rather than "DOMContentLoaded"
if (document.readyState === "complete") {
  addTabStopsToScrollableElements();
} else {
  window.addEventListener("load", addTabStopsToScrollableElements);
}
