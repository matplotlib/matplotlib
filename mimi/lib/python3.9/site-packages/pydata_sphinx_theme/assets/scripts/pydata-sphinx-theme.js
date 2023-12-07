// Define the custom behavior of the page
import { documentReady } from "./mixin";

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
    10
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
        "div:not(.search-button__search-container) > form.bd-search"
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
      if ((event.ctrlKey || event.metaKey) && event.code == "KeyK") {
        event.preventDefault();
        toggleSearchField();
      }
      // also allow Escape key to hide (but not show) the dynamic search field
      else if (document.activeElement === input && event.code == "Escape") {
        toggleSearchField();
      }
    },
    true
  );
};

/**
 * Change the search hint to `meta key` if we are a Mac
 */
var changeSearchShortcutKey = () => {
  let forms = document.querySelectorAll("form.bd-search");
  var isMac = window.navigator.platform.toUpperCase().indexOf("MAC") >= 0;
  if (isMac) {
    forms.forEach(
      (f) => (f.querySelector("kbd.kbd-shortcut__modifier").innerText = "⌘")
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
 * Check if corresponding page path exists in other version of docs
 * and, if so, go there instead of the homepage of the other docs version
 *
 * @param {event} event the event that trigger the check
 */
function checkPageExistsAndRedirect(event) {
  const currentFilePath = `${DOCUMENTATION_OPTIONS.pagename}.html`,
    tryUrl = event.target.getAttribute("href");
  let otherDocsHomepage = tryUrl.replace(currentFilePath, "");

  fetch(tryUrl, { method: "HEAD" })
    .then(() => {
      location.href = tryUrl;
    }) // if the page exists, go there
    .catch((error) => {
      location.href = otherDocsHomepage;
    });

  // this prevents the browser from following the href of the clicked node
  // (which is fine because this function takes care of redirecting)
  return false;
}

// Populate the version switcher from the JSON config file
var themeSwitchBtns = document.querySelectorAll(".version-switcher__button");
if (themeSwitchBtns.length) {
  fetch(DOCUMENTATION_OPTIONS.theme_switcher_json_url)
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      const currentFilePath = `${DOCUMENTATION_OPTIONS.pagename}.html`;
      themeSwitchBtns.forEach((btn) => {
        // Set empty strings by default so that these attributes exist and can be used in CSS selectors
        btn.dataset["activeVersionName"] = "";
        btn.dataset["activeVersion"] = "";
      });
      // create links to the corresponding page in the other docs versions
      data.forEach((entry) => {
        // if no custom name specified (e.g., "latest"), use version string
        if (!("name" in entry)) {
          entry.name = entry.version;
        }
        // create the node
        const span = document.createElement("span");
        span.textContent = `${entry.name}`;

        const node = document.createElement("a");
        node.setAttribute(
          "class",
          "list-group-item list-group-item-action py-1"
        );
        node.setAttribute("href", `${entry.url}${currentFilePath}`);
        node.appendChild(span);

        // on click, AJAX calls will check if the linked page exists before
        // trying to redirect, and if not, will redirect to the homepage
        // for that version of the docs.
        node.onclick = checkPageExistsAndRedirect;
        // Add dataset values for the version and name in case people want
        // to apply CSS styling based on this information.
        node.dataset["versionName"] = entry.name;
        node.dataset["version"] = entry.version;

        document.querySelector(".version-switcher__menu").append(node);
        // replace dropdown button text with the preferred display name of
        // this version, rather than using sphinx's {{ version }} variable.
        // also highlight the dropdown entry for the currently-viewed
        // version's entry
        if (
          entry.version ==
          "DOCUMENTATION_OPTIONS.version_switcher_version_match"
        ) {
          node.classList.add("active");
          themeSwitchBtns.forEach((btn) => {
            btn.innerText = btn.dataset["activeVersionName"] = entry.name;
            btn.dataset["activeVersion"] = entry.version;
          });
        }
      });
    });
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

/*******************************************************************************
 * Call functions after document loading.
 */

documentReady(addModeListener);
documentReady(scrollToActive);
documentReady(addTOCInteractivity);
documentReady(setupSearchButtons);
documentReady(initRTDObserver);
