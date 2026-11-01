import js from "@eslint/js";
import globals from "globals";
import prettier from "eslint-config-prettier/flat";

export default [
  {
    ignores: ["jquery-ui-*/", "node_modules/**"],
  },
  js.configs.recommended,
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.jquery,
        IPython: "readonly",
        MozWebSocket: "readonly",
        mpl: "readonly",
        mpl_ondownload: "readonly",
      },
    },
    rules: {
      indent: ["error", 2, { SwitchCase: 1 }],
      "max-len": ["error", { code: 100 }],
      "no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          caughtErrors: "none",
        }
      ],
      quotes: ["error", "double", { avoidEscape: true }],
    },
  },
  {
    files: ["js/**/*.js"],
    rules: {
      indent: ["error", 4, { SwitchCase: 1 }],
      quotes: ["error", "single", { avoidEscape: true }],
    },
  },
  {
    files: ["js/mpl_pyodide.js"],
    rules: {
      "no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          caughtErrors: "none",
          args: "none",
        }
      ],
    },
  },
  prettier,
  {
    rules: {
      "comma-dangle": [
        "error",
        { functions: "never", objects: "always-multiline" }
      ],
    },
  }
];
