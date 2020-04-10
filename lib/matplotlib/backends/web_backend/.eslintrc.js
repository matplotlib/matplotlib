module.exports = {
  root: true,
  ignorePatterns: ["jquery-ui-*/", "node_modules/"],
  env: {
    browser: true,
    jquery: true,
  },
  extends: ["eslint:recommended", "prettier"],
  globals: {
    IPython: "readonly",
    MozWebSocket: "readonly",
  },
  rules: {
    "no-unused-vars": [
      "error",
      {
        argsIgnorePattern: "^_",
      },
    ],
  },
};
