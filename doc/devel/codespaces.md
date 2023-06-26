# Contributing to Matplotlib using GitHub codespaces

You've discovered a bug or something else you want to change
in Matplotlib — excellent!

You've worked out a way to fix it — even better!

You want to tell us about it — best of all!

This project is a community effort, and everyone is welcome to contribute.
Everyone within the community is expected to abide by our
[Code of Conduct](../../CODE_OF_CONDUCT.md).

## GitHub codespaces contribution workflow

The preferred way to contribute to Matplotlib is to fork the main
repository at https://github.com/matplotlib/matplotlib, then submit a "pull
request" (PR). You can do this by cloning a copy of the Maplotlib repository to
your own computer, or alternatively using
[GitHub Codespaces](https://docs.github.com/codespaces) (a cloud-based
in-browser development environment, that comes with the appropriated setup to
contribute to Matplotlib).

A brief overview of the workflows is as follows.

1. Go to the GitHub web interface
2. Fork the [project repository](https://github.com/matplotlib/matplotlib):
3. Open codespaces on your fork by clicking on the green "Code" button on the
   GitHub web interface and selecting the "Codespaces" tab. Next, click on "Open
   codespaces on <your fork name>". You will be able to change branches later,
   so you can select the default `main` branch.

   After the codespace is created, you will be taken to a new browser tab where
   you can use the terminal to activate a pre-defined conda environment called
   `mpl-dev`:

   ```
   conda activate mpl-dev
   ```

4. Install the local version of Matplotlib with:

   ```
   python -m pip install -e .
   ```

   (See [Setting up Matplotlib for development](https://matplotlib.org/devdocs/devel/development_setup.html)
   for detailed instructions.)

5. Create a branch to hold your changes:

   ```
   git checkout -b my-feature origin/main
   ```

   and start making changes. Never work in the `main` branch!

6. Work on this task using Git to do the version control. Codespaces persist for
   some time (check the [documentation for details](https://docs.github.com/codespaces/getting-started/the-codespace-lifecycle))
   and can be managed on https://github.com/codespaces. When you're done editing
   e.g., `lib/matplotlib/collections.py`, do:

   ```
   git add lib/matplotlib/collections.py
   git commit
   ```

   to record your changes in Git, then push them to your GitHub fork with:

   ```
   git push -u origin my-feature
   ```

Finally, go to the web page of your fork of the Matplotlib repo, and click
'Pull request' to send your changes to the maintainers for review.

## Other stuff you may want to do

* If you need to run tests to verify your changes before sending your PR, you
  can run (in your `mpl-dev` conda environment):

  ```
  pytest
  ```

* If you need to build the documentation, you can move to the `doc` folder and
  run (in your `mpl-dev` conda environment):

  ```
  cd doc
  make html
  ```

* If you need to open a GUI window with Matplotlib output, our Codespaces
  configuration includes a
  [light-weight Fluxbox-based desktop](https://github.com/devcontainers/features/tree/main/src/desktop-lite).
  You can use it by connecting to this desktop via your web browser. To do this:

  1. Press `F1` or `Ctrl/Cmd+Shift+P` and select `Ports: Focus on Ports View` in
     VS Code to bring it into focus. Open the ports view in your tool, select
     the `noVNC` port, and click the Globe icon.
  2. In the browser that appears, click the Connect button and enter the desktop
     password (`vscode` by default).

  Check the
  [GitHub instructions](https://github.com/devcontainers/features/tree/main/src/desktop-lite#connecting-to-the-desktop)
  for more details on connecting to the desktop.
