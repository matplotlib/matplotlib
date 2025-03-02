# Report for assignment 4

This is a template for your report. You are free to modify it as needed.
It is not required to use markdown for your report either, but the report
has to be delivered in a standard, cross-platform format.

## Project

Name: matplotlib

URL: https://github.com/matplotlib/matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations. Matplotlib can be used in Python scripts, Python/IPython shells, web application servers as well as various graphical user interface toolkits, which produces
high-quality figures for different types of input.

## Onboarding experience

Did you choose a new project or continue on the previous one?

For lab4, we chose a new project ([matplotlib](https://github.com/matplotlib/matplotlib)) instead of the previous one ([jsoniter](https://github.com/json-iterator/java))

If you changed the project, how did your experience differ from before?

## Effort spent

For each team member, how much time was spent in

1. plenary discussions/meetings;

2. discussions within parts of the group;

3. reading documentation;

4. configuration and setup;

5. analyzing code/output;

6. writing documentation;

7. writing code;

8. running code?

For setting up tools and libraries (step 4), enumerate all dependencies
you took care of and where you spent your time, if that time exceeds
30 minutes.

## Overview of issue(s) and work done.

Title: \[Bug\]: unaligned multiline text when using math mode

URL: https://github.com/matplotlib/matplotlib/issues/29527

Summary in one or two sentences

When rendering multiline text where one line includes math mode (e.g., `$...$`), while another does not, the lines are not perfectly aligned. Specifically, the line containing math mode text appears slightly misaligned compared to the non-math text line, leading to an unintended visual offset.

Scope (functionality and code affected).

**Functinality**: This issue affects the rendering of multiline text in Matplotlib, particularly when mixing regular text and math mode text. The misalignment causes inconsistent vertical positioning of the text lines.

**Code affected**: The issue is primarily related to the `matplotlib.text.Text` class (in `text.py`), particularly the `_get_layout()` method, which is responsible for computing the layout of multiline text. It also involves `MathTextParser.parse()` in `_mathtext.py`, where differences in baseline and height calculations between math mode and regular text lead to misalignment. Additionaly, it relates to how the `backend` works/is when matplotlib running on different platforms (e.g., `linux`, `macos` or `windows`). Specifically, `draw_mathtext()` and `draw_text()` (see in `backend_agg.py` as an example) directly affect it, and a tricky function is `_get_text_metrics_with_cache()` which is called in `_get_layout()` in `text.py`, since there are the **same characters** in the two lines in this issue.

**Work done**

- 

## Requirements for the new feature or requirements affected by functionality being refactored

Optional (point 3): trace tests to requirements.

## Code changes

### Patch

(copy your changes or the add git command to show them)

git diff ...

Optional (point 4): the patch is clean.

Optional (point 5): considered for acceptance (passes all automated checks).

## Test results

Overall results with link to a copy or excerpt of the logs (before/after
refactoring).

## UML class diagram and its description

### Key changes/classes affected

Optional (point 1): Architectural overview.

Optional (point 2): relation to design pattern(s).

## Overall experience

What are your main take-aways from this project? What did you learn?

How did you grow as a team, using the Essence standard to evaluate yourself?

Optional (point 6): How would you put your work in context with best software engineering practice?

Optional (point 7): Is there something special you want to mention here?
