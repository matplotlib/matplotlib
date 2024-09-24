/*
 * A C++ Example
 * =============
 *
 * This tests parsing examples in languages other than Python
 */

// %%
// Pygments thinks preprocessor directives are a type of comment, which is fun. The
// following should *not* be merged into this formatted text block.
#include <vector>

int main(int argc, char** argv)
{
    // %%
    // It's likely that we may want to intersperse formatted text blocks
    // within the ``main`` method, where the contents are indented. We should
    // retain the current indentation level in the following code block.
    std::vector<double> y;
    for (int i = 0; i < 10; i++) {
        y.push_back(i * i);
    }

    /**************************/
    /* Here comes the end!    */
    /**************************/
    return 0;
}
