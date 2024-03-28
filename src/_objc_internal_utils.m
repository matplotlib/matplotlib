#include <Cocoa/Cocoa.h>

int
_macos_display_is_valid(void)
{
    NSApplicationLoad();
    NSScreen *main = [NSScreen mainScreen];
    if (main != nil) {
        return 1;
    } else {
        return 0;
    }
}
