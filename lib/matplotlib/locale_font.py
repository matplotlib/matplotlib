from locale import getdefaultlocale
from os.path import sep
from sys import stderr

from matplotlib import rcParams
from matplotlib.font_manager import fontManager


locale_font = {
    "zh_CN": ["SimSun", "SimHei", "Songti SC"]
}


def use_locale_font():
    lc = getdefaultlocale()[0]
    available_font_names = [f.name for f in fontManager.ttflist]

    if lc in locale_font.keys():
        expected_fonts = locale_font[lc]
        for f in expected_fonts:
            if f in available_font_names:
                rcParams["font.family"] = f
                break
        else:
            stderr.write(f"No fonts for {lc} was found. ")
    else:
        stderr.write(f"Locale {lc} is not configured\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    use_locale_font()
    plt.plot([1, 2, 3])
    plt.title("测试")
    plt.show()
