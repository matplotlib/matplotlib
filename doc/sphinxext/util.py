import sys


def matplotlib_reduced_latex_scraper(block, block_vars, gallery_conf,
                                     **kwargs):
    """
    Reduce srcset when creating a PDF.

    Because sphinx-gallery runs *very* early, we cannot modify this even in the
    earliest builder-inited signal. Thus we do it at scraping time.
    """
    from sphinx_gallery.scrapers import matplotlib_scraper

    if gallery_conf['builder_name'] == 'latex':
        gallery_conf['image_srcset'] = []
    return matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)


# Clear basic_units module to re-register with unit registry on import.
def clear_basic_units(gallery_conf, fname):
    return sys.modules.pop('basic_units', None)
