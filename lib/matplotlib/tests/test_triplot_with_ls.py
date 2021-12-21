from matplotlib.testing.decorators import check_figures_equal


@check_figures_equal()
def test_triplot_with_ls(fig_test, fig_ref):
    fig_test.subplots().triplot([0, 2, 1], [0, 0, 1], [[0, 1, 2]], ls='--')
    fig_ref.subplots().triplot([0, 2, 1], [0, 0, 1], [
        [0, 1, 2]], linestyle='--')
