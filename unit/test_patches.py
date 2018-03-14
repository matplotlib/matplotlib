import matplotlib.patches as mpatches

source_facecolor = (0.234, 0.123, 0.135, 0.322)
source_egdecolor = (0.728, 0.682, 0.945, 0.268)

def test_when_update_from_and_set_alpha_then_only_alpha_changes():
    # given
    source = mpatches.Rectangle((0, 0), 1., 1., facecolor=source_facecolor, edgecolor=source_egdecolor)
    updated = mpatches.Rectangle((1., 0), 1., 1., facecolor='pink', edgecolor="green")
    # when
    updated.update_from(source)
    # then
    assert updated.get_facecolor() == source_facecolor
    assert updated.get_edgecolor() == source_egdecolor
    # when
    updated.set_alpha(0.777)
    # then
    expected_facecolor = source_facecolor[0:3] + (0.777,)
    expected_edgecolor = source_egdecolor[0:3] + (0.777,)
    assert updated.get_facecolor() == expected_facecolor
    assert updated.get_edgecolor() == expected_edgecolor

