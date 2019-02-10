from matplotlib.sankey import Sankey


def test_sankey():
    # lets just create a sankey instance and check the code runs
    sankey = Sankey()
    sankey.add()


def test_label():
    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1])
    assert s.diagrams[0].texts[0].get_text() == 'First\n0.25'
