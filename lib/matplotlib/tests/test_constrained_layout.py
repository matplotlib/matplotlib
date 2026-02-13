import matplotlib.pyplot as plt


def test_text_outside_axes_saved(tmp_path):
    fig = plt.figure(layout='constrained')
    ax = fig.add_subplot()
    ax.text(0.0, -0.5, "My Text")
    save_path = tmp_path / "test.png"
    fig.savefig(save_path)
    assert save_path.exists()
