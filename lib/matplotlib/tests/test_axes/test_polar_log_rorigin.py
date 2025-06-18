import matplotlib.pyplot as plt

def test_polar_log_rorigin_rendering():
    fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    axs[0].set_title("Before Fix (log + rorigin)")
    axs[0].set_yscale("log")
    axs[0].set_rorigin(0.5)

    axs[1].set_title("After Fix (log + rorigin)")
    axs[1].set_yscale("log")
    axs[1].set_rorigin(0.5)

    fig.savefig("test_polar_log_rorigin.png")  # Optional if using image diff
