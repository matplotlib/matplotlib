import numpy as np
import pytest

from matplotlib import pyplot as plt


def test_thetalim_valid_invalid():
    ax = plt.subplot(projection='polar')
    ax.set_thetalim(0, 2 * np.pi)  # doesn't raise.
    ax.set_thetalim(thetamin=800, thetamax=440)  # doesn't raise.
    with pytest.raises(ValueError, match='The angle range must be <= 2 pi'):
        ax.set_thetalim(0, 3 * np.pi)
    with pytest.raises(ValueError,
                       match='The angle range must be <= 360 degrees'):
        ax.set_thetalim(thetamin=800, thetamax=400)
