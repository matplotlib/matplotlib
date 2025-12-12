import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

<<<<<<< HEAD

=======
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
@image_comparison(baseline_images=['text_as_path.eps'])
def test_text_as_path_ps():
    plt.rcParams['ps.pathtext'] = True
    fig, ax = plt.subplots()
    ax.text(0.25, 0.25, 'c')
    ax.text(0.25, 0.5, 'a')
<<<<<<< HEAD
    ax.text(0.25, 0.75, 'x')
<<<<<<< HEAD
=======

    
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
=======
    ax.text(0.25, 0.75, 'x')
>>>>>>> 08be6cf794 (Removed whitespace)
