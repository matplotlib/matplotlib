import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a424f555d13ebadbb6089ecca939ea831ddd2d40

=======
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
=======

>>>>>>> 28390f61bd (Fixed coding style)
=======

<<<<<<< HEAD
>>>>>>> 36ebb70f4c (Added pytest for new feature, rebased commits)
=======

>>>>>>> 63af2cefba (Added text-as-path functionality to backend_ps.py)
=======
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
=======

>>>>>>> 28390f61bd (Fixed coding style)
=======
>>>>>>> 646cec28d9ed402921f628deedd345a58b588a9a
>>>>>>> a424f555d13ebadbb6089ecca939ea831ddd2d40
@image_comparison(baseline_images=['text_as_path.eps'])
def test_text_as_path_ps():
    plt.rcParams['ps.pathtext'] = True
    fig, ax = plt.subplots()
    ax.text(0.25, 0.25, 'c')
    ax.text(0.25, 0.5, 'a')
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a424f555d13ebadbb6089ecca939ea831ddd2d40
    ax.text(0.25, 0.75, 'x')
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a424f555d13ebadbb6089ecca939ea831ddd2d40
=======

    
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
=======
    ax.text(0.25, 0.75, 'x')
>>>>>>> 08be6cf794 (Removed whitespace)
=======
    ax.text(0.25, 0.75, 'x')
    
>>>>>>> 28390f61bd (Fixed coding style)
=======
  
>>>>>>> 4828055021 (Further fixed coding style)
=======
>>>>>>> 7a60bcb9c3 (Further fixed coding style. Sigh.)
=======
    ax.text(0.25, 0.75, 'x')
<<<<<<< HEAD
>>>>>>> 36ebb70f4c (Added pytest for new feature, rebased commits)
=======
    ax.text(0.25, 0.75, 'x')
>>>>>>> 63af2cefba (Added text-as-path functionality to backend_ps.py)
=======
    ax.text(0.25, 0.75, 'x')

    
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
=======
    ax.text(0.25, 0.75, 'x')
>>>>>>> 08be6cf794 (Removed whitespace)
=======
    ax.text(0.25, 0.75, 'x')
    
>>>>>>> 28390f61bd (Fixed coding style)
=======
  
>>>>>>> 4828055021 (Further fixed coding style)
=======
>>>>>>> 7a60bcb9c3 (Further fixed coding style. Sigh.)
=======
>>>>>>> 646cec28d9ed402921f628deedd345a58b588a9a
>>>>>>> a424f555d13ebadbb6089ecca939ea831ddd2d40
