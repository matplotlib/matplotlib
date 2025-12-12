import matplotlib.pyplot as plt

plt.rcParams['ps.pathtext'] = True
fig, ax = plt.subplots()
<<<<<<< HEAD
ax.set_xlabel('Testing X')
ax.set_ylabel('Testing Y')
=======
>>>>>>> 9eb51af777 (Implemented pytest for new feature)
ax.text(0.25, 0.25, 'c')
ax.text(0.25, 0.5, 'a')
ax.text(0.25, 0.75, 'x')
fig.savefig(
    'C:/Users/kbele/Documents/matplotlib-dev/lib/matplotlib/tests/baseline_images/test_pathtext/text_as_path.eps',
    format='eps'
            )
plt.rcParams['ps.pathtext'] = False
