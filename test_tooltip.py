# # ## Check if your tooltip data is being stored
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Test if tooltip is being stored
# # x = np.linspace(0, 10, 5)
# # y = np.sin(x)

# # line, = plt.plot(x, y, 'o-', tooltip=['Point A', 'Point B', 'Point C', 'Point D', 'Point E'])

# # # Check if tooltip was stored
# # print("Tooltip stored:", line.get_tooltip())
# # print("Line has tooltip method:", hasattr(line, 'get_tooltip'))

# # # Not showing the plot yet, just testing the storage




# ## Test the backend
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np

# print("Backend:", matplotlib.get_backend())

# # Test with a simple plot and try hovering
# x = np.linspace(0, 10, 5)
# y = np.sin(x)

# fig, ax = plt.subplots()
# line, = ax.plot(x, y, 'o-', tooltip=['Point A', 'Point B', 'Point C', 'Point D', 'Point E'], picker=5)

# print("Tooltip stored:", line.get_tooltip())
# print("Figure has tooltip handling:", hasattr(fig, '_on_mouse_move_for_tooltip'))

# # Test if canvas has tooltip methods
# print("Canvas has show_tooltip:", hasattr(fig.canvas, 'show_tooltip'))
# print("Canvas has hide_tooltip:", hasattr(fig.canvas, 'hide_tooltip'))

# plt.title('Try hovering over the points')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Test tooltips with console output
# x = np.linspace(0, 10, 5)
# y = np.sin(x)

# plt.figure()
# plt.plot(x, y, 'o-', tooltip=['Point A', 'Point B', 'Point C', 'Point D', 'Point E'])
# plt.title('Move mouse over points - check console for tooltip messages')
# plt.show()