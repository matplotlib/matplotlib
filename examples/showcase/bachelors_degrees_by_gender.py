import matplotlib.pyplot as plt
import pandas as pd

# Activate the Tableau 20 styling. This call does most of the styling for you.
plt.style.use("tableau20")

gender_degree_data = pd.read_csv("http://files.figshare.com/1726892/percent_bachelors_degrees_women_usa.csv")

# You typically want your plot to be ~1.33x wider than tall. This plot is a rare
# exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
plt.figure(figsize=(12, 14))

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
plt.xlim(1970, 2010.1)
plt.ylim(-0.25, 90)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(range(1970, 2011, 10))
plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)])

# Now that the plot is prepared, it's time to actually plot the data!
# Note that I plotted the majors in order of the highest % in the final year.
majors = ['Health Professions', 'Public Administration', 'Education', 'Psychology',  
          'Foreign Languages', 'English', 'Communications\nand Journalism',  
          'Art and Performance', 'Biology', 'Agriculture',  
          'Social Sciences and History', 'Business', 'Math and Statistics',  
          'Architecture', 'Physical Sciences', 'Computer Science',  
          'Engineering']

y_offsets = {"Foreign Languages":0.5, "English":-0.5, "Communications\nand Journalism":0.75,
             "Art and Performance":-0.25, "Agriculture":1.25, "Social Sciences and History":0.25,
             "Business":-0.75, "Math and Statistics":0.75, "Architecture":-0.75,
             "Computer Science":0.75, "Engineering":-0.25}

for rank, column in enumerate(majors):
    # Plot each line separately with its own color, using the Tableau 20  
    # color set in order.
    line = plt.plot(gender_degree_data.Year.unique(),
                    gender_degree_data[column.replace("\n", " ")].values)
    
    line_color = line[0].get_color()
    
    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    y_pos = gender_degree_data[column.replace("\n", " ")].values[-1] - 0.5
    
    if column in y_offsets:
        y_pos += y_offsets[column]
      
    # Again, make sure that all labels are large enough to be easily read  
    # by the viewer.  
    plt.text(2010.5, y_pos, column, color=line_color)

# matplotlib's title() call centers the title on the plot, but not the graph,
# so I used the text() call to customize where the title goes.
  
# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.
  
# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
plt.text(1995, 92,"Percentage of Bachelor's degrees conferred to women in the "
             "U.S.A. by major (1970-2010)", fontsize=17, ha="center") 

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
plt.savefig("percent-bachelors-degrees-women-usa.png", bbox_inches="tight")
