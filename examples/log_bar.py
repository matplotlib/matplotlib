from matplotlib import matlab

data = ((3,1000), (10,3), (100,30), (500, 800), (50,1))

matlab.xlabel("FOO")
matlab.ylabel("FOO")
matlab.title("Testing")
matlab.gca().set_yscale('log')

dim = len(data[0])
w = 0.75
dimw = w / dim

x = matlab.arange(len(data))
for i in range(len(data[0])) :
    y = [d[i] for d in data]
    b = matlab.bar(x + i * dimw, y, dimw, bottom=0.001)
matlab.gca().set_xticks(x + w / 2)
matlab.gca().set_ylim( (0.001,1000))

matlab.show()


