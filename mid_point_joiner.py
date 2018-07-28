import matplotlib.pyplot as plt

def Mid_joiner():
    i = int(input("Enter How many no you want: "))
    x1 = int(input("Enter x1: "))
    x2 = int(input("Enter x2: "))
    x3 = int(input("Enter x3: "))
    x4 = int(input("Enter x4: "))
    y1 = int(input("Enter y1: "))
    y2 = int(input("Enter y2: "))
    y3 = int(input("Enter y3: "))
    y4 = int(input("Enter y4: "))
    while i > 0:
        i += -1
        m1 = [((x1 + x2)/2), ((x2 + x3)/2), ((x3 + x4)/2), ((x4 + x1)/2), ((x1 + x2)/2)]
        m2 = [((y1 + y2)/2), ((y2 + y3)/2), ((y3 + y4)/2), ((y4 + y1)/2), ((y1 + y2)/2)]
        x1 = m1[0]
        x2 = m1[1]
        x3 = m1[2]
        x4 = m1[3]
        m1[4] = x1
        y1 = m2[0]
        y2 = m2[1]
        y3 = m2[2]
        y4 = m2[3]
        m2[4]  = y1
        plt.plot(m1, m2)
    plt.show()

print("Your mid Drawing is", Mid_joiner())
