import sys, time, os
from helpers import rand_val, rand_point, rand_bbox, rand_transform
from matplotlib.numerix.mlab import rand


def report_memory(i):
    pid = os.getpid()
    if sys.platform=='sunos5':
        command = 'ps -p %d -o rss,osz' % pid
    else:
        'ps -p %d -o rss,sz' % pid
    a2 = os.popen(command).readlines()
    print i, '  ', a2[1],
    return int(a2[1].split()[1])


N = 200
for i in range(N):
    v1, v2, v3, v4, v5 = rand_val(5)
    b1 = v1 + v2
    b2 = v3 -v4
    b3 = v1*v2*b2 - b1


    p1 = rand_point()
    box1 = rand_bbox()
    t = rand_transform()
    N = 10000
    x, y = rand(N), rand(N)
    xt, yt = t.numerix_x_y(x, y)
    xys = t.seq_xy_tups( zip(x,y) )
    val = report_memory(i)
    if i==1: start = val

end = val
print 'Average memory consumed per loop: %1.4f\n' % ((end-start)/float(N))

