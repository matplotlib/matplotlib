from matplotlib.matlab import *
subplot(111, axisbg='y')
plot([1,2,3])
x = arange(0.0, 3.0, 0.1)
#plot(x, sin(2*pi*x))
grid(True)
xlabel(r'$\Delta_i$')
ylabel(r'$\Delta_{i+1}$')
tex = r'$\cal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\rm{sin}(2 \pi f x_i)$'
#tex = r'$\alpha\beta\gamma$'
text(1, 2.6, tex, fontsize=20)
title(r'$\Delta_i \rm{versus} \Delta_{i+1}$', fontsize=15)
savefig('mathtext_demo', dpi=100)
show()
