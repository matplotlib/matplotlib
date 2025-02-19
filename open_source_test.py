import matplotlib.pyplot as plt
I = plt.imread("notre-dame.jpg")
m,n,p = I.shape #m=largeur, n=hauteur, p=uplet
J=[[[0 for d in range(p)] for e in range(n)] for f in range(m)]
K=[[[0 for d in range(p)] for e in range(n)] for f in range(m)]
L=[[[0 for d in range(p)] for e in range(n)] for f in range(m)]
for i in range(m) :
    for j in range(n) :
        J[i][j]=[int(I[i][j][0]),0,0]
        K[i][j]=[0,int(I[i][j][1]),0]
        L[i][j]=[0,0,int(I[i][j][2])]
fig, ax=plt.subplots(4,figsize=(28,10))
ax[0].imshow(I)
ax[1].imshow(J)
ax[2].imshow(K)
ax[3].imshow(L)
ax[0].set_axis_off()
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[3].set_axis_off()
ax[0].set_title('Image complete')
ax[1].set_title('Composante rouge')
ax[2].set_title('Composante bleue')
ax[3].set_title('Composante verte')
fig.suptitle("Image de Notre-Dame", x=0.51,y=.93)
plt.imsave("Notre-Dame_rouge.png",J) #The problem is here!
plt.savefig("Notre-Dame_composante_ex1.png",dpi=750,bbox_inches="tight")
plt.savefig("Notre-Dame_composante_ex1.pdf",dpi=750,bbox_inches="tight")
plt.show()
