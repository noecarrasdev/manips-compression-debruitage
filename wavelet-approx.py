from __future__ import division
from nt_toolbox.general import *
from nt_toolbox.signal import *
from numpy import *
import matplotlib.pyplot as plt
import pywt

# select filters
h0 = [0, .482962913145, .836516303738, .224143868042, -.129409522551]
norm = (sum([h0[i] ** 2 for i in range(len(h0))]))**(1/2)
h = [h0[i]/norm for i in range(len(h0))]
# here Db 4 ??
# TODO : change filter for better avec pywt

# compute high-pass filter from low pass
u = power(-ones(len(h) - 1), range(1, len(h)))  # alternate +1/-1
g = concatenate(([0], h[-1:0:-1] * u))
print(h)
print(g)

# image loading
'''
n = 256
name = 'nt_toolbox/data/flowers.png'
f = load_image(name, n)
'''
# pb dans les dependances avec skimage

# other image to load
f = pywt.data.camera()
n = f.shape[0]
print(n)

# boundaries for compression
Jmin = 0
Jmax = log2(n)-1
print(Jmax)

# compute Linear WT
fW = perform_wavortho_transf(f,Jmin,+1,h)
'''
eta = 4
fWLin = zeros((n,n))
fWLin[:n/eta:,:n/eta:] = fW[:n/eta:,:n/eta:] # WTF ? trouver comment passer ce bout de code MATLAB en python ?
fLin = perform_wavortho_transf(fWLin,Jmin,-1,h)

# display
elin = snr(f,fLin)
imageplot(clamp(fLin), 'Linear, SNR=' + str(elin), [1,2,2])
plt.show()
'''


# Non linear WT
T = .2
fWT = fW * (abs(fW)>T)
plt.subplot(1,2,1)
plot_wavelet(fW,Jmin)
plt.subplot(1,2,2)
plot_wavelet(fWT,Jmin)

plt.show()

# fin du jupyter pour la comparaison linéaire // non linéaire ?

print('code finished')

'''
# TODO : essayer cet exo? :
# Exercise 3: Display a 2-D wavelet by applying the backward transform to a Dirac (i.e. all zeros excepted a single 1 at a well-chosen position).


f_dirac = array([[0] * 512] *512)
f_dirac[0][0] = 1
print(f_dirac.shape)

f_dirac_back = perform_wavortho_transf(f_dirac,Jmin,-1,h)

plt.figure()
plot_wavelet(f_dirac_back,Jmin)
plt.show()
'''