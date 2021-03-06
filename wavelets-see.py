import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import pywt
import pywt.data

# select the Wavelet
WT = 'db1'
# list_wt = pywt.wavelist('db')
w = pywt.Wavelet(WT)

# different filters
(phi_d, psi_d, x) = w.wavefun(level=3)
(h_d, g_d, h_r, g_r) = (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)

X = range(len(h_d))
print('dec lo filter : ', h_d)

# Print
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(X, h_d, 'ro')
plt.title(WT + ' : mirror filter low pass')

plt.subplot(2, 2, 2)
plt.plot(X, g_d, 'ro')
plt.title(WT + ' : mirror filter high pass')

plt.subplot(2, 2, 3)
plt.plot(x, phi_d)
plt.title(WT + ' : scaling function')

plt.subplot(2, 2, 4)
plt.plot(x, psi_d)
plt.title(WT + ' : wavelet function')

plt.show()


# vanishing moments

print(WT, ' scaling function has ', w.vanishing_moments_phi, ' vanishing moments.')
print(WT, ' wavelet function has ', w.vanishing_moments_psi, ' vanishing moments.')

# test sizes

db = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']


def vanish(wt):
    h = wt.dec_lo
    van = wt.vanishing_moments_psi
    return (len(h), van)


def test_family(db):
    for WT in db:
        w = pywt.Wavelet(WT)
        (l_h, l_v) = vanish(w)
        if 2 * l_v != l_h:
            return 1
    return 0


#print(test_family(db))


'''
# filtres lo pass

h_fft = fft(h_d)
h_m_fft = [abs(h_fft[i]) for i in range(len(h_fft)//2)]
xft = range(len(h_m_fft))
plt.figure()
plt.plot(xft, h_m_fft)
plt.title(WT + ' mirror conjugate lowpass filter')

plt.show()
'''