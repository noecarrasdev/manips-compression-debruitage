import numpy as np
import scipy
import matplotlib.pyplot as plt
import pywt
import pywt.data
from scipy.fftpack import dct, idct

# imports image
f = pywt.data.camera()
n = f.shape[0]  # size of the image
print('image size : ', n)
plt.imshow(f, cmap='gray')
plt.axis('off')
plt.show()


# OK, image size 512, image displays (ajouter titre etc)

# functions
def thresh(fw, M):
    # sort a 1D copy of fw in descending order
    a = np.sort(np.ravel(abs(fw)))[::-1]
    T = a[M]
    return fw * (abs(fw) > T)


# measure of error
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# compression ratio
ratio = 1.5 / 100
M = int(ratio * n ** 2)


# DCT
def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=1, norm='ortho'), axis=0, norm='ortho')



# DCT compression
fd = dct2(f)
fd1 = thresh(fd, M)
f1 = idct2(fd1)


# WAVELET TRANSFORM
WT = 'db2'  # celebrated Daubechies Wavelet
# WT = 'haar' # produces a blocky approximation

# compute WT forms
fw, S = pywt.coeffs_to_array(pywt.wavedec2(f, WT))
fw1 = thresh(fw, M)
f2 = pywt.waverec2(pywt.array_to_coeffs(fw1, S, output_format='wavedec2'), WT)


# display
display = False
if display:
    plt.subplot(1, 2, 1)
    plt.imshow(fd1 != 0); plt.axis('off'); plt.title('DCT coefs')
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(f1,0,255), cmap='gray'); plt.axis('off')
    plt.title('Compres. PSNR=' + '{:.2f}'.format(psnr(f, f1)) + 'dB')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(fw1 != 0); plt.axis('off'); plt.title('Wavelets coefs')
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(f2,0,255), cmap='gray'); plt.axis('off')
    plt.title('Compres. PSNR=' + '{:.2f}'.format(psnr(f,f2)) + 'dB' )
    plt.show()

    plt.imshow(np.clip(f1,0,255), cmap='gray')
    plt.axis('off'); plt.title('DCT.compression')
    plt.show()

    plt.imshow(np.clip(f2,0,255), cmap='gray')
    plt.axis('off'); plt.title('Wav.compression')
    plt.show()


def compression(WT, N):
    f = pywt.data.camera()
    n = f.shape[0]  # size of the image
    plt.imshow(f, cmap='gray')
    plt.axis('off')
    plt.title('original image')
    plt.show()

    fw, S = pywt.coeffs_to_array(pywt.wavedec2(f, WT))

    for x in range(1, N):
        ratio = x / N
        M = int(ratio * n ** 2)

        fw1 = thresh(fw, M)
        f2 = pywt.waverec2(pywt.array_to_coeffs(fw1, S, output_format='wavedec2'), WT)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(fw1 != 0)
        plt.axis('off')
        plt.title(WT + ' wavelet with compression level : ' + str(ratio))
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(f2, 0, 255), cmap='gray')
        plt.axis('off')
        plt.title('Compres. PSNR=' + '{:.2f}'.format(psnr(f, f2)) + 'dB')
        plt.show()


compression('db2', 4)

print('code exited')