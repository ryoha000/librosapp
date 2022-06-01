cd /app/kissfft
make KISSFFT_STATIC=1 all
cp libkissfft-float.a ../libkissfft-float.a
cp kiss_fft.h ../kiss_fft.h
