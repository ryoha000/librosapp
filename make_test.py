import json
import librosa
import numpy as np

SAMPLE_RATE = 16000


def hz_to_mel(f: float, htk: bool = False):
    freq = np.asanyarray(f)

    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    # TODO: 多分このfreq.ndimは0
    if freq.ndim:
        # If we have array data, vectorize
        log_t = freq >= min_log_hz
        mels[log_t] = min_log_mel + np.log(freq[log_t] / min_log_hz) / logstep
    elif freq >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(freq / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, *, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * \
            np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def original_stft(audio, n_fft=512, hop_length=None):
    if hop_length is None:
        hop_length = int(n_fft // 4)
    # NOTE: np.pad の挙動
    # dummy = np.zeros(audio.shape[0] + int(n_fft // 2 * 2))
    # for i in range(dummy.shape[0]):
    #     if (i < n_fft // 2) or (i >= audio.shape[0] + n_fft // 2):
    #         dummy[i] = 0.0
    #     else:
    #         dummy[i] = audio[i - n_fft // 2]
    # print('allclose np.pad: ', np.allclose(dummy, audio)) # NOTE: 下にもっていくとTrue
    # NOTE: 消す
    audio = np.pad(audio, (int(n_fft//2), int(n_fft//2)), mode='constant')
    window = np.hanning(n_fft)
    # NOTE: np.hanning
    # dummy = np.zeros((n_fft,))
    # for i in range(n_fft):
    #     n = 1 - n_fft + 2 * i
    #     dummy[i] = 0.5 + 0.5 * np.cos(np.pi * float(n) / float(n_fft - 1))
    # print('allclose np.hanning: ', np.allclose(dummy, window)) # True
    # NOTE: np.hanning
    res = []
    cols = int((audio.shape[0] - n_fft) // hop_length) + 1
    for col in range(cols):
        start = col * hop_length
        frames = audio[start:start+n_fft] * window
        res.append(np.fft.fft(frames)[:n_fft//2+1])
    return np.array(res).T


def original_spectrum(audio, n_fft=512, hop_length=None, power=1.0):
    if hop_length is None:
        hop_length = int(n_fft // 4)
    return np.abs(original_stft(audio, n_fft=n_fft, hop_length=hop_length)) ** power


def original_melfilter(sr, n_fft, n_mels):
    length = int(1+n_fft//2)
    fmin = 0.0
    fmax = float(sr) / 2
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, length))

    fft_freqs = np.zeros((length, ))
    for i in range(length):
        fft_freqs[i] = sr / n_fft * i

    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=False)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)

    # NOTE: np.diff
    # dummy = np.zeros((mel_f.shape[0] - 1,))
    # for i in range(mel_f.shape[0] - 1):
    #     dummy[i] = mel_f[i + 1] - mel_f[i]
    # print('allclose np.diff: ', np.allclose(dummy, fdiff))
    # NOTE: np.diff

    # NOTE: np.subtract.outer の挙動
    # orig_ramps = np.zeros((mel_f.shape[0], fft_freqs.shape[0]))
    # for i in range(mel_f.shape[0]):
    #     for j in range(fft_freqs.shape[0]):
    #         orig_ramps[i][j] = mel_f[i] - fft_freqs[j]
    # print('isclose', np.allclose(ramps, orig_ramps)) # NOTE: True
    # NOTE: 消す

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    # NOTE: np.linespace
    # dummy = np.zeros((n_mels,))
    # step = (max_mel - min_mel) / float(n_mels - 1)
    # for i in range(n_mels):
    #     dummy[i] = min_mel + step * float(i)
    # print('mel_frequencies np.linespace: ', np.allclose(dummy, mels))
    # NOTE: np.linespace

    return mel_to_hz(mels, htk=htk)


def original_melspec(audio, n_fft=512, n_mels=60, power=2):
    S = original_spectrum(audio, n_fft=n_fft, power=power)
    mel_basis = original_melfilter(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_mels)
    print('S shape: ', S.shape, ', mel_basis shape: ', mel_basis.shape)
    res = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)

    # NOTE: np.einsumの挙動
    # dummy = np.zeros((mel_basis.shape[0], S.shape[1]))
    # for m in range(mel_basis.shape[0]):
    #     for t in range(S.shape[1]):
    #         val = 0.0
    #         for f in range(S.shape[0]):
    #             val += S[f][t] * mel_basis[m][f]
    #         dummy[m][t] = val
    # print('allclose einsum: ', np.allclose(res, dummy)) # True
    # NOTE: np.einsumの挙動
    return res


if __name__ == '__main__':
    audio, sr = librosa.load('sample.wav', sr=SAMPLE_RATE, mono=True)
    assert sr == SAMPLE_RATE
    audio = audio[:512*10]

    with open('test_data.json', mode='w', encoding='utf-8') as f:
        dic = dict()
        dic['input'] = audio.tolist()
        dic['_spectrum_512_128_1.0'] = original_spectrum(
            audio, n_fft=512, hop_length=128, power=1.0).tolist()
        actual = original_stft(audio, n_fft=512, hop_length=128)
        result = []
        for i in range(actual.shape[0]):
            result.append([])
            for j in range(actual.shape[1]):
                result[i].append(actual[i][j].real)
                result[i].append(actual[i][j].imag)
        dic['stft_512_128'] = result
        dic['mel_16000_512_60'] = original_melfilter(
            sr=SAMPLE_RATE, n_fft=512, n_mels=60).tolist()
        dic['melspectrogram_16000_512_60_128_2.0'] = original_melspec(
            audio, n_fft=512, n_mels=60).tolist()
        f.write(json.dumps(dic))
