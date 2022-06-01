#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "kiss_fft.h"

using std::string;
using std::vector;

constexpr float LIBROSA_PI = 3.14159265358979323846;

namespace librosa {
struct stft_arg {
  vector<float> y;
  int n_fft;
  int hop_length;
  /*int win_length;
  string window;
  bool center;
  string pad_mode;*/

  stft_arg() {
    n_fft = 2048;
    hop_length = -1;
    /*win_length = -1;
    window = "hann";
    center = false;
    pad_mode = "constant";*/
  }
};

vector<vector<kiss_fft_cpx>> transpose(vector<vector<kiss_fft_cpx>> v) {
  if (v.size() == 0) {
    vector<vector<kiss_fft_cpx>> res(0, vector<kiss_fft_cpx>());
    return res;
  }

  vector<vector<kiss_fft_cpx>> res(v[0].size(), vector<kiss_fft_cpx>());
  for (int i = 0; i < v.size(); i++) {
    for (int j = 0; j < v[i].size(); j++) {
      res[j].push_back(v[i][j]);
    }
  }

  return res;
}

vector<vector<kiss_fft_cpx>> stft(stft_arg* arg) {
  if (arg->hop_length <= 0) {
    arg->hop_length = arg->n_fft / 4;
  }

  auto padded_len = arg->y.size() + arg->n_fft / 2 * 2;
  auto cx_in = vector<kiss_fft_cpx>(padded_len);
  for (int i = 0; i < cx_in.size(); i++) {
    cx_in[i] = kiss_fft_cpx();
    if (i < arg->n_fft / 2 || i >= arg->y.size() + arg->n_fft / 2) {
      cx_in[i].r = 0.0f;
      cx_in[i].i = 0.0f;
    } else {
      cx_in[i].r = arg->y[i - arg->n_fft / 2];
      cx_in[i].i = 0.0f;
    }
  }

  auto hanning = vector<float>(arg->n_fft);
  for (int i = 0; i < arg->n_fft; i++) {
    auto n = 1 - arg->n_fft + 2 * i;
    hanning[i] =
        0.5 + 0.5 * cosf(LIBROSA_PI * float(n) / float(arg->n_fft - 1));
  }

  kiss_fft_cfg cfg = kiss_fft_alloc(arg->n_fft, false, 0, 0);

  auto cols = (cx_in.size() - arg->n_fft) / arg->hop_length + 1;
  auto result = vector<vector<kiss_fft_cpx>>(
      cols, vector<kiss_fft_cpx>(arg->n_fft / 2 + 1));
  auto fft_input = vector<kiss_fft_cpx>(arg->n_fft);
  auto fft_result = vector<kiss_fft_cpx>(arg->n_fft);
  for (int col = 0; col < cols; col++) {
    auto start = col * arg->hop_length;
    for (int i = 0; i < arg->n_fft; i++) {
      fft_input[i].r = cx_in[start + i].r * hanning[i];
      fft_input[i].i = 0.0f;
    }
    kiss_fft(cfg, fft_input.data(), fft_result.data());
    for (int j = 0; j < arg->n_fft / 2 + 1; j++) {
      result[col][j].r = fft_result[j].r;
      result[col][j].i = fft_result[j].i;
    }
  }
  auto res = librosa::transpose(result);

  kiss_fft_free(cfg);
  return res;
}

namespace core {
namespace convert {
vector<float> hz_to_mel(vector<float> freqs, bool htk = false) {
  vector<float> mels(freqs.size());
  if (htk) {
    for (int i = 0; i < mels.size(); i++) {
      mels[i] = 2595.0 * log10f(1.0 + freqs[i] / 700.0);
    }
    return mels;
  }

  float fmin = 0.0;
  float f_sp = 200.0 / 3;

  for (int i = 0; i < mels.size(); i++) {
    mels[i] = (freqs[i] - fmin) / f_sp;
  }

  float min_log_hz = 1000.0;
  float min_log_mel = (min_log_hz - fmin) / f_sp;
  float logstep = logf(6.4) / 27.0;

  for (int i = 0; i < mels.size(); i++) {
    if (freqs[i] >= min_log_hz) {
      mels[i] = min_log_mel + logf(freqs[i] / min_log_hz) / logstep;
    }
  }

  return mels;
}

vector<float> mel_to_hz(vector<float> mels, bool htk = false) {
  vector<float> freqs(mels.size());
  if (htk) {
    for (int i = 0; i < mels.size(); i++) {
      freqs[i] = 700.0 * (powf(10.0, mels[i] / 2595.0) - 1.0);
    }
    return freqs;
  }

  float f_min = 0.0;
  float f_sp = 200.0 / 3;

  for (int i = 0; i < mels.size(); i++) {
    freqs[i] = f_min + f_sp * mels[i];
  }

  float min_log_hz = 1000.0;
  float min_log_mel = (min_log_hz - f_min) / f_sp;
  float logstep = logf(6.4) / 27.0;

  for (int i = 0; i < mels.size(); i++) {
    if (mels[i] >= min_log_mel) {
      freqs[i] = min_log_hz * expf(logstep * (mels[i] - min_log_mel));
    }
  }

  return freqs;
}

struct mel_frequencies_arg {
  int n_mels;
  float fmin;
  float fmax;
  bool htk;

  mel_frequencies_arg() {
    n_mels = 128;
    fmin = 0.0;
    fmax = 11025.0;
    htk = false;
  }
};

vector<float> mel_frequencies(mel_frequencies_arg* arg) {
  auto fmin_v = vector<float>(1, arg->fmin);
  auto fmax_v = vector<float>(1, arg->fmax);
  float min_mel = hz_to_mel(fmin_v, arg->htk)[0];
  float max_mel = hz_to_mel(fmax_v, arg->htk)[0];

  auto step = (max_mel - min_mel) / float(arg->n_mels - 1);
  vector<float> mels = vector<float>(arg->n_mels);
  for (int i = 0; i < arg->n_mels; i++) {
    mels[i] = min_mel + step * float(i);
  }

  return mel_to_hz(mels, arg->htk);
}
}  // namespace convert

namespace spectrum {
struct _spectrogram_arg {
  vector<float> y;
  // vector<float> S;
  int n_fft;
  int hop_length;
  int power;
  /*int win_length;
  string window;
  bool center;
  string pad_mode;*/

  _spectrogram_arg() {
    n_fft = 2048;
    hop_length = 512;
    power = 1;
  }
};

vector<vector<float>> _spectrogram(_spectrogram_arg* arg) {
  stft_arg s_arg;
  s_arg.y = arg->y;
  s_arg.n_fft = arg->n_fft;
  s_arg.hop_length = arg->hop_length;

  auto stft_result = stft(&s_arg);
  if (stft_result.size() == 0) {
    return vector<vector<float>>();
  }

  auto spec = vector<vector<float>>(stft_result.size(),
                                    vector<float>(stft_result[0].size()));
  for (int i = 0; i < stft_result.size(); i++) {
    for (int j = 0; j < stft_result[i].size(); j++) {
      auto norm = sqrtf(powf(stft_result[i][j].i, 2.0) +
                        powf(stft_result[i][j].r, 2.0));

      spec[i][j] = powf(norm, arg->power);
    }
  }
  return spec;
}
}  // namespace spectrum
}  // namespace core

namespace filters {
struct mel_arg {
  int sr;
  int n_fft;
  int n_mels;
  float fmin;
  float fmax;
  bool htk;
  // string norm;
  // ?? dtype;

  mel_arg() {
    n_mels = 128;
    fmin = 0.0;
    fmax = -1.0;
    htk = false;
    /*norm = "slaney";
    dtype = np.float32;*/
  }
};

vector<vector<float>> mel(mel_arg* arg) {
  int length = 1 + arg->n_fft / 2;
  if (arg->fmax < 0.0) {
    arg->fmax = float(arg->sr) / 2.0;
  }

  vector<vector<float>> weights(arg->n_mels, vector<float>(length));

  vector<float> fft_freqs(length);
  for (int i = 0; i < length; i++) {
    fft_freqs[i] = float(arg->sr) / float(arg->n_fft) * float(i);
  }

  librosa::core::convert::mel_frequencies_arg mel_freq_arg;
  mel_freq_arg.fmin = arg->fmin;
  mel_freq_arg.fmax = arg->fmax;
  mel_freq_arg.n_mels = arg->n_mels + 2;
  mel_freq_arg.htk = arg->htk;
  auto mel_f = librosa::core::convert::mel_frequencies(&mel_freq_arg);

  vector<float> fdiff(mel_f.size() - 1);
  for (int i = 0; i < fdiff.size(); i++) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  vector<vector<float>> ramps(mel_f.size(), vector<float>(fft_freqs.size()));
  for (int i = 0; i < mel_f.size(); i++) {
    for (int j = 0; j < fft_freqs.size(); j++) {
      ramps[i][j] = mel_f[i] - fft_freqs[j];
    }
  }

  auto lower = vector<float>(fft_freqs.size());
  auto upper = vector<float>(fft_freqs.size());
  for (int i = 0; i < arg->n_mels; i++) {
    for (int j = 0; j < lower.size(); j++) {
      lower[j] = -1 * ramps[i][j] / fdiff[i];
    }

    for (int j = 0; j < lower.size(); j++) {
      upper[j] = ramps[i + 2][j] / fdiff[i + 1];
    }

    // weights[i] = np.maximum(0, np.minimum(lower, upper));
    for (int j = 0; j < lower.size(); j++) {
      auto lower_upper_minimum = 0.0;
      if (lower[j] > upper[j]) {
        lower_upper_minimum = upper[j];
      } else {
        lower_upper_minimum = lower[j];
      }

      if (lower_upper_minimum > 0.0) {
        weights[i][j] = lower_upper_minimum;
      } else {
        weights[i][j] = 0.0;
      }
    }
  }

  for (int i = 0; i < arg->n_mels; i++) {
    auto enorm = 2.0 / (mel_f[2 + i] - mel_f[i]);
    for (int j = 0; j < length; j++) {
      weights[i][j] = enorm * weights[i][j];
    }
  }

  return weights;
}
}  // namespace filters

namespace feature {
struct melspectrogram_arg {
  vector<float> y;
  int sr;
  // vector<vector<float>> S;
  int n_fft;
  int n_mels;
  int hop_length;
  float power;
  /*int win_length;
  string window;
  bool center;
  string pad_mode;*/
  bool htk;

  melspectrogram_arg() {
    sr = 22050;
    n_fft = 2048;
    n_mels = 128;
    hop_length = 512;
    power = 2.0;
    htk = false;
  }
};

vector<vector<float>> melspectrogram(melspectrogram_arg* arg) {
  librosa::core::spectrum::_spectrogram_arg spec_arg;
  spec_arg.y = arg->y;
  spec_arg.n_fft = arg->n_fft;
  spec_arg.hop_length = arg->hop_length;
  spec_arg.n_fft = arg->n_fft;
  spec_arg.power = arg->power;

  auto S = librosa::core::spectrum::_spectrogram(&spec_arg);
  if (S.size() == 0) {
    return vector<vector<float>>();
  }

  librosa::filters::mel_arg mel_arg;
  mel_arg.sr = arg->sr;
  mel_arg.n_fft = arg->n_fft;
  mel_arg.n_mels = arg->n_mels;
  mel_arg.htk = arg->htk;

  auto mel_basis = librosa::filters::mel(&mel_arg);

  // return np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
  auto melspec =
      vector<vector<float>>(mel_basis.size(), vector<float>(S[0].size()));
  for (int m = 0; m < mel_basis.size(); m++) {
    for (int t = 0; t < S[0].size(); t++) {
      float val = 0.0;
      for (int f = 0; f < S.size(); f++) {
        val += S[f][t] * mel_basis[m][f];
      }
      melspec[m][t] = val;
    }
  }

  return melspec;
}
}  // namespace feature
}  // namespace librosa
