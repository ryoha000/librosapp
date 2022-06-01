#pragma once

#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "eigen/Eigen/Core"
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/FFT"
#include "kiss_fft.h"

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::complex;
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

vector<vector<std::complex<float>>> convert_complex_matrix(
    Matrix<complex<float>, Dynamic, Dynamic> m) {
  auto res_vec = vector<vector<complex<float>>>(m.rows());
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      std::complex<float> a(m(i, j).real(), m(i, j).imag());
      res_vec[i].push_back(a);
    }
  }
  return res_vec;
}

vector<vector<float>> convert_matrix(Matrix<float, Dynamic, Dynamic> m) {
  auto res_vec = vector<vector<float>>(m.rows());
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      res_vec[i].push_back(m(i, j));
    }
  }
  return res_vec;
}

Matrix<complex<float>, Dynamic, Dynamic> stft(stft_arg* arg) {
  auto padded_len = arg->y.size() + arg->n_fft / 2 * 2;
  auto input = Matrix<float, 1, Dynamic>(padded_len);
  for (int i = 0; i < padded_len; i++) {
    if (i < arg->n_fft / 2 || i >= arg->y.size() + arg->n_fft / 2) {
      input(i) = 0.0;
    } else {
      input(i) = arg->y[i - arg->n_fft / 2];
    }
  }

  auto hanning = Matrix<float, 1, Dynamic>(1, arg->n_fft);
  for (int i = 0; i < arg->n_fft; i++) {
    auto n = 1 - arg->n_fft + 2 * i;
    hanning(i) =
        0.5 + 0.5 * cosf(LIBROSA_PI * float(n) / float(arg->n_fft - 1));
  }

  auto cols = (padded_len - arg->n_fft) / arg->hop_length + 1;
  auto result =
      Matrix<complex<float>, Dynamic, Dynamic>(cols, arg->n_fft / 2 + 1);
  auto fft_input = vector<float>(arg->n_fft);
  auto fft_result = vector<complex<float>>(arg->n_fft);
  for (int col = 0; col < cols; col++) {
    auto start = col * arg->hop_length;
    auto fft_input_m =
        input(0, Eigen::seqN(start, arg->n_fft)).cwiseProduct(hanning);

    for (int i = 0; i < arg->n_fft; i++) {
      fft_input[i] = fft_input_m(0, i);
    }

    Eigen::FFT<float> fft;
    fft.fwd(fft_result, fft_input);

    for (int j = 0; j < arg->n_fft / 2 + 1; j++) {
      result(col, j) = fft_result[j];
    }
  }

  return result.transpose();
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

VectorXd mel_to_hz(VectorXd mels, bool htk = false) {
  VectorXd freqs(mels.size());
  if (htk) {
    for (int i = 0; i < mels.size(); i++) {
      freqs(i) = 700.0 * (powf(10.0, mels(i) / 2595.0) - 1.0);
    }
    return freqs;
  }

  float f_min = 0.0;
  float f_sp = 200.0 / 3;

  freqs = f_sp * mels + VectorXd::Constant(mels.size(), f_min);

  float min_log_hz = 1000.0;
  float min_log_mel = (min_log_hz - f_min) / f_sp;
  float logstep = logf(6.4) / 27.0;

  for (int i = 0; i < mels.size(); i++) {
    if (mels(i) >= min_log_mel) {
      freqs(i) = min_log_hz * expf(logstep * (mels(i) - min_log_mel));
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

VectorXd mel_frequencies(mel_frequencies_arg* arg) {
  auto fmin_v = vector<float>(1, arg->fmin);
  auto fmax_v = vector<float>(1, arg->fmax);
  float min_mel = hz_to_mel(fmin_v, arg->htk)[0];
  float max_mel = hz_to_mel(fmax_v, arg->htk)[0];

  auto mels = VectorXd::LinSpaced(arg->n_mels, min_mel, max_mel);

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

Matrix<float, Dynamic, Dynamic> _spectrogram(_spectrogram_arg* arg) {
  stft_arg s_arg;
  s_arg.y = arg->y;
  s_arg.n_fft = arg->n_fft;
  s_arg.hop_length = arg->hop_length;

  auto stft_result = stft(&s_arg);
  if (stft_result.rows() == 0) {
    return Matrix<float, Dynamic, Dynamic>(0, 0);
  }

  auto a = stft_result.cwiseAbs();
  return a.pow(arg->power);
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

Matrix<float, Dynamic, Dynamic> mel(mel_arg* arg) {
  int length = 1 + arg->n_fft / 2;
  if (arg->fmax < 0.0) {
    arg->fmax = float(arg->sr) / 2.0;
  }

  Matrix<float, Dynamic, Dynamic> weights(arg->n_mels, length);

  auto fft_freqs = VectorXd::LinSpaced(length, 0.0, float(length - 1)) *
                   float(arg->sr) / float(arg->n_fft);

  librosa::core::convert::mel_frequencies_arg mel_freq_arg;
  mel_freq_arg.fmin = arg->fmin;
  mel_freq_arg.fmax = arg->fmax;
  mel_freq_arg.n_mels = arg->n_mels + 2;
  mel_freq_arg.htk = arg->htk;
  auto mel_f = librosa::core::convert::mel_frequencies(&mel_freq_arg);

  VectorXd fdiff(mel_f.size() - 1);
  for (int i = 0; i < fdiff.size(); i++) {
    fdiff(i) = mel_f(i + 1) - mel_f(i);
  }

  // TODO: 流石にどうにかできそう
  Matrix<float, Dynamic, Dynamic> ramps(mel_f.size(), fft_freqs.size());
  for (int i = 0; i < mel_f.size(); i++) {
    for (int j = 0; j < fft_freqs.size(); j++) {
      ramps(i, j) = mel_f(i) - fft_freqs(j);
    }
  }

  auto lower = VectorXd(fft_freqs.size());
  auto upper = VectorXd(fft_freqs.size());
  for (int i = 0; i < arg->n_mels; i++) {
    auto lower = -1.0 * ramps(i, Eigen::placeholders::all) / fdiff(i);
    // for (int j = 0; j < lower.size(); j++) {
    //   lower[j] = -1 * ramps[i][j] / fdiff[i];
    // }

    auto upper = ramps(i + 2, Eigen::placeholders::all) / fdiff(i + 1);
    // for (int j = 0; j < lower.size(); j++) {
    //   upper[j] = ramps[i + 2][j] / fdiff[i + 1];
    // }

    // weights[i] = np.maximum(0, np.minimum(lower, upper));
    weights(i, Eigen::placeholders::all) = lower.cwiseMin(upper).cwiseMax(0.0);
  }

  for (int i = 0; i < arg->n_mels; i++) {
    auto enorm = 2.0 / (mel_f(2 + i) - mel_f(i));
    weights(i, Eigen::placeholders::all) *= enorm;
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

Matrix<float, Dynamic, Dynamic> melspectrogram(melspectrogram_arg* arg) {
  librosa::core::spectrum::_spectrogram_arg spec_arg;
  spec_arg.y = arg->y;
  spec_arg.n_fft = arg->n_fft;
  spec_arg.hop_length = arg->hop_length;
  spec_arg.n_fft = arg->n_fft;
  spec_arg.power = arg->power;

  auto S = librosa::core::spectrum::_spectrogram(&spec_arg);
  if (S.rows() == 0) {
    return Matrix<float, Dynamic, Dynamic>(0, 0);
  }

  librosa::filters::mel_arg mel_arg;
  mel_arg.sr = arg->sr;
  mel_arg.n_fft = arg->n_fft;
  mel_arg.n_mels = arg->n_mels;
  mel_arg.htk = arg->htk;

  auto mel_basis = librosa::filters::mel(&mel_arg);

  // return np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
  Matrix<float, Dynamic, Dynamic> melspec(mel_basis.rows(), S.cols());
  for (int m = 0; m < mel_basis.rows(); m++) {
    for (int t = 0; t < S.cols(); t++) {
      float val = 0.0;
      for (int f = 0; f < S.rows(); f++) {
        val += S(f, t) * mel_basis(m, f);
      }
      melspec(m, t) = val;
    }
  }

  return melspec;
}
}  // namespace feature
}  // namespace librosa
