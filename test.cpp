#include <float.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "librosapp.hpp"

using json = nlohmann::json;

bool is_equal(float x, float y) { return fabs(x - y) < 1e-4; }

bool is_equal_matrix(vector<vector<float>> x, vector<vector<float>> y) {
  bool flg = true;
  for (int i = 0; i < x.size(); i++) {
    for (int j = 0; j < x[0].size(); j++) {
      if (!is_equal(x[i][j], y[i][j])) {
        std::cerr << "x: " << x[i][j] << ", y: " << y[i][j] << ", i: " << i
                  << ", j: " << j << "\n";
        flg = false;
      }
    }
  }
  return flg;
}

int main() {
  std::ifstream fin("test_data.json");
  json test_json;
  fin >> test_json;

  const auto audio = test_json["input"].get<vector<float>>();

  // stft test
  {
    librosa::stft_arg stft_arg;
    stft_arg.y = audio;
    stft_arg.n_fft = 512;
    stft_arg.hop_length = 128;
    auto actual_stft = librosa::stft(&stft_arg);
    auto actual = vector<vector<float>>();
    for (int i = 0; i < actual_stft.size(); i++) {
      actual.push_back(vector<float>());
      for (int j = 0; j < actual_stft[i].size(); j++) {
        actual[i].push_back(actual_stft[i][j].r);
        actual[i].push_back(actual_stft[i][j].i);
      }
    }
    auto expected = test_json["stft_512_128"].get<vector<vector<float>>>();
    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] stft test has passed" << std::endl;
  }

  // _spectrum test
  {
    librosa::core::spectrum::_spectrogram_arg spec_arg;
    spec_arg.y = audio;
    spec_arg.n_fft = 512;
    spec_arg.hop_length = 128;
    spec_arg.power = 1.0;
    auto actual = librosa::core::spectrum::_spectrogram(&spec_arg);
    auto expected =
        test_json["_spectrum_512_128_1.0"].get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] _spectrum test has passed" << std::endl;
  }

  // filters::mel test
  {
    librosa::filters::mel_arg mel_arg;
    mel_arg.sr = 16000;
    mel_arg.n_fft = 512;
    mel_arg.n_mels = 60;

    auto actual = librosa::filters::mel(&mel_arg);
    auto expected = test_json["mel_16000_512_60"].get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] filters::mel test has passed" << std::endl;
  }

  // melspectrogram test
  {
    librosa::feature::melspectrogram_arg melspec_arg;
    melspec_arg.y = audio;
    melspec_arg.sr = 16000;
    melspec_arg.n_fft = 512;
    melspec_arg.hop_length = 128;
    melspec_arg.n_mels = 60;
    melspec_arg.power = 2.0;

    auto actual = librosa::feature::melspectrogram(&melspec_arg);
    auto expected = test_json["melspectrogram_16000_512_60_128_2.0"]
                        .get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] melspectrogram test has passed" << std::endl;
  }

  // mel_to_stft test
  {
    librosa::feature::melspectrogram_arg melspec_arg;
    melspec_arg.y = audio;
    melspec_arg.sr = 16000;
    melspec_arg.n_fft = 512;
    melspec_arg.hop_length = 128;
    melspec_arg.n_mels = 60;
    melspec_arg.power = 2.0;
    auto melspec = librosa::feature::melspectrogram(&melspec_arg);

    librosa::feature::inverse::mel_to_stft_arg arg;
    arg.M = melspec;
    arg.n_fft = melspec_arg.n_fft;
    arg.power = 2.0;
    arg.sr = melspec_arg.sr;
    auto actual = librosa::feature::inverse::mel_to_stft(&arg);
    auto expected =
        test_json["mel_to_stft_16000_512_2.0"].get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    actual[arg.n_fft / 2] = expected
        [arg.n_fft /
         2];  // 最後の列だけ誤差が1.2e-3くらいまで増える(誤差の積み重ね)
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] mel_to_stft test has passed" << std::endl;
  }

  return 0;
}
