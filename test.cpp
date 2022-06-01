#include <float.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include "json.hpp"
#include "librosapp.hpp"

using json = nlohmann::json;

bool is_equal(float x, float y) { return fabs(x - y) < 1e-4; }

bool is_equal_matrix(vector<vector<float>> x, vector<vector<float>> y) {
  for (int i = 0; i < x.size(); i++) {
    for (int j = 0; j < x[0].size(); j++) {
      if (!is_equal(x[i][j], y[i][j])) {
        std::cerr << "x: " << x[i][j] << ", y: " << y[i][j] << ", i: " << i
                  << ", j: " << j << "\n";
        return false;
      }
    }
  }
  return true;
}

int main() {
  std::ifstream fin("test_data.json");
  json test_json;
  fin >> test_json;

  const auto audio = test_json["input"].get<vector<float>>();

  // stft test
  {
    auto start = std::chrono::system_clock::now();
    librosa::stft_arg stft_arg;
    stft_arg.y = audio;
    stft_arg.n_fft = 512;
    stft_arg.hop_length = 128;
    auto actual_stft_complex = librosa::stft(&stft_arg);
    auto actual_complex = librosa::convert_complex_matrix(actual_stft_complex);
    auto actual = vector<vector<float>>();
    for (int i = 0; i < actual_complex.size(); i++) {
      actual.push_back(vector<float>());
      for (int j = 0; j < actual_complex[i].size(); j++) {
        actual[i].push_back(actual_complex[i][j].real());
        actual[i].push_back(actual_complex[i][j].imag());
      }
    }
    auto expected = test_json["stft_512_128"].get<vector<vector<float>>>();
    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] stft test has passed" << std::endl;
    auto dur = std::chrono::system_clock::now() - start;
    auto msec =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << msec << " milli sec \n";
  }

  // _spectrum test
  {
    auto start = std::chrono::system_clock::now();
    librosa::core::spectrum::_spectrogram_arg spec_arg;
    spec_arg.y = audio;
    spec_arg.n_fft = 512;
    spec_arg.hop_length = 128;
    spec_arg.power = 1.0;
    auto actual_mat = librosa::core::spectrum::_spectrogram(&spec_arg);
    auto actual = librosa::convert_matrix(actual_mat);
    auto expected =
        test_json["_spectrum_512_128_1.0"].get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] _spectrum test has passed" << std::endl;
    auto dur = std::chrono::system_clock::now() - start;
    auto msec =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << msec << " milli sec \n";
  }

  // filters::mel test
  {
    auto start = std::chrono::system_clock::now();
    librosa::filters::mel_arg mel_arg;
    mel_arg.sr = 16000;
    mel_arg.n_fft = 512;
    mel_arg.n_mels = 60;

    auto actual_mat = librosa::filters::mel(&mel_arg);
    auto actual = librosa::convert_matrix(actual_mat);
    auto expected = test_json["mel_16000_512_60"].get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] filters::mel test has passed" << std::endl;
    auto dur = std::chrono::system_clock::now() - start;
    auto msec =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << msec << " milli sec \n";
  }

  // melspectrogram test
  {
    auto start = std::chrono::system_clock::now();
    librosa::feature::melspectrogram_arg melspec_arg;
    melspec_arg.y = audio;
    melspec_arg.sr = 16000;
    melspec_arg.n_fft = 512;
    melspec_arg.hop_length = 128;
    melspec_arg.n_mels = 60;

    auto actual_mat = librosa::feature::melspectrogram(&melspec_arg);
    auto actual = librosa::convert_matrix(actual_mat);
    auto expected = test_json["melspectrogram_16000_512_60_128_2.0"]
                        .get<vector<vector<float>>>();

    assert(actual.size() == expected.size());
    assert(actual[0].size() == expected[0].size());
    assert(is_equal_matrix(actual, expected));

    std::cout << "[SUCCESS] melspectrogram test has passed" << std::endl;
    auto dur = std::chrono::system_clock::now() - start;
    auto msec =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << msec << " milli sec \n";
  }

  return 0;
}
