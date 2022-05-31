#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "kiss_fft.h"

using std::vector;
using std::string;

constexpr float LIBROSA_PI = 3.14159265358979323846;

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

namespace librosa
{	
	vector<vector<kiss_fft_cpx>> stft(stft_arg* arg) {
		if (arg->hop_length <= 0)
		{
			arg->hop_length = arg->n_fft / 4;
		}

		auto padded_len = arg->y.size() + arg->n_fft / 2 * 2;
		auto cx_in = vector<kiss_fft_cpx>(padded_len);
		for (int i = 0; i < cx_in.size(); i++)
		{	
			cx_in[i] = kiss_fft_cpx();
			if (i < arg->n_fft / 2 || i >= arg->y.size() + arg->n_fft / 2)
			{
				cx_in[i].r = 0.0f;
				cx_in[i].i = 0.0f;
			}
			else
			{
				cx_in[i].r = arg->y[i - arg->n_fft / 2];
				cx_in[i].i = 0.0f;
			}
			cx_in[i];
		}

		auto hanning = vector<float>(arg->n_fft);
		for (int i = 0; i < arg->n_fft; i++)
		{
			auto n = 1 - arg->n_fft + 2 * i;
			hanning[i] = 0.5 + 0.5 * cosf(LIBROSA_PI * float(n) / float(arg->n_fft - 1));
		}

		kiss_fft_cfg cfg = kiss_fft_alloc(arg->n_fft, false, 0, 0);

		auto cols = (cx_in.size() - arg->n_fft) / arg->hop_length + 1;
		auto result = vector<vector<kiss_fft_cpx>>(cols);
		for (int i = 0; i < cols; i++)
		{
			auto fft_result = vector<kiss_fft_cpx>(arg->n_fft);
			auto start = i * arg->hop_length;
			kiss_fft(cfg, &cx_in[start], fft_result.data());
			result[i] = fft_result;
		}
		auto res = transpose(result);
		return res;
	}

	vector<vector<kiss_fft_cpx>> transpose(vector<vector<kiss_fft_cpx>> v)
	{
		vector<vector<kiss_fft_cpx>> res(0, vector<kiss_fft_cpx>());
		if (v.size() == 0) {
			vector<vector<kiss_fft_cpx>> res(0, vector<kiss_fft_cpx>());
			return res;
		}

		vector<vector<kiss_fft_cpx>> res(v[0].size(), vector<kiss_fft_cpx>());
		for (int i = 0; i < v.size(); i++)
		{
			for (int j = 0; j < v[i].size(); j++)
			{
				res[j].push_back(v[i][j]);
			}
		}

		return res;
	}
}
