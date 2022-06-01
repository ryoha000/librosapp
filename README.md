# librosapp

## About
This repository is a transplant of librosa by C++.

The following functions are supported in this repository after end-to-end testing.
- librosa::stft
- librosa::filters::mel
- librosa::feature::melspectrogram
- librosa::feature::inverse::mel_to_stft


## Requirement
### 1. install submodules
`$ git submodule update --init`

### 2. build kissfft
This is done to build kissfft.  If you want to build by other methods, skip this step.  
**NOTE: If you are a Windows user, use an absolute path for `-v`option.**  
`$ docker run -v C:\workspace\librosapp:/app -it gcc:12.1 bash /app/build_kissfft.sh`

### 3. Run sample code
You can run the sample code by executing `$ ./run_test.sh` or `$ docker-compose up --build`

## Usage
### stft
```
    librosa::stft_arg stft_arg;
    stft_arg.y = audio;

    auto result = librosa::stft(&stft_arg);
```

### _spectgram
```
    librosa::core::spectrum::_spectrogram_arg spec_arg;
    spec_arg.y = audio;

    auto result = librosa::core::spectrum::_spectrogram(&spec_arg);
```

### filters::mel
```
    librosa::filters::mel_arg mel_arg;

    auto mel_basis = librosa::filters::mel(&mel_arg);
```

### melspectrogram
```
    librosa::feature::melspectrogram_arg melspec_arg;
    melspec_arg.y = audio;

    auto melspec = librosa::feature::melspectrogram(&melspec_arg);
```

### mel_to_stft
```
    librosa::feature::melspectrogram_arg melspec_arg;
    melspec_arg.y = audio;
    auto melspec = librosa::feature::melspectrogram(&melspec_arg);

    librosa::feature::inverse::mel_to_stft_arg arg;
    arg.M = melspec;
    arg.n_fft = melspec_arg.n_fft;
    arg.power = 2.0;
    arg.sr = melspec_arg.sr;

    auto spec = librosa::feature::inverse::mel_to_stft(&arg);
```

## related project
[librosa](https://github.com/librosa/librosa)  
[kissfft](https://github.com/mborgerding/kissfft)  
[Eigen](https://gitlab.com/libeigen/eigen)  
[LBFGSpp](https://github.com/yixuan/LBFGSpp)
