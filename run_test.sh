#!/bin/bash

g++ -std=c++17 -o test_lib test.cpp libkissfft-float.a
./test_lib
