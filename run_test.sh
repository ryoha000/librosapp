#!/bin/bash

echo "[INFO] build test start"
g++ -std=c++17 -o test_lib test.cpp libkissfft-float.a -Ieigen -ILBFGSpp/include
echo "[INFO] build test end"
echo "[INFO] run test"
./test_lib
echo "[INFO] end test"
