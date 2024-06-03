/*
* Copyright (c) 2016, Gabor Adam Fodor <fogggab@yahoo.com>
* All rights reserved.
*
* License:
*
* This program is provided for scientific and educational purposed only.
* Feel free to use and/or modify it for such purposes, but you are kindly
* asked not to redistribute this or derivative works in source or executable
* form. A license must be obtained from the author of the code for any other use.
*
*/
#include <opencv2/opencv.hpp>

/**
* (Haar) Discrete Wavelet tranform implementation.
* Code provided by Andrey Smorodov
* See: http://stackoverflow.com/questions/20071854/wavelet-transform-in-opencv
**/

namespace HaarDWT{
	void cvHaarWavelet(cv::Mat &src, cv::Mat &dst, int NIter);
	void cvInvHaarWavelet(cv::Mat &src, cv::Mat &dst, int NIter, int SHRINKAGE_TYPE=0, float SHRINKAGE_T=50);
}

