#pragma once
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
#include <vector>

/**
* Fast discrete curvelet tranform implementation following the Matlab implementation
* See reference Matlab code available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
**/

namespace FDCT{

	//Returns forward transform coefficients of image 'in' into 'scale' number of resolution levels
	std::vector<std::vector<cv::Mat>> fdct_wrapping(cv::Mat &in, int scale);

	//Returns inverse transform image of coefficients in 'C' of size MxN
	cv::Mat ifdct_wrapping(std::vector<std::vector<cv::Mat>> &C, int M, int N);

	/**
	* Internal functions used during decomposition/reconstrution
	**/
	cv::Mat rot90(cv::Mat &x, int k);
	void meshgrid(int sx, int ex, int sy, int ey, cv::Mat &xx, cv::Mat &yy);
	void fdct_wrapping_window(std::vector<float> &x, std::vector<float> &wl, std::vector<float> &wr);
	void fdct_wrapping_window(cv::Mat &x, cv::Mat &wl, cv::Mat &wr);
	cv::Mat fftshift(cv::Mat &x);
	cv::Mat ifftshift(cv::Mat &x);
	cv::Mat circshift(cv::Mat &M, int x, int y);
}
