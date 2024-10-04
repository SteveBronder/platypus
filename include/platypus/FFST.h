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
* Fast finite shearlet tranform implementation following the Matlab implementation
* of the image processing group of TU Kaiserslautern. The matlab code and detailed
* publications on their method can be found at:
* http://www.mathematik.uni-kl.de/imagepro/software/ffst/
* A copy of the same matlab files is also available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
*
* NOTE: This implementation ONLY works for images of size 512x512, skipping over
* generating the filter-banks on the run to speed up computation speed. Instead,
* all frequency domain filters are stored in a big lookup table that is initalized
* the first time the transform is called.
**/

namespace FFST{
	/**
	* In an admittedly unelegant fashion, non-zero frequency domain coefficients for 512x512
	* image decomposition/reconstruction is stored as a constant in arrays 'coeff_pos' and
	* 'coeff_val'. This is used for initializing the filterbank in set_coeffs_512x512.
	**/
	extern float coeff_pos[616952][3];
	extern float coeff_val[616952];

	//Forward/inverse transform functions
	std::vector<cv::Mat> shearletTransformSpect(cv::Mat &img);
	cv::Mat inverseShearletTransformSpect(std::vector<cv::Mat> &ST);

	//Initializes the filterbank for 512x512 images
	void set_coeffs_512x512();

	//Internal functions used during decomposition/reconstrution
	cv::Mat fftshift(cv::Mat &x);
	cv::Mat ifftshift(cv::Mat &x);
	cv::Mat circshift(cv::Mat &M, int x, int y);
}

