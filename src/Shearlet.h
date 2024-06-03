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
* Discrete shearlet transform implementation finite-length filters following the Matlab
* implementation of Glenn R. Easley, Demetrio Labate. and Wang-Q Lim. The matlab code and detailed
* publications on their method can be found at:
* http://www.math.uh.edu/~dlabate/software.html
*
* NOTE: This implementation ONLY works for images of size 512x512, skipping over
* generating the filter-banks on the run to speed up computation speed. Instead,
* all frequency domain filters are stored in a big lookup table that is initalized
* by calling function getFilterBank().
**/

namespace Shearlet{
	extern cv::Mat g0, g1, h0, h1;
	extern float f16_4[16][16][16];
	extern float f32_3[8][32][32];
	extern float f32_6[64][32][32];
	extern float f128_4[16][128][128];
	extern float f64_4[16][64][64];
	extern float f32_4[16][32][32];
	extern float f64_6[64][64][64];
	extern float f128_5[32][128][128];
	extern float f128_6[64][128][128];

	//Stores precomputed filters
	extern std::vector<std::vector<cv::Mat>> shear_filter;
	
	//Forward transform
	void nsst_dec2(
		cv::Mat &in,								//Input image
		std::vector<int> &decomp,					//number of angular directions for each decomposition level
		std::vector<int> &dsize,					//filter size for each decomposition level
		std::vector<std::vector<cv::Mat>> &dst,		//decomposition coefficients
		std::vector<std::vector<cv::Mat>> &shear_f	//shearlet filters for each decomposition level
	);

	//Reverse transform
	void nsst_rec2(
		std::vector<std::vector<cv::Mat>> &dst,		//input decomposition coefficients
		std::vector<std::vector<cv::Mat>> &shear_f, //shearlet filters from decomposition
		cv::Mat &out								//reconstructed image
	);

	void getFilterBank(int L, std::vector<int> &decomp, std::vector<int> &dsize);
	
	//Auxiliary functions used internally
	std::vector<cv::Mat> atrousdec(cv::Mat &in, int level);
	cv::Mat atrousrec(std::vector<cv::Mat> &y);

	cv::Mat upsample2df(cv::Mat &filter, int power);
	cv::Mat symext(cv::Mat &in, cv::Mat &filter, int s1, int s2);
	cv::Mat atrousc(cv::Mat &S, cv::Mat &F, int L);
}