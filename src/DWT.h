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
#include <list>
#include <vector>

/**
* Dual tree wavelet transform implementaton following the Matlab implementation
* See reference Matlab code available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
**/

namespace DWT{
	extern cv::Mat h0o, h0a, h0b, h1o, h1a, h1b;	//Decomposition filters
	extern cv::Mat g0o, g0a, g0b, g1o, g1a, g1b;	//Reconstruction filters
	
	//Forward transform of 'img' into L levels, results returned in 'w1' and 'w2' in matrix form
	void cdwt2(cv::Mat &img, int L, cv::Mat &w1, cv::Mat &w2);

	//Forward transform of 'img' into L levels, results returned in vector form 'C' with reconstruction information in 'S1' and 'S2'
	void dtwavedec2(cv::Mat &img, int L, std::vector<int> &S1, std::vector<int> &S2, std::vector<float> &C);
	std::list<float> dtwavedec2(cv::Mat &img, int L, std::vector<int> &S1, std::vector<int> &S2);

	//Inverse transform of 'w1' and 'w2' of 'L' levels back into image 'img'
	void icdwt2(int L, cv::Mat &w1, cv::Mat &w2, cv::Mat &img);

	//Inverse transform of vector form 'C' and reconstruction information in 'S1' and 'S2' of 'L' levels back to image 'img'
	void dtwaverec2(cv::Mat &img, int L, std::vector<int> &S1, std::vector<int> &S2, std::vector<float> &C);
	
	/**
	* Internal functions used during decomposition/reconstrution
	**/
	void icdwt2_bands(int L, std::vector<std::vector<cv::Mat>> &in, cv::Mat &img);
	void cdwt2_bands(cv::Mat &img, int L, std::vector<std::vector<cv::Mat>> &out);

	void colifilt(cv::Mat &in, cv::Mat &ha, cv::Mat &hb, cv::Mat &out);

	void colfilter(cv::Mat &in, cv::Mat &filter, cv::Mat &out);
	void coldfilt(cv::Mat &in, cv::Mat &ha, cv::Mat &hb, cv::Mat &out);
	
	int icwtband2(cv::Mat &w1, cv::Mat &w2, std::vector<int> &S1, std::vector<int> &S2, int l, int o, std::vector<float> &c);
	void cwtband2(std::vector<float> &C, std::vector<int> &S1, std::vector<int> &S2, int l, int o, cv::Mat &Z);
	void cwtband2(std::vector<float> &C, std::vector<int> &S1, std::vector<int> &S2, int l, int o, cv::Mat &w1, cv::Mat &w2);

	int icwtband6(std::vector<cv::Mat> &Z, std::vector<int> &S1, std::vector<int> &S2, int l, std::vector<float> &C);
	void cwtband6(std::vector<float> &C, std::vector<int> &S1, std::vector<int> &S2, int l, std::vector<cv::Mat> &Z);

	void c2q(cv::Mat &inw1, cv::Mat &inw2, cv::Mat &out);
	void q2c(cv::Mat &in, cv::Mat &out);
	void q2c(cv::Mat &in, cv::Mat &w1, cv::Mat &w2);
}