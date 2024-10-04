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
* Morphological Component Analysis (MCA) implementation based on the MCALab Matlab
* implementation. The matlab code and detailed publications on their method can be found at:
* https://fadili.users.greyc.fr/demos/WaveRestore/downloads/mcalab/Home.html
* A copy of our matlab files that were translated into cpp is also available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
*
**/

namespace MCA{
	const int SHEARLET  =   0;		//Sheerlet dictionary
	const int DTWDC		=	1;		//Dual-tree wavelet decomposition dictionary
	const int FDCT		=	2;		//Curvelet decomposition dictionary

	//Separate image 'in' into a texture and cartoon part using the dictionaries specified in dict
	void MCA_Bcr(cv::Mat &in, std::vector<int> &dict, cv::Mat &texture, cv::Mat &cartoon);

	//Auxiliary functions for the MCA decomposition
	//More details are given in the Matlab code from which these functions were translated
	cv::Mat TVCorrection(cv::Mat &x, float gamma);
	float softThreshold(float val, float lambda);
	void calculateL2Norm(int n, std::vector<int> &dict, std::vector<std::vector<std::vector<float>>> &norm);
	float startingPoint(cv::Mat &in, std::vector<int> &dict, std::vector<std::vector<std::vector<float>>> &norm);
	cv::Mat analysis_threshold_synthesis(cv::Mat &in, int dict, float lambda, std::vector<std::vector<float>> &norm);
	float getResidualNorm(cv::Mat &residual);
}

