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

#include <platypus/CradleFunctions.h>
#include <opencv2/opencv.hpp>
#include <vector>

/**
* Collection of all functions that make texture separation, and in particular, wood grain removal possible.
* The main function is textureRemove(), that applies an unsupervised learning algorithm to train the
* statistical model of the wood grain.
**/

namespace TextureRemoval{
	
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;

	//Structure representing all the parameters of the wood-grain statistical model
	//A detailed explanation of what each variable stands for is given in the IPOL article
	struct cradle_model_fitting{
		std::vector<cv::Mat> kappa_v;
		std::vector<cv::Mat> Gamma_v;
		std::vector<float> rho_v;
		std::vector<cv::Mat> xi_v;
		std::vector<cv::Mat> Lambda_v;
		std::vector<cv::Mat> ps_v;
		std::vector<cv::Mat> etac_v;
		std::vector<cv::Mat> etanc_v;
	};

	cradle_model_fitting gibbsSampling(std::vector<std::vector<float>> &cradle, std::vector<std::vector<float>> &noncradle);
	
	//Function responsible for separation
	void post_inference(
		cradle_model_fitting &model,				//Statistical model to be used for the separation
		std::vector<std::vector<float>> &c,			//Coefficients of the cradle component to be separated
		std::vector<std::vector<float>> &nc,		//Samples of non-cradled doefficients
		std::vector<std::vector<float>> &difference	//Separation result is stored here 
	);

	//Entry point to texture separation
	void textureRemove(
		cv::Mat &in,								//Input image for wood grain separation
		cv::Mat &mask,								//Mask component, as returned by cradle removal step
		cv::Mat &out,								//Result image is stored here
		const CradleFunctions::MarkedSegments &ms	//Processing information, as returned by cradle removal step
	);

	//Take 'cnt' samples, selected randomly from 'dts' and returned in 'samples' 
	void sampleDataset(std::vector<std::vector<float>> &dts, std::vector<std::vector<float>> &samples, int cnt);

	//Reconstruct image block between points (sx,sy) (ex,ey) with a local shift of (csx,csy)
	void reconstructBlock(cv::Mat &texture, std::vector<cv::Mat> &coeffs, int sx, int sy, int csx, int csy, int cex, int cey);
	
	//Normalize non-cradle samples, return sample mean and variance
	void normalizeNonCradle(std::vector<std::vector<float>> &nc, std::vector<float> &mean, std::vector<float> &var);

	//Normalize cradle samples using mean and variance specified
	void normalizeSamples(std::vector<std::vector<float>> &c, std::vector<float> &mean, std::vector<float> &var);

	//Undo normalization
	void unNormalizeSamples(std::vector<std::vector<float>> &cradle, std::vector<float> &mean, std::vector<float> &var);
	void unNormalizeSamplesCrossSection(std::vector<std::vector<float>> &cradle, std::vector<float> &meanh, std::vector<float> &varh, std::vector<float> &meanv, std::vector<float> &varv);
	
	//Cholesky decomposition of matrix A
	cv::Mat Cholesky(cv::Mat &A);
	cv::Mat CholeskyLower(cv::Mat &A);

	//Sampling from multi-variate normal distribution with 'mean' and 'covar' covariance specified
	std::vector<float> mvnpdf(std::vector<std::vector<float>> &X, std::vector<float> &mean, cv::Mat &covar);
	std::vector<float> mvnpdf(std::vector<std::vector<float>> &X, cv::Mat &mean, cv::Mat &covar);
}