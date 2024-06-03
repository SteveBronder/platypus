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

#include <platypus/Shearlet.h>
#include <platypus/HaarDWT.h>
#include <platypus/MCA.h>
#include <platypus/DWT.h>
#include <platypus/FDCT.h>

/**
* Morphological Component Analysis (MCA) implementation based on the MCALab Matlab
* implementation. The matlab code and detailed publications on their method can be found at:
* https://fadili.users.greyc.fr/demos/WaveRestore/downloads/mcalab/Home.html
* A copy of our matlab files that were translated into cpp is also available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
*
**/

namespace MCA{
	const float sigma = 1E-6;			//Stopping threshold
	const int itermax = 100;			//Max nr of iterations
	const float tvregparam = 0.0;		//TV regularization parameter(usually applied to the piece - wise smooth component, e.g. UDWT or curvelet dictionary) 
	const int tvregdict = -1;			//TV regularization dictionary index
	const float MCA_thershold = 5e-4;	//Stopping criteria for MCA (stop if residual norm is below this value)
	const float MCA_diff_lim  = 1e-6;	//Stopping criteria for MCA (stop if difference is below this value)

	//Decomposition parameters under Shearlet dictionary
	int dcomp_v[] = { 4, 4, 4, 4, 4,};
	std::vector<int> dcomp(dcomp_v, dcomp_v + sizeof(dcomp_v) / sizeof(int));
	int dsize_v[] = { 128, 128, 128, 128, 128 };
	std::vector<int> dsize(dsize_v, dsize_v + sizeof(dsize_v) / sizeof(int));


	//Separate image 'in' into a texture and cartoon part using the dictionaries specified in dict
	void MCA_Bcr(cv::Mat &in, std::vector<int> &dict, cv::Mat &texture, cv::Mat &cartoon){

		// Initializations
		int N, M, n, max;
		N = in.rows;
		M = in.cols;
		
		//Pad input to 512 block size
		n = 512;
		cv::Mat lin(n, n, CV_32F, cv::Scalar(0));
		for (int i = 0; i < N; i++){
			for (int j = 0; j < M; j++){
				lin.at<float>(i, j) = in.at<float>(i, j);
			}
		}

		float delta, deltamax, lambda;

		//Initialize filterbank for Shearlet transform
		for (int i = 0; i < dict.size(); i++){
			if (dict[i] == SHEARLET){

				//Pre-calculate filter bank to speed-up calculations
				Shearlet::getFilterBank(lin.rows, dcomp, dsize);
			}
		}

		//Calculate norms
		std::vector<std::vector<std::vector<float>>> norms;
		calculateL2Norm(n, dict, norms);

		//Starting point
		deltamax = startingPoint(lin, dict, norms);
		delta = deltamax;

		//Starting threshold
		lambda = std::pow(deltamax / sigma, 1.0 / (1 - itermax));	// Exponential decrease

		//Initialize reconstruction parts
		cv::Mat residual, ra;
		std::vector<cv::Mat> part(dict.size());
		for (int i = 0; i < dict.size(); i++){
			part[i] = cv::Mat(n, n, CV_32F, cv::Scalar(0));
		}

		// Start the modified Block Relaxation Algorithm
		float residual_norm = getResidualNorm(lin);
		float prev = residual_norm + 1;
		lin.copyTo(residual);
		int iter = 0;
		int increase = 0;
		float residual_norm_best = residual_norm + 1;
		cv::Mat best;

		//While solution is still improving sufficiently..
		while ((residual_norm > MCA_thershold) && (increase < 4)){

			//Cycle over dictionaries
			for (int j = 0; j < dict.size(); j++){

				//Update Part assuming other parts fixed
				ra = part[j] + residual;

				//Decomposition - Thresholdin - Reconstrution
				part[j] = analysis_threshold_synthesis(ra, dict[j], delta, norms[j]);

				//If TVCorrection enabled (tvregparam != 0), apply it on specified dictionary
				if (tvregdict == j && tvregparam != 0){
					part[j] = TVCorrection(part[tvregdict], tvregparam);
				}
			}

			//Update step
			delta *= lambda;

			//Calculate the residual image
			lin.copyTo(residual);
			for (int j = 0; j < dict.size(); j++){
				residual -= part[j];
			}
			residual_norm = getResidualNorm(residual);

			//Check if minimum reached
			if (residual_norm < residual_norm_best){
				residual_norm_best = residual_norm;
				part[1].copyTo(best);
				increase = 0;
			}
			else{
				increase++;
			}
			increase = 0;

			iter++;
			if (iter > itermax){
				residual_norm = 0;
			}

			if (std::abs(residual_norm - prev) < MCA_diff_lim && (residual_norm < 5e-3)){
				residual_norm = 0;
			}else{
				prev = residual_norm;
			}
		}
		
		//Save out final parts
		cartoon = best;
		texture = lin - best;
	}

	float getResidualNorm(cv::Mat &residual){
		float norm = 0;
		for (int i = 0; i < residual.rows; i++){
			for (int j = 0; j < residual.cols; j++){
				float val = residual.at<float>(i, j);
				norm += val*val;
			}
		}
		return std::sqrt(norm) / residual.rows / residual.cols;
	}

	cv::Mat TVCorrection(cv::Mat &x, float gamma){
		// Total variation implemented using the approximate(exact in 1D) equivalence between the TV norm and the l_1 norm of the Haar(heaviside) coefficients.
		cv::Mat out, tmp;

		//Decompose
		HaarDWT::cvHaarWavelet(x, tmp, 1);

		//Filter LH, HH, HL components
		int w = x.rows;
		int h = x.cols;
		for (int i = 0; i < w; i++){
			for (int j = 0; j < h; j++){
				if (i > w / 2 || j > h/2){
					tmp.at<float>(i, j) = softThreshold(tmp.at<float>(i, j), gamma);
				}
			}
		}

		//Reconstruct
		HaarDWT::cvInvHaarWavelet(tmp, out, 1);

		return out;
	}

	// Return sign(val) * min(|val - lambda|, 0)
	float softThreshold(float val, float lambda){
		float aval = std::abs(val);
		if (aval <= lambda)
			return 0;
		if (val > 0)
			return val - lambda;
		else
			return val + lambda;
	}

	cv::Mat analysis_threshold_synthesis(cv::Mat &in, int dict, float lambda, std::vector<std::vector<float>> &norm){
		//Perform decomposition + thresholding + reconctruction for a single dictionary
		cv::Mat out;

		//Decomposition in function of dictionaries
		if (dict == FDCT){

			//Decomposition
			std::vector<std::vector<cv::Mat>> dst;
			dst = FDCT::fdct_wrapping(in, 7);

			//Iterate though all coefficients
			for (int j1 = 0; j1 < dst.size(); j1++){
				if (j1 == 0){
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < (dst[j1][j2]).rows; k++){
							for (int l = 0; l < (dst[j1][j2]).cols; l++){
								//Zero low pass
								dst[j1][j2].at<float>(k, l) = 0;
							}
						}
					}
				}
				else{
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < (dst[j1][j2]).rows; k++){
							for (int l = 0; l < (dst[j1][j2]).cols; l++){
								//Filter in function of lambda
								float val = std::abs(dst[j1][j2].at<float>(k, l) / norm[j1][j2]);
								if (val < lambda){
									dst[j1][j2].at<float>(k, l) = 0;
								}
							}
						}
					}
				}
			}

			//Reconstruct image
			out = FDCT::ifdct_wrapping(dst, in.rows, in.cols);
		}

		//Decomposition in function of dictionaries
		if (dict == SHEARLET){

			//Decomposition
			std::vector<std::vector<cv::Mat>> dst;
			Shearlet::nsst_dec2(in, dcomp, dsize, dst, Shearlet::shear_filter);

			//Iterate though all coefficients
			for (int j1 = 0; j1 < dst.size(); j1++){
				if (j1 == 0){
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < (dst[j1][j2]).cols; k++){
							for (int l = 0; l < (dst[j1][j2]).rows; l++){
								//Zero low pass
								dst[j1][j2].at<float>(k, l) = 0;
							}
						}
					}
				}
				else{
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < (dst[j1][j2]).cols; k++){
							for (int l = 0; l < (dst[j1][j2]).rows; l++){
								//Filter in function of lambda
								float val = std::abs(dst[j1][j2].at<float>(k, l) / norm[j1][j2]);
								if (val < lambda){
									dst[j1][j2].at<float>(k, l) = 0;
								}
							}
						}
					}
				}
			}

			//Reconstruct image
			Shearlet::nsst_rec2(dst, Shearlet::shear_filter, out);
		}

		//Dual Tree Wavelet Decomposition
		if (dict == DTWDC){
			//Decomposition
			std::vector<std::vector<cv::Mat>> decomp;
			DWT::cdwt2_bands(in, 6, decomp);

			for (int j1 = 0; j1 < decomp.size(); j1++){
				for (int j2 = 0; j2 < decomp[j1].size(); j2++){
					for (int k = 0; k < decomp[j1][j2].rows; k++){
						for (int l = 0; l < decomp[j1][j2].cols; l++){
							float re = decomp[j1][j2].at<cv::Point2f>(k, l).x;
							float im = decomp[j1][j2].at<cv::Point2f>(k, l).y;

							//Filter in function of lambda
							float val = std::sqrt(re*re + im*im) / norm[j1][j2];
							if (val < lambda){
								decomp[j1][j2].at<cv::Point2f>(k, l).x = 0;
								decomp[j1][j2].at<cv::Point2f>(k, l).y = 0;
							}
						}
					}
				}
			}

			//Reconstruct image
			DWT::icdwt2_bands(6, decomp, out);
		}

		//EXTEND FOR ADDITIONAL DICTIONARIES HERE


		return out;
	}

	void calculateL2Norm(int n, std::vector<int> &dict, std::vector<std::vector<std::vector<float>>> &norm){
		//Array of pointer to normalization structure
		(norm) = std::vector<std::vector<std::vector<float>>>((dict).size());

		//normalized Dirac image
		cv::Mat dirac(n, n, CV_32F, cv::Scalar(0));
		dirac.at<float>(n / 2, n / 2) = n;

		//Calculate norm for each dictionary
		for (int i = 0; i < dict.size(); i++){
			if (dict[i] == FDCT){

				//Decomposition
				std::vector<std::vector<cv::Mat>> dst;
				dst = FDCT::fdct_wrapping(dirac, 7);

				//Normalize
				norm[i] = std::vector<std::vector<float>>((dst).size());

				for (int j = 0; j < (dst).size(); j++){
					norm[i][j] = std::vector<float>(dst[j].size());
					for (int k = 0; k < (dst[j]).size(); k++){
						float tot = 0;
						for (int a = 0; a < dst[j][k].rows; a++){
							for (int b = 0; b < dst[j][k].cols; b++){
								float val = dst[j][k].at<float>(a, b);
								tot += val*val;
							}
						}
						norm[i][j][k] = std::sqrt(tot) / std::sqrt(dst[j][k].rows * dst[j][k].cols);
					}
				}
			}

			//Decomposition in function of dictionaries
			if (dict[i] == SHEARLET){

				//Decomposition
				std::vector<std::vector<cv::Mat>> dst;
				Shearlet::nsst_dec2(dirac, dcomp, dsize, dst, Shearlet::shear_filter);
				
				//Normalize
				norm[i] = std::vector<std::vector<float>>((dst).size());

				for (int j = 0; j < (dst).size(); j++){
					norm[i][j] = std::vector<float>(dst[j].size());
					for (int k = 0; k < (dst[j]).size(); k++){
						float tot = 0;
						for (int a = 0; a < dst[j][k].rows; a++){
							for (int b = 0; b < dst[j][k].cols; b++){
								float val = dst[j][k].at<float>(a, b);
								tot += val*val;
							}
						}
						norm[i][j][k] = std::sqrt(tot) / std::sqrt(dst[j][k].rows * dst[j][k].cols);
					}
				}
			}
			if (dict[i] == DTWDC){
				//Decomposition
				std::vector<std::vector<cv::Mat>> decomp;
				DWT::cdwt2_bands(dirac, 6, decomp);

				//Normalize
				norm[i] = std::vector<std::vector<float>>((decomp).size());
				for (int j1 = 0; j1 < decomp.size(); j1++){
					//Initialize
					norm[i][j1] = std::vector<float>(decomp[j1].size());
					for (int j2 = 0; j2 < decomp[j1].size(); j2++){
						float tot = 0;
						for (int k = 0; k < (decomp[j1][j2]).cols; k++){
							for (int l = 0; l < (decomp[j1][j2]).rows; l++){
								//Get norm of complex values
								float absval;
								float re = (decomp[j1][j2]).at<cv::Point2f>(k, l).x;
								float im = (decomp[j1][j2]).at<cv::Point2f>(k, l).y;
								absval = re*re + im*im;
								tot += absval;
							}
						}
						//Store norm
						norm[i][j1][j2] = std::sqrt(tot) / std::sqrt(2 * decomp[j1][j2].rows * decomp[j1][j2].cols);
					}
				}
			}
		}
	}

	float startingPoint(cv::Mat &in, std::vector<int> &dict, std::vector<std::vector<std::vector<float>>> &norm){
		//Return min of max of coefficient absolute values for the given dictionaries 
		float min = -1;

		//Go through all dictionaries
		for (int i = 0; i < (dict).size(); i++){
			float lmax = -1;

			if ((dict)[i] == FDCT){

				//Decomposition
				std::vector<std::vector<cv::Mat>> dst;
				dst = FDCT::fdct_wrapping(in, 7);

				//Iterate though all coefficients - SKIP OVER LOWEST LEVEL
				for (int j1 = 1; j1 < dst.size(); j1++){
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < dst[j1][j2].rows; k++){
							for (int l = 0; l < dst[j1][j2].cols; l++){

								//Get normalized absolute value
								float absval = abs(dst[j1][j2].at<float>(k, l)) / norm[i][j1][j2];

								// if better
								if (absval > lmax){
									lmax = absval;
								}
							}
						}
					}
				}
			}

			//Decomposition in function of dictionaries
			if ((dict)[i] == SHEARLET){

				//Decomposition
				std::vector<std::vector<cv::Mat>> dst;
				Shearlet::nsst_dec2(in, dcomp, dsize, dst, Shearlet::shear_filter);

				//Iterate though all coefficients - SKIP OVER LOWEST LEVEL
				for (int j1 = 1; j1 < dst.size(); j1++){
					for (int j2 = 0; j2 < dst[j1].size(); j2++){
						for (int k = 0; k < dst[j1][j2].rows; k++){
							for (int l = 0; l < dst[j1][j2].cols; l++){

								//Get normalized absolute value
								float absval = dst[j1][j2].at<float>(k, l) / norm[i][j1][j2];
								if (absval < 0)
									absval = -absval;

								//Check if better
								if (absval > lmax){
									lmax = absval;
								}
							}
						}
					}
				}
			}

			//Dual Tree Wavelet Decomposition
			if ((dict)[i] == DTWDC){
				//Decomposition
				std::vector<std::vector<cv::Mat>> decomp;
				DWT::cdwt2_bands(in, 6, decomp);

				//Iterate though all coefficients in w1
				for (int j1 = 0; j1 < decomp.size(); j1++){
					for (int j2 = 0; j2 < decomp[j1].size(); j2++){
						for (int k = 0; k < decomp[j1][j2].cols; k++){
							for (int l = 0; l < decomp[j1][j2].rows; l++){
								//Get absolute value - complex values
								float absval;
								float re = decomp[j1][j2].at<cv::Point2f>(k, l).x;
								float im = decomp[j1][j2].at<cv::Point2f>(k, l).y;

								//Get normalized absolute value
								absval = sqrt(re*re + im*im) / norm[i][j1][j2];

								//Check if better
								if (absval > lmax){
									lmax = absval;
								}
							}
						}
					}
				}
			}


			//EXTEND FOR ADDITIONAL DICTIONARIES HERE

			//Get minima of maxes
			if (min == -1 || lmax < min){
				min = lmax;
			}
		}
		return min;
	}
}