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
#include <platypus/DWT.h>
#include <opencv2/opencv.hpp>
#include <vector>

/**
* Dual tree wavelet transform implementaton following the original Matlab implementation.
* See reference Matlab code available via the Platypus project at
* http://www.project-platypus.net/cradledownload.html
* Some parameters from the Matlab implementation are fixed in this code
*	biort - near_sym_a
*	qshift - qshift_a
* For a more detailed explanation on the functions in this file, check out the Matlab
* implementation.
**/

namespace DWT{

	int ORIENTATION_L = 0;
	int ORIENTATION_V = 1;
	int ORIENTATION_H = 2;
	int ORIENTATION_D = 3;

	void icdwt2_bands(int L, std::vector<std::vector<cv::Mat>> &in, cv::Mat &img){
		int N = (in)[(in).size()-1][0].rows * 2;
		int Lx = std::log2(N + 0.5);
		int cn = 0, ind;

		//Create decomposition sizes
		std::vector<int> S1(L), S2(L);
		int tmp = N;
		for (int i = 0; i < L; i++){
			S1[L - i - 1] = tmp;
			S2[L - i - 1] = tmp;
			tmp /= 2;
		}

		for (int i = 0; i < L; i++){
			cn += 3 * S1[i] * S2[i];
		}
		cn += S1[0] * S2[0];

		//Allocate C
		std::vector<float> C(4 * N*N);
		std::vector<float> c; //For passing over to functions

		//LoLo
		ind = icwtband2(in[0][0], in[0][1], S1, S2, L, ORIENTATION_L, c);
		//Copy
		for (int i = 0; i < c.size(); i++)
			C[ind + i] = c[i];

		//Higher bands
		for (int ll = 1; ll <= L; ll++){
			ind = icwtband6(in[L - ll + 1], S1, S2, ll, c);
			//Copy
			for (int i = 0; i < c.size(); i++)
				C[ind + i] = c[i];
		}

		//Inverse transform
		dtwaverec2(img, L, S1, S2, C);
	}

	void cdwt2_bands(cv::Mat &img, int L, std::vector<std::vector<cv::Mat>> &out){
		
		//Call wavelet decomposition
		std::vector<int> S1, S2;
		std::vector<float> C;
		dtwavedec2(img, L, S1, S2, C);

		//Put bands back together in (*out)
		(out) = std::vector<std::vector<cv::Mat>>(L + 1);

		//Get lowest band matrix
		(out)[0] = std::vector<cv::Mat>(2);
		cwtband2(C, S1, S2, L, ORIENTATION_L, out[0][0], out[0][1]);

		//Get all resolution levels
		for (int i = 1; i <= L; i++){
			cwtband6(C, S1, S2, i, out[L-i+1]);
		}
	}

	void icdwt2(int L, cv::Mat &w1, cv::Mat &w2, cv::Mat &img){
		int N = (w1).rows;
		int Lx = std::log2(N + 0.5);

		//Create decomposition sizes
		std::vector<int> S1(L), S2(L);
		int tmp = N;
		for (int i = 0; i < L; i++){
			S1[L - i - 1] = tmp;
			S2[L - i - 1] = tmp;
			tmp /= 2;
		}

		//Allocate C
		std::vector<float> C(4*N*N);
		std::vector<float> c; //For passing over to functions

        cv::Mat t1, t2;
        t1 =(w1)(cv::Range(0, std::exp2(Lx - L) + 0.5), cv::Range(0, std::exp2(Lx - L) + 0.5));
        t2 =(w1)(cv::Range(0, std::exp2(Lx - L) + 0.5), cv::Range(0, std::exp2(Lx - L) + 0.5));
		//Band LoLo
		int ind = icwtband2(t1, t2,	S1, S2, L, ORIENTATION_L, c);
		//Copy
		for (int i = 0; i < c.size(); i++)
			C[ind + i] = c[i];

		//Additional bands
		for (int ll = Lx; ll >= (Lx - L + 1); ll--){
			int k = std::exp2(ll-1)+0.5;
            t1 =(w1)(cv::Range(0, k), cv::Range(k, 2*k));
            t2 =(w2)(cv::Range(0, k), cv::Range(k, 2*k));
			ind = icwtband2(t1,t2,S1, S2, Lx - ll + 1, ORIENTATION_V, c);
			for (int i = 0; i < c.size(); i++)
				C[ind + i] = c[i];

            
            t1 =(w1)(cv::Range(k, 2*k), cv::Range(0, k));
            t2 =(w2)(cv::Range(k, 2*k), cv::Range(0, k));
			ind = icwtband2(t1,t2,S1, S2, Lx - ll + 1, ORIENTATION_H, c);
			for (int i = 0; i < c.size(); i++)
				C[ind + i] = c[i];

            
            t1 =(w1)(cv::Range(k, 2*k), cv::Range(k, 2 * k));
            t2 =(w2)(cv::Range(k, 2 * k), cv::Range(k, 2 * k));
			ind = icwtband2(t1,t2,S1, S2, Lx - ll + 1, ORIENTATION_D, c);
			for (int i = 0; i < c.size(); i++)
				C[ind + i] = c[i];
		}

		dtwaverec2(img, L, S1, S2, C);
	}

	int icwtband6(std::vector<cv::Mat> &Z, std::vector<int> &S1, std::vector<int> &S2, int l, std::vector<float> &c){
		int start_sec = -1, size_seg, rs, cs;

		rs = (S1)[(S1).size() - l];
		cs = (S2)[(S1).size() - l];

		cv::Mat yh, yv, yd;

		//Convert complex back to float
		c2q((Z)[0], (Z)[1], yh);
		c2q((Z)[2], (Z)[3], yv);
		c2q((Z)[4], (Z)[5], yd);

		//Assemble these as the output vector

		//Make array out of it
		c = std::vector<float>(3*rs*cs);
		int ind = 0;
		//Horizontal
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				(c)[ind] = yh.at<float>(i, j);
				ind++;
			}
		}
		//Vertical
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				(c)[ind] = yv.at<float>(i, j);
				ind++;
			}
		}
		//Diagonal
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				(c)[ind] = yd.at<float>(i, j);
				ind++;
			}
		}

		start_sec = (S1)[0] * (S2)[0];
		for (int i = 0; i < (int)((S1).size() - l); i++){
			start_sec += (S1)[i] * (S2)[i] * 3;
		}
		//start_sec--;	//MATLAB indexing shift compensation

		return start_sec;
	}

	//Returns start index of copying
	int icwtband2(cv::Mat &w1, cv::Mat &w2, std::vector<int> &S1, std::vector<int> &S2, int l, int o, std::vector<float> &c){
		int start_sec = -1, size_seg, rs, cs;

		rs = (S1)[(S1).size() - l];
		cs = (S2)[(S1).size() - l];

		cv::Mat ym;
		c2q(w1, w2, ym);	//Convert complex back to float

		//Make array out of it
		c = std::vector<float>(rs*cs);
		int ind = 0;
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				(c)[ind] = ym.at<float>(i, j);
				ind++;
			}
		}

		size_seg = rs*cs;
		start_sec = (S1)[0] * (S2)[0];
		for (int i = 0; i < (int)((S1).size() - l); i++){
			start_sec += (S1)[i] * (S2)[i] * 3;
		}
		start_sec--;	//MATLAB indexing shift compensation

		//std::copy((*C).begin(), (*C).end(), (*Cvector));
		
		if (o == ORIENTATION_L){
			return 0;
		}
		else if (o == ORIENTATION_H){
			return start_sec + 1;
		}
		else if (o == ORIENTATION_V){
			return start_sec + size_seg + 1;
		}
		else{
			//This should be ORIENTATION_D
			return start_sec + size_seg * 2 + 1;
		}
	}

	void cdwt2(cv::Mat &img, int L, cv::Mat &w1, cv::Mat &w2){
		int Lx = std::log2((img).rows)+0.5;
		
		//Call wavelet decomposition
		std::vector<int> S1, S2;
		std::vector<float> C;
		dtwavedec2(img, L, S1, S2, C);

		//Initialize results
		(w1) = cv::Mat((img).rows, (img).cols, CV_32FC2, cv::Scalar(0));
		(w2) = cv::Mat((img).rows, (img).cols, CV_32FC2, cv::Scalar(0));
		
		for (int ll = Lx; ll >= Lx - L + 1; ll--){


			int k = std::exp2(ll-1) + 0.5;
			cv::Mat b75, b15, b45;

			cwtband2(C, S1, S2, Lx - ll + 1, ORIENTATION_V, b75);
			cwtband2(C, S1, S2, Lx - ll + 1, ORIENTATION_H, b15);
			cwtband2(C, S1, S2, Lx - ll + 1, ORIENTATION_D, b45);

			//Copy to correct locations
			//B75
			for (int i = 0; i < k; i++){
				for (int j = k; j < 2 * k; j++){
					(w1).at<cv::Point2f>(i, j).x = b75.at<cv::Point2f>(i, j - k).x;
					(w1).at<cv::Point2f>(i, j).y = b75.at<cv::Point2f>(i, j - k).y;

					(w2).at<cv::Point2f>(i, j).x = b75.at<cv::Point2f>(i + k, j - k).x;
					(w2).at<cv::Point2f>(i, j).y = b75.at<cv::Point2f>(i + k, j - k).y;
				}
			}
			//B15
			for (int i = k; i < 2 * k; i++){
				for (int j = 0; j < k; j++){
					(w1).at<cv::Point2f>(i, j).x = b15.at<cv::Point2f>(i - k, j).x;
					(w1).at<cv::Point2f>(i, j).y = b15.at<cv::Point2f>(i - k, j).y;

					(w2).at<cv::Point2f>(i, j).x = b15.at<cv::Point2f>(i, j).x;
					(w2).at<cv::Point2f>(i, j).y = b15.at<cv::Point2f>(i, j).y;
				}
			}
			//B45
			for (int i = k; i < 2 * k; i++){
				for (int j = k; j < 2 * k; j++){
					(w1).at<cv::Point2f>(i, j).x = b45.at<cv::Point2f>(i - k, j - k).x;
					(w1).at<cv::Point2f>(i, j).y = b45.at<cv::Point2f>(i - k, j - k).y;

					(w2).at<cv::Point2f>(i, j).x = b45.at<cv::Point2f>(i, j - k).x;
					(w2).at<cv::Point2f>(i, j).y = b45.at<cv::Point2f>(i, j - k).y;
				}
			}
		}

		cv::Mat bl;
		cwtband2(C, S1, S2, L, ORIENTATION_L, bl);
		int k = std::exp2(Lx - L) + 0.5;
		for (int i = 0; i < k; i++){
			for (int j = 0; j < k; j++){
				(w1).at<cv::Point2f>(i, j).x = bl.at<cv::Point2f>(i, j).x;
				(w1).at<cv::Point2f>(i, j).y = bl.at<cv::Point2f>(i, j).y;

				(w2).at<cv::Point2f>(i, j).x = bl.at<cv::Point2f>(i + k, j).x;
				(w2).at<cv::Point2f>(i, j).y = bl.at<cv::Point2f>(i + k, j).y;
			}
		}

		//Clear up list/vector
		C.clear();

		//And DONE :D
	}

	void cwtband6(std::vector<float> &C, std::vector<int> &S1, std::vector<int> &S2, int l, std::vector<cv::Mat> &Z){
		int start_sec = -1, size_seg, rs, cs, rp, cp, start, finish;

		//Allocate memory for 6 bands
		(Z) = std::vector<cv::Mat>(6);

		rs = (S1)[(S1).size() - l];
		cs = (S2)[(S2).size() - l];
		cv::Mat reshape = cv::Mat(rs, cs, CV_32F);
		size_seg = rs*cs;

		rp = (S1).size() - l;
		cp = (S2).size() - l;

		//Get start_sec for required decomposition level
		start_sec = (S1)[0] * (S2)[0];
		for (int i = 0; i < rp; i++){
			start_sec += (S1)[i] * (S2)[i] * 3;
		}
		start_sec--; //Matlab index -1

		//Horizontal
		start = start_sec + 1;
		finish = start + size_seg - 1;
	
		//Reshape elements
		int ind = start;
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				reshape.at<float>(i, j) = (C)[ind];
				ind++;
			}
		}
		q2c(reshape, (Z)[0], (Z)[1]);

		//Vertical
		start = start_sec + size_seg + 1;
		finish = start + size_seg - 1;

		//Reshape elements
		ind = start;
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				reshape.at<float>(i, j) = (C)[ind];
				ind++;
			}
		}
		q2c(reshape, (Z)[2], (Z)[3]);

		//Diagonal
		start = start_sec + size_seg * 2 + 1;
		finish = start + size_seg - 1;

		//Reshape elements
		ind = start;
		for (int j = 0; j < cs; j++){
			for (int i = 0; i < rs; i++){
				reshape.at<float>(i, j) = (C)[ind];
				ind++;
			}
		}
		q2c(reshape, (Z)[4], (Z)[5]);
	}

	void cwtband2(std::vector<float> &Cvector, std::vector<int> &S1, std::vector<int> &S2, int l, int o, cv::Mat &w1, cv::Mat &w2){
		int start_sec = -1, size_seg, rs, cs;

		//std::copy((*C).begin(), (*C).end(), (*Cvector));

		if (o == ORIENTATION_L){
			//Get the lowest level of decomposition values
			rs = (S1)[0];
			cs = (S2)[0];

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);
			//Reshape elements
			int ind = 0;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, w1, w2);

			return;
		}
		else{
			//i.e., for all subimages except LoLo
			rs = (S1)[(S1).size() - l];
			cs = (S2)[(S2).size() - l];
			int ss = rs*cs;
			size_seg = rs*cs;
			int rp = (S1).size() - l;
			int cp = (S2).size() - l;
			start_sec = (S1)[0] * (S2)[0];
			for (int i = 0; i < rp; i++){
				start_sec += (S1)[i] * (S2)[i] * 3;
			}
			start_sec--; //Matlab index -1
		}

		if (o == ORIENTATION_H){
			int start = start_sec + 1;
			int finish = start + size_seg - 1;

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);
			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, w1, w2);
		}
		else if (o == ORIENTATION_V){
			int start = start_sec + size_seg + 1;
			int finish = start + size_seg - 1;

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);

			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, w1, w2);
		}
		else{
			//This should be ORIENTATION_D
			int start = start_sec + 2 * size_seg + 1;
			int finish = start + size_seg - 1;

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);

			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, w1, w2);
		}
	}

	void cwtband2(std::vector<float> &Cvector, std::vector<int> &S1, std::vector<int> &S2, int l, int o, cv::Mat &Z){
		int start_sec = -1, size_seg, rs, cs;

		//std::copy((*C).begin(), (*C).end(), (*Cvector));

		if (o == ORIENTATION_L){
			//Get the lowest level of decomposition values
			rs = (S1)[0];
			cs = (S2)[0];

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);
			//Reshape elements
			int ind = 0;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, Z);

			return;
		}
		else{
			//i.e., for all subimages except LoLo
			rs = (S1)[(S1).size() - l];
			cs = (S2)[(S2).size() - l];
			int ss = rs*cs;
			size_seg = rs*cs;
			int rp = (S1).size() - l;
			int cp = (S2).size() - l;
			start_sec = (S1)[0] * (S2)[0];
			for (int i = 0; i < rp; i++){
				start_sec += (S1)[i] * (S2)[i] * 3;
			}
			start_sec--; //Matlab index -1
		}

		if (o == ORIENTATION_H){
			int start = start_sec + 1;
			int finish = start + size_seg - 1;

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);
			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, Z);
		}
		else if (o == ORIENTATION_V){
			int start = start_sec + size_seg + 1;
			int finish = start + size_seg - 1;
			
			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);

			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, Z);
		}
		else{
			//This should be ORIENTATION_D
			int start = start_sec + 2*size_seg + 1;
			int finish = start + size_seg - 1;

			cv::Mat reshape = cv::Mat(rs, cs, CV_32F);

			//Reshape elements
			int ind = start;
			for (int j = 0; j < cs; j++){
				for (int i = 0; i < rs; i++){
					reshape.at<float>(i, j) = (Cvector)[ind];
					ind++;
				}
			}

			//Make it complex
			q2c(reshape, Z);
		}
	}

	void c2q(cv::Mat &inw1, cv::Mat &inw2, cv::Mat &out){
		//Convert from quads in *in to complex numbers in *out.
		(out) = cv::Mat((inw1).rows*2, (inw1).cols*2, CV_32F);	//Convert back to real numbers

		//Fill up (*out)
		double scale = std::sqrt(0.5);
		for (int i = 0; i < (inw1).rows; i++){
			for (int j = 0; j < (inw1).cols; j++){
				//(2k,2k)
				(out).at<float>(i * 2, j * 2) = ((inw1).at<cv::Point2f>(i, j).x + (inw2).at<cv::Point2f>(i, j).x)*scale;
				//(2k,2k+1)
				(out).at<float>(i * 2, j * 2 + 1) = ((inw1).at<cv::Point2f>(i, j).y + (inw2).at<cv::Point2f>(i, j).y)*scale;
				//(2k+1,2k)
				(out).at<float>(i * 2 + 1, j * 2) = ((inw1).at<cv::Point2f>(i, j).y - (inw2).at<cv::Point2f>(i, j).y)*scale;
				//(2k+1,2k+1)
				(out).at<float>(i * 2 + 1, j * 2 + 1) = ((inw2).at<cv::Point2f>(i, j).x - (inw1).at<cv::Point2f>(i, j).x)*scale;
			}
		}
	}

	void q2c(cv::Mat &in, cv::Mat &out){
		//Convert from quads in *in to complex numbers in *out.
		(out) = cv::Mat((in).rows, (in).cols/2, CV_32FC2);	//Use two channels for representing complex numbers!!

		cv::Mat a, b, c, d;
		a = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		b = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		c = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		d = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);

		for (int i = 0; i < (in).rows/2; i++){
			for (int j = 0; j < (in).cols/2; j++){
				a.at<float>(i, j) = (in).at<float>(i * 2, j * 2);
				b.at<float>(i, j) = (in).at<float>(i * 2, j * 2 + 1);
				c.at<float>(i, j) = (in).at<float>(i * 2 + 1, j * 2);
				d.at<float>(i, j) = (in).at<float>(i * 2 + 1, j * 2 + 1);
			}
		}
		//Fill up (*out)
		double scale = std::sqrt(0.5);
		for (int i = 0; i < (in).rows / 2; i++){
			for (int j = 0; j < (in).cols / 2; j++){
				//Real
				(out).at<cv::Point2f>(i, j).x = scale*(a.at<float>(i, j) - d.at<float>(i, j)); //[a - d]
				(out).at<cv::Point2f>(i + (in).rows / 2, j).x = scale*(a.at<float>(i, j) + d.at<float>(i, j)); //[a + d]

				//Imaginary
				(out).at<cv::Point2f>(i, j).y = scale*(b.at<float>(i, j) + c.at<float>(i, j)); //[b + c]
				(out).at<cv::Point2f>(i + (in).rows / 2, j).y = scale*(b.at<float>(i, j) - c.at<float>(i, j)); //[b - c]
			}
		}
	}

	void q2c(cv::Mat &in, cv::Mat &w1, cv::Mat &w2){
		//Convert from quads in *in to complex numbers in *out.
		(w1) = cv::Mat((in).rows / 2, (in).cols / 2, CV_32FC2);
		(w2) = cv::Mat((in).rows / 2, (in).cols / 2, CV_32FC2);

		cv::Mat a, b, c, d;
		a = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		b = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		c = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);
		d = cv::Mat((in).rows / 2, (in).cols / 2, CV_32F);

		for (int i = 0; i < (in).rows / 2; i++){
			for (int j = 0; j < (in).cols / 2; j++){
				a.at<float>(i, j) = (in).at<float>(i * 2, j * 2);
				b.at<float>(i, j) = (in).at<float>(i * 2, j * 2 + 1);
				c.at<float>(i, j) = (in).at<float>(i * 2 + 1, j * 2);
				d.at<float>(i, j) = (in).at<float>(i * 2 + 1, j * 2 + 1);
			}
		}
		//Fill up (*out)
		double scale = std::sqrt(0.5);
		for (int i = 0; i < (in).rows / 2; i++){
			for (int j = 0; j < (in).cols / 2; j++){
				//Real
				(w1).at<cv::Point2f>(i, j).x = scale*(a.at<float>(i, j) - d.at<float>(i, j)); //[a - d]
				(w2).at<cv::Point2f>(i, j).x = scale*(a.at<float>(i, j) + d.at<float>(i, j)); //[a + d]

				//Imaginary
				(w1).at<cv::Point2f>(i, j).y = scale*(b.at<float>(i, j) + c.at<float>(i, j)); //[b + c]
				(w2).at<cv::Point2f>(i, j).y = scale*(b.at<float>(i, j) - c.at<float>(i, j)); //[b - c]
			}
		}
	}

	void dtwaverec2(cv::Mat &Z, int L, std::vector<int> &S1, std::vector<int> &S2, std::vector<float> &C){
		int finish = 0, start = 0;
		int a = (S1).size();
		int current_level = a, index;
		cv::Mat ll, lh, hh, hl;

		while (current_level >= 2){
			int sx = (S1)[a - current_level];
			int sxx = sx*sx;
			start = 0 + finish;

			if (current_level == a){
				ll = cv::Mat(sx, sx, CV_32F);
				index = start;
				for (int j = 0; j < sx; j++){
					for (int i = 0; i < sx; i++){
						ll.at<float>(i, j) = (C)[index];
						index++;
					}
				}
				start += sxx;
			}
			else{
				ll = (Z).clone();
			}

			//Reshape other bands
			lh = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					lh.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			start += sxx;

			hl = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					hl.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			start += sxx;

			hh = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					hh.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			//Update finish
			finish = start + sxx;

			
			//Do even Qshift filters on columns.
			cv::Mat y1, y2, tm1, tm2, tm3, tm4;
			colifilt(ll, g0b, g0a, tm1);
			colifilt(lh, g1b, g1a, tm2);
			colifilt(hl, g0b, g0a, tm3);
			colifilt(hh, g1b, g1a, tm4);
			
			//std::cout << lh << std::endl;

			y1 = tm1 + tm2;
			y2 = tm3 + tm4;

			//Transpose
			cv::Mat y1t, y2t, z1, z2, zs;
			cv::transpose(y1, y1t);
			cv::transpose(y2, y2t);

			//Do even Qshift filters on rows
			colifilt(y1t, g0b, g0a, z1);
			colifilt(y2t, g1b, g1a, z2);

			zs = z1 + z2;
			cv::transpose(zs, (Z));
			current_level--;

			//std::cout << (*Z) << std::endl;
		}

		if (current_level == 1){
			int sx = (S1)[a - current_level];
			int sxx = sx*sx;
			start = 0 + finish;

			if (current_level == a){
				ll = cv::Mat(sx, sx, CV_32F);
				index = start;
				for (int j = 0; j < sx; j++){
					for (int i = 0; i < sx; i++){
						ll.at<float>(i, j) = (C)[index];
						index++;
					}
				}
				start += sxx;
			}
			else{
				ll = (Z).clone();
			}

			//Reshape other bands
			lh = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					lh.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			start += sxx;

			hl = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					hl.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			start += sxx;

			hh = cv::Mat(sx, sx, CV_32F);
			index = start;
			for (int j = 0; j < sx; j++){
				for (int i = 0; i < sx; i++){
					hh.at<float>(i, j) = (C)[index];
					index++;
				}
			}
			//Update finish
			finish = start + sxx - 1;

			//Do even Qshift filters on columns.
			cv::Mat y1, y2, tm1, tm2, tm3, tm4;
			colfilter(ll, g0o, tm1);
			colfilter(lh, g1o, tm2);
			colfilter(hl, g0o, tm3);
			colfilter(hh, g1o, tm4);

			y1 = tm1 + tm2;
			y2 = tm3 + tm4;

			//Transpose
			cv::Mat y1t, y2t, z1, z2, zs;
			cv::transpose(y1, y1t);
			cv::transpose(y2, y2t);

			//Do even Qshift filters on rows
			colfilter(y1t, g0o, z1);
			colfilter(y2t, g1o, z2);

			zs = z1 + z2;
			cv::transpose(zs, (Z));

			current_level--;
		}
	}

	void dtwavedec2(cv::Mat &img, int L, std::vector<int> &S1, std::vector<int> &S2, std::vector<float> &C){

		//Initialize
		C = std::vector<float>();
		(S1) = std::vector<int>(0);
		(S2) = std::vector<int>(0);
		
		if (L == 0)
			return;

		(C) = std::vector<float>(4 * (img).rows * (img).cols);
		int index = (C).size() - 1;

		cv::Mat Lo, Hi, LoLo, LoHi1, HiLo1, HiHi1, LoHi, HiLo, HiHi;
		if (L >= 1){
			cv::Mat Lot, Hit, LoLot, LoHi1t, HiLo1t, HiHi1t;
			// Do odd top - level filters on rows.
			colfilter(img, h0o, Lot);
			colfilter(img, h1o, Hit);

			//Transpose
			cv::transpose(Lot, Lo);
			cv::transpose(Hit, Hi);

			// Do odd top - level filters on columns.
			colfilter(Lo, h0o, LoLot);
			colfilter(Hi, h0o, LoHi1t);
			colfilter(Lo, h1o, HiLo1t);
			colfilter(Hi, h1o, HiHi1t);

			//Transpose
			cv::transpose(LoLot, LoLo);
			cv::transpose(LoHi1t, LoHi1);
			cv::transpose(HiLo1t, HiLo1);
			cv::transpose(HiHi1t, HiHi1);

			(S1).insert((S1).begin(), LoLo.rows);
			(S2).insert((S2).begin(), LoLo.cols);

			//Push back values to C accordinf to:  C = [ LoHi(:) ; HiLo(:); HiHi(:); C];
			for (int i = HiHi1.cols - 1; i >= 0; i--){
				for (int j = HiHi1.rows - 1; j >= 0; j--){
					(C)[index] = HiHi1.at<float>(j, i);
					index--;
				}
			}
			for (int i = HiLo1.cols - 1; i >= 0; i--){
				for (int j = HiLo1.rows - 1; j >= 0; j--){
					(C)[index] = HiLo1.at<float>(j, i);
					index--;
				}
			}
			for (int i = LoHi1.cols - 1; i >= 0; i--){
				for (int j = LoHi1.rows - 1; j >= 0; j--){
					(C)[index] = LoHi1.at<float>(j, i);
					index--;
				}
			}
		}

		if (L >= 2){
			for (int count = 2; count <= L; count++){
				cv::Mat Lot, Hit, LoLot, LoHit, HiLot, HiHit;

				// Do even Qshift filters on rows.
				coldfilt(LoLo, h0b, h0a, Lot);
				coldfilt(LoLo, h1b, h1a, Hit);

				//Transpose
				cv::transpose(Lot, Lo);
				cv::transpose(Hit, Hi);

				//Do even Qshift filters on columns.
				coldfilt(Lo, h0b, h0a, LoLot); // LoLo
				coldfilt(Hi, h0b, h0a, LoHit); // LoHi = > Horizontal
				coldfilt(Lo, h1b, h1a, HiLot); // HiLo = > Vertical
				coldfilt(Hi, h1b, h1a, HiHit); // HiHi = > Diagonal
				
				//Transpose
				cv::transpose(LoLot, LoLo);
				cv::transpose(LoHit, LoHi);
				cv::transpose(HiLot, HiLo);
				cv::transpose(HiHit, HiHi);

				//Push back values to C accordinf to:  C = [ LoHi(:) ; HiLo(:); HiHi(:); C];
				for (int i = HiHi.cols - 1; i >= 0; i--){
					for (int j = HiHi.rows - 1; j >= 0; j--){
						(C)[index] = HiHi.at<float>(j, i);
						index--;
					}
				}
				for (int i = HiLo.cols - 1; i >= 0; i--){
					for (int j = HiLo.rows - 1; j >= 0; j--){
						(C)[index] = HiLo.at<float>(j, i);
						index--;
					}
				}
				for (int i = LoHi.cols - 1; i >= 0; i--){
					for (int j = LoHi.rows - 1; j >= 0; j--){
						(C)[index] = LoHi.at<float>(j, i);
						index--;
					}
				}

				(S1).insert((S1).begin(), LoLo.rows);
				(S2).insert((S2).begin(), LoLo.cols);
			}
		}

		// Finally, add in LoLo and level 1 subbands.
		for (int i = LoLo.cols - 1; i >= 0; i--){
			for (int j = LoLo.rows - 1; j >= 0; j--){
				(C)[index] = LoLo.at<float>(j, i);
				index--;
			}
		}
		//At this point, index should be 0
	}

	void colfilter(cv::Mat &in, cv::Mat &filter, cv::Mat &out){
		//Extend with reflective padding
		cv::filter2D(in, out, CV_32F, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
	}

	void colifilt(cv::Mat &in, cv::Mat &ha, cv::Mat &hb, cv::Mat &out){
		//Initialize odd/even filters
		cv::Mat hae, hao, hbe, hbo;
		hae = cv::Mat((ha).rows / 2, 1, CV_32F, cv::Scalar(0));
		hao = cv::Mat((ha).rows / 2, 1, CV_32F, cv::Scalar(0));
		hbe = cv::Mat((hb).rows / 2, 1, CV_32F, cv::Scalar(0));
		hbo = cv::Mat((hb).rows / 2, 1, CV_32F, cv::Scalar(0));

		//Select odd/even filters
		int s = 0;
		for (int i = 0; i < (ha).rows; i += 2){
			hao.at<float>((ha).rows / 2 - s - 1, 0) = (ha).at<float>(i, 0);
			hbo.at<float>((ha).rows / 2 - s - 1, 0) = (hb).at<float>(i, 0);
			s++;
		}
		s = 0;
		for (int i = 1; i < (ha).rows; i += 2){
			hae.at<float>((ha).rows / 2 - s - 1, 0) = (ha).at<float>(i, 0);
			hbe.at<float>((ha).rows / 2 - s - 1, 0) = (hb).at<float>(i, 0);
			s++;
		}

		//Check low or high band
		double sum = 0;
		for (int i = 0; i < (ha).rows; i++){
			sum += (ha).at<float>(i, 0)*(hb).at<float>(i, 0);
		}

		//Create reflected margins
		int m = (ha).rows;
		int m2 = (ha).rows/2;
		cv::Mat refi((in).rows + 2 * m2, (in).cols, CV_32F);

		//Initialize
		for (int i = 0; i < (in).rows; i++){
			for (int j = 0; j < (in).cols; j++){
				refi.at<float>(i + m2, j) = (in).at<float>(i, j);
			}
		}
		//Add reflexive edges
		for (int i = 0; i < m2; i++){
			for (int j = 0; j < (in).cols; j++){
				refi.at<float>(m2 - i - 1, j) = (in).at<float>(i, j);
				refi.at<float>((in).rows + m2 + i, j) = (in).at<float>((in).rows - i - 1, j);
			}
		}

		//Get 4 subselections for filtering
		cv::Mat sh0, sh1, sh2, sh3;
		int t = ((in).rows + m - 1 - 3) / 2 + 1;
		sh0 = cv::Mat(t, (in).cols, CV_32F);
		sh1 = cv::Mat(t, (in).cols, CV_32F);
		sh2 = cv::Mat(t, (in).cols, CV_32F);
		sh3 = cv::Mat(t, (in).cols, CV_32F);

		for (int i = 0; i < t; i++){
			for (int j = 0; j < (in).cols; j++){
				if (sum > 0){
					sh0.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1, j);
					sh1.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1 + 1, j);
					sh2.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1, j);
					sh3.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1 + 1, j);
				}
				else{
					sh0.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1 + 1, j);
					sh1.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1, j);
					sh2.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1 + 1, j);
					sh3.at<float>(i, j) = refi.at<float>((i + 1) * 2 - 1, j);
				}
			}
		}

		//Filter subselections
		cv::Mat fsh0, fsh1, fsh2, fsh3;
		cv::filter2D(sh0, fsh0, CV_32F, hao, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh1, fsh1, CV_32F, hbo, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh2, fsh2, CV_32F, hae, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh3, fsh3, CV_32F, hbe, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Copy everything to output
		(out) = cv::Mat((in).rows * 2, (in).cols, CV_32F);

		for (int i = 0; i < (in).rows/2; i++){
			for (int j = 0; j < (in).cols; j++){
				(out).at<float>(i * 4, j) = fsh0.at<float>(i + (int)std::floor(m / 4), j);
				(out).at<float>(i * 4 + 1, j) = fsh1.at<float>(i + (int)std::floor(m / 4), j);
				(out).at<float>(i * 4 + 2, j) = fsh2.at<float>(i + (int)std::floor(m / 4), j);
				(out).at<float>(i * 4 + 3, j) = fsh3.at<float>(i + (int)std::floor(m / 4), j);
			}
		}

		//Extend with reflective padding
		//cv::filter2D(*in, *out, CV_32F, *filter, cv::Point(-1, -1), cv::BORDER_REFLECT);
	}

	void coldfilt(cv::Mat &in, cv::Mat &ha, cv::Mat &hb, cv::Mat &out){
		//Initialize odd/even filters
		cv::Mat hae, hao, hbe, hbo;
		hae = cv::Mat((ha).rows / 2, 1, CV_32F, cv::Scalar(0));
		hao = cv::Mat((ha).rows / 2, 1, CV_32F, cv::Scalar(0));
		hbe = cv::Mat((hb).rows / 2, 1, CV_32F, cv::Scalar(0));
		hbo = cv::Mat((hb).rows / 2, 1, CV_32F, cv::Scalar(0));

		//Select odd/even filters
		int s = 0;
		for (int i = 0; i < (ha).rows; i += 2){
			hao.at<float>((ha).rows/2-s-1, 0) = (ha).at<float>(i, 0);
			hbo.at<float>((ha).rows / 2 - s - 1, 0) = (hb).at<float>(i, 0);
			s++;
		}
		s = 0;
		for (int i = 1; i < (ha).rows; i += 2){
			hae.at<float>((ha).rows / 2 - s - 1, 0) = (ha).at<float>(i, 0);
			hbe.at<float>((ha).rows / 2 - s - 1, 0) = (hb).at<float>(i, 0);
			s++;
		}

		//Check low or high band
		double sum = 0;
		for (int i = 0; i < (ha).rows; i++){
			sum += (ha).at<float>(i, 0)*(hb).at<float>(i, 0);
		}

		//Create reflected margins
		int m = (ha).rows;
		cv::Mat refi((in).rows+2*m, (in).cols, CV_32F);

		//Initialize
		for (int i = 0; i < (in).rows; i++){
			for (int j = 0; j < (in).cols; j++){
				refi.at<float>(i+m, j) = (in).at<float>(i, j);
			}
		}
		//Add reflexive edges
		for (int i = 0; i < m; i++){
			for (int j = 0; j < (in).cols; j++){
				refi.at<float>(m - i - 1, j) = (in).at<float>(i, j);
				refi.at<float>((in).rows + m + i, j) = (in).at<float>((in).rows - i - 1, j);
			}
		}

		//Get 4 subselections for filtering
		cv::Mat sh0, sh1, sh2, sh3;
		int t = ((in).rows + 2 * m - 2 - 6) / 4 + 1;
		sh0 = cv::Mat(t, (in).cols, CV_32F);
		sh1 = cv::Mat(t, (in).cols, CV_32F);
		sh2 = cv::Mat(t, (in).cols, CV_32F);
		sh3 = cv::Mat(t, (in).cols, CV_32F);

		for (int i = 0; i < t; i++){
			for (int j = 0; j < (in).cols; j++){
				sh0.at<float>(i, j) = refi.at<float>((i+1) * 4 + 2 - 1, j);
				sh1.at<float>(i, j) = refi.at<float>((i + 1) * 4 + 2 - 1 - 1, j);
				sh2.at<float>(i, j) = refi.at<float>((i + 1) * 4 + 2 - 1 - 2, j);
				sh3.at<float>(i, j) = refi.at<float>((i + 1) * 4 + 2 - 1 - 3, j);
			}
		}

		//Filter subselections
		cv::Mat fsh0, fsh1, fsh2, fsh3;
		cv::filter2D(sh0, fsh0, CV_32F, hbo, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh1, fsh1, CV_32F, hao, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh2, fsh2, CV_32F, hbe, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(sh3, fsh3, CV_32F, hae, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		//Copy everything to output
		(out) = cv::Mat((in).rows / 2, (in).cols, CV_32F);

		if (sum > 0){
			for (int i = 0; i < (in).rows / 4; i++){
				for (int j = 0; j < (in).cols; j++){
						(out).at<float>(i * 2, j) = fsh1.at<float>((int)std::floor(m / 4) + i, j) + fsh3.at<float>((int)std::floor(m / 4) + i, j);
						(out).at<float>(i * 2 + 1, j) = fsh0.at<float>((int)std::floor(m / 4) + i, j) + fsh2.at<float>((int)std::floor(m / 4) + i, j);
				}
			}
		}
		else{
			for (int i = 0; i < (in).rows / 4; i++){
				for (int j = 0; j < (in).cols; j++){
					(out).at<float>(i * 2 + 1, j) = fsh1.at<float>((int)std::floor(m / 4) + i, j) + fsh3.at<float>((int)std::floor(m / 4) + i, j);
					(out).at<float>(i * 2, j) = fsh0.at<float>((int)std::floor(m / 4) + i, j) + fsh2.at<float>((int)std::floor(m / 4) + i, j);
				}
			}
		}

		//Extend with reflective padding
		//cv::filter2D(*in, *out, CV_32F, *filter, cv::Point(-1, -1), cv::BORDER_REFLECT);
	}

	//Initialize all filters here
	//Decomposition filters
	cv::Mat h0o = (cv::Mat_<float>(5, 1) << -0.05, 0.25, 0.6, 0.25, -0.05);
	cv::Mat h0a = (cv::Mat_<float>(10, 1) << 0.0511304052838317,
		-0.0139753702468888,
		-0.109836051665971,
		0.263839561058938,
		0.766628467793037,
		0.563655710127052,
		0.000873622695217097,
		-0.100231219507476,
		-0.00168968127252815,
		-0.00618188189211644);
	cv::Mat h0b = (cv::Mat_<float>(10, 1) << -0.00618188189211644,
		-0.00168968127252815,
		-0.100231219507476,
		0.000873622695217097,
		0.563655710127052,
		0.766628467793037,
		0.263839561058938,
		-0.109836051665971,
		-0.0139753702468888,
		0.0511304052838317);
	cv::Mat h1o = (cv::Mat_<float>(7, 1) << 0.0107142857142857,
		-0.0535714285714286,
		-0.260714285714286,
		0.607142857142857,
		-0.260714285714286,
		-0.0535714285714286,
		0.0107142857142857);
	cv::Mat h1a = (cv::Mat_<float>(10, 1) << -0.00618188189211644,
		0.00168968127252815,
		-0.100231219507476,
		-0.000873622695217097,
		0.563655710127052,
		-0.766628467793037,
		0.263839561058938,
		0.109836051665971,
		-0.0139753702468888,
		-0.0511304052838317);
	cv::Mat h1b = (cv::Mat_<float>(10, 1) << -0.0511304052838317,
		-0.0139753702468888,
		0.109836051665971,
		0.263839561058938,
		-0.766628467793037,
		0.563655710127052,
		-0.000873622695217097,
		-0.100231219507476,
		0.00168968127252815,
		-0.00618188189211644);
	//Reconstruction filters
	cv::Mat g0a = (cv::Mat_<float>(10, 1) << -0.00618188189211644,
		-0.00168968127252815,
		-0.100231219507476,
		0.000873622695217097,
		0.563655710127052,
		0.766628467793037,
		0.263839561058938,
		-0.109836051665971,
		-0.0139753702468888,
		0.0511304052838317);
	cv::Mat g0b = (cv::Mat_<float>(10, 1) << 0.0511304052838317,
		-0.0139753702468888,
		-0.109836051665971,
		0.263839561058938,
		0.766628467793037,
		0.563655710127052,
		0.000873622695217097,
		-0.100231219507476,
		-0.00168968127252815,
		-0.00618188189211644);
	cv::Mat g0o = (cv::Mat_<float>(7, 1) << -0.0107142857142857,
		-0.0535714285714286,
		0.260714285714286,
		0.607142857142857,
		0.260714285714286,
		-0.0535714285714286,
		-0.0107142857142857);
	cv::Mat g1a = (cv::Mat_<float>(10, 1) << -0.0511304052838317,
		-0.0139753702468888,
		0.109836051665971,
		0.263839561058938,
		-0.766628467793037,
		0.563655710127052,
		-0.000873622695217097,
		-0.100231219507476,
		0.00168968127252815,
		-0.00618188189211644);
	cv::Mat g1b = (cv::Mat_<float>(10, 1) << -0.00618188189211644,
		0.00168968127252815,
		-0.100231219507476,
		-0.000873622695217097,
		0.563655710127052,
		-0.766628467793037,
		0.263839561058938,
		0.109836051665971,
		-0.0139753702468888,
		-0.0511304052838317);
	cv::Mat g1o = (cv::Mat_<float>(5, 1) << -0.0500000000000000,
		-0.250000000000000,
		0.600000000000000,
		-0.250000000000000,
		-0.0500000000000000);
}