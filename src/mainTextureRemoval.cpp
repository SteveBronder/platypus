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
#include "CradleFunctions.h"
#include "TextureRemoval.h"
#include "FFST.h"
#include <opencv2/flann.hpp>

/**
* This piece of demo code demonstrates wood-grain separation for an image that already went through cradle removal.
* (See mainCradleRemoval.cpp for more info on how to do that.)
* If the image specified does not exist, the code results in an error.
*
* Parameters:
*   arg[1]				path+filename of input image to be processed
*   arg[2] & arg[3]		X and Y center coordinates of the 464x464 block where wood-grain separation will be executed
*
* Output:
*  "inc.png"		a 464x464 crop of the original
*  "out.png"		result after cradle removal, without wood-grain separation
*  "out2.png"		result after cradle removal and wood-grain separation
*  "cradle.png"		cradle component, without wood-grain separation
*  "cradle2.png"	cradle component, after wood-grain separation
**/

#define OVERLAP 48			// OVERLAP/2 is used for border extension to avoid edge artifacts after separation
#define N 512				//Image block width (after OVERLAP is added)
#define M 512				//Image block height (after OVERLAP is added)
#define SN 4				//Sub-sampling factor for rows
#define SM 4				//Sub-sampling factor for columns
#define PSN 9				//Sub-sampling factor for rows for clustering
#define PSM 9				//Sub-sampling factor for columns for clustering
#define TARGET_DIM 26		//Total non-zero coefficients in shearlet decomposition
#define NR_NEIGHBOURS 5		//Nr neighbors for Nearest-neighbour (NN) search
#define MAX_SAMPLES 1000	//Max samples used for training statistical model

//Shearlet decomposition horizontal/vertical angle parameters
int target_v[] = { 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 };
int target_h[] = { 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

int main(int argc, char** argv)
{
	cv::Mat img, nointensity, mask_full, out, tmp;
	std::string filename = argv[1];

	//Read in images/files from the first step + conversions to float
	//Original image file
	tmp = cv::imread(filename + "_original.png", cv::IMREAD_GRAYSCALE);
	img = cv::Mat(tmp.rows, tmp.cols, CV_32F);
	//Conversion to float
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			img.at<float>(i, j) = (float)tmp.at<uchar>(i, j);
		}
	}

	//Mask file
	mask_full = cv::imread(filename + "_mask.png", cv::IMREAD_GRAYSCALE);

	//Intensity removed image file
	tmp = cv::imread(filename + "_nointensity.png", cv::IMREAD_GRAYSCALE);
	nointensity = cv::Mat(tmp.rows, tmp.cols, CV_32F);
	//Conversion to float
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			nointensity.at<float>(i, j) = (float)tmp.at<uchar>(i, j);
		}
	}

	//Marked segment file
	CradleFunctions::MarkedSegments ms = CradleFunctions::readMarkedSegmentsFile(filename + ".msf");

	//Crop image block according to user input
	int cy = std::atoi(argv[2]);
	int cx = std::atoi(argv[3]);

	int lx = cx - (M - OVERLAP) / 2;
	int ly = cy - (N - OVERLAP) / 2;

	//Make sure it is within bounds
	if (lx < 0){
		lx = 0;
	}
	if (ly < 0){
		ly = 0;
	}

	int ley = std::min(ly + N - OVERLAP, img.cols);
	int lex = std::min(lx + M - OVERLAP, img.rows);

	//Take crop
	cv::Mat input_crop = img(cv::Range(lx, lex), cv::Range(ly, ley));
	cv::Mat nointensity_crop = nointensity(cv::Range(lx, lex), cv::Range(ly, ley));
	cv::Mat cradle_crop = input_crop - nointensity_crop;
	cv::Mat mask_crop = mask_full(cv::Range(lx, lex), cv::Range(ly, ley));

	/*
	NOTE: To speed things things up in this demo, the mask / cradle descriptor structure MarkedSegments ms
	is altered in this application.We crop the piece_mask part to match the selected region and unite
	all horizontal / vertical pieces into one single descriptor.This way, the algorithm only learns only
	two separation models, a horizontal and a vertical one.This little 'trick' speeds up the separation
	considerably without resulting in visual artifacts on a 464x464 selection, but would fail miserably
	if applied on large, high - resolution x - rays imageswhere the wood - grain varies from cradle piece 
    to cradle piece.
	*/

	ms.piece_mask = ms.piece_mask(cv::Range(lx, lex), cv::Range(ly, ley));
	std::vector<int> hpieces;
	std::vector<int> vpieces;

	//Put all segment IDs corresponding to a horizontal/vertical cradle piece in one single vector
	for (int i = 0; i < ms.pieceIDh.size(); i++){
		for (int j = 0; j < ms.pieceIDh[i].size(); j++){
			int match = 0;
			for (int k = 0; k < hpieces.size(); k++){
				if (ms.pieceIDh[i][j] == hpieces[k]){
					match = 1;
					break;
				}
			}
			if (match == 0){
				//Add identifier to hpieces
				hpieces.push_back(ms.pieceIDh[i][j]);
			}
		}
	}
	for (int i = 0; i < ms.pieceIDv.size(); i++){
		for (int j = 0; j < ms.pieceIDv[i].size(); j++){
			int match = 0;
			for (int k = 0; k < vpieces.size(); k++){
				if (ms.pieceIDv[i][j] == vpieces[k]){
					match = 1;
					break;
				}
			}
			if (match == 0){
				//Add identifier to hpieces
				vpieces.push_back(ms.pieceIDv[i][j]);
			}
		}
	}
	ms.pieceIDh.clear();
	ms.pieceIDh.push_back(hpieces);
	ms.pieceIDv.clear();
	ms.pieceIDv.push_back(vpieces);

	//Cropped input
	cv::imwrite("inc.png", input_crop);
	cv::imwrite("out.png", nointensity_crop);
	cv::imwrite("cradle.png", cradle_crop);

	//Create borders for image
	cv::Mat in, mask, piecemark;
	cv::copyMakeBorder(nointensity_crop, in, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, cv::BORDER_REFLECT);

	//Create borders for mask
	cv::copyMakeBorder(mask_crop, mask, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, cv::BORDER_REFLECT);

	//Create border for piece mask
	cv::copyMakeBorder(ms.piece_mask, piecemark, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, OVERLAP / 2, cv::BORDER_REFLECT);

	//Create (huge) arrays of cradle/non-cradle sampled coefficients
	std::vector<std::vector<float>> full_samples(in.rows * in.cols);
	std::vector<int> sample_index(in.rows * in.cols);

	//Index for sample position within array
	int fsample_pos = 0;

	//Type of sampled piece (horizontal/vertical/cross section)
	std::vector<int> sample_type(ms.pieceIDh.size() + ms.pieceIDv.size() + 2);

	//Mark cradle directions
	for (int i = 0; i < ms.pieceIDh.size(); i++){
		sample_type[i + 2] = CradleFunctions::HORIZONTAL_DIR;
	}
	for (int i = 0; i < ms.pieceIDv.size(); i++){
		sample_type[i + 2 + ms.pieceIDh.size()] = CradleFunctions::VERTICAL_DIR;
	}

	int sx, sy, ex, ey;
	sx = sy = 0;
	ex = in.rows;
	ey = in.cols;

	//Apply fast finite shearlet transform on texture component
	std::vector<cv::Mat> coeffs;
	coeffs = FFST::shearletTransformSpect(in);

	//Add points to training set
	for (int i = 0; i < ex - sx; i += SN){
		for (int j = 0; j < ey - sy; j += SM){

			ushort pi = piecemark.at<ushort>(i, j) + 1;	//Index of the piece
			int coeff_size = TARGET_DIM;

			if (pi > 1){

				//Find horizontal cradle containing this segment (if any)
				int hi = -1;
				for (int temp1 = 0; temp1 < ms.pieceIDh.size(); temp1++){
					for (int temp2 = 0; temp2 < ms.pieceIDh[temp1].size(); temp2++){
						if (ms.pieceIDh[temp1][temp2] == pi - 1){
							hi = temp1 + 2;
						}
					}
				}

				if (hi != -1){
					//Add sample to horizontal piece
					full_samples[fsample_pos] = std::vector<float>(coeff_size);
					sample_index[fsample_pos] = hi;
					sample_type[hi] = CradleFunctions::HORIZONTAL_DIR;

					//Fill up sample - horizontal
					int lindex = 0;
					for (int l = 0; l < 61; l++) if (target_h[l] == 1){
						full_samples[fsample_pos][lindex] = coeffs[l].at<float>(i, j);
						lindex++;
					}
					fsample_pos++;
				}

				//Find vertical cradle containing this segment (if any)
				int vi = -1;
				for (int temp1 = 0; temp1 < ms.pieceIDv.size(); temp1++){
					for (int temp2 = 0; temp2 < ms.pieceIDv[temp1].size(); temp2++){
						if (ms.pieceIDv[temp1][temp2] == pi - 1){
							vi = temp1 + 2 + ms.pieceIDh.size();
						}
					}
				}
				if (vi != -1){
					//Add sample to vertical piece
					full_samples[fsample_pos] = std::vector<float>(coeff_size);
					sample_index[fsample_pos] = vi;
					sample_type[vi] = CradleFunctions::VERTICAL_DIR;

					///Fill up sample - vertical
					int lindex = 0;
					for (int l = 0; l < 61; l++) if (target_v[l] == 1){
						full_samples[fsample_pos][lindex] = coeffs[l].at<float>(i, j);
						lindex++;
					}
					fsample_pos++;
				}
			}
			else{
				//No horizontal or vertical mask piece present
				pi = 0;	//Horizontal non-cradle index
				int coeff_size = TARGET_DIM;
				full_samples[fsample_pos] = std::vector<float>(coeff_size);
				sample_index[fsample_pos] = pi;

				//Fill up sample - horizontal
				int lindex = 0;
				for (int l = 0; l < 61; l++) if (target_h[l] == 1){
					full_samples[fsample_pos][lindex] = coeffs[l].at<float>(i, j);
					lindex++;
				}
				fsample_pos++;

				pi = 1;	//Vertical non-cradle index
				full_samples[fsample_pos] = std::vector<float>(coeff_size);
				sample_index[fsample_pos] = pi;

				//Fill up sample - vertical
				lindex = 0;
				for (int l = 0; l < 61; l++) if (target_v[l] == 1){
					full_samples[fsample_pos][lindex] = coeffs[l].at<float>(i, j);
					lindex++;
				}
				fsample_pos++;
			}
		}
	}

	//Sub-sampling
	full_samples.resize(fsample_pos);
	std::vector<int> sample_pos(sample_type.size());
	std::vector<std::vector<std::vector<float>>> sample_select(sample_type.size());

	//Randomly select samples to reduce computation time
	for (int i = 0; i < sample_select.size(); i++){
		int local_pos = 0;
		std::vector<std::vector<float>> local(full_samples.size());
		//Find all samples with corresponding number
		for (int j = 0; j < full_samples.size(); j++){
			if (sample_index[j] == i){
				//Copy sample to local selection
				local[local_pos] = std::vector<float>(full_samples[j].size());
				for (int k = 0; k < full_samples[j].size(); k++){
					local[local_pos][k] = full_samples[j][k];
				}
				local_pos++;
			}
		}

		//Resize
		local.resize(local_pos);

		//Sample randomly
		TextureRemoval::sampleDataset(local, sample_select[i], MAX_SAMPLES);
	}

	//Drop global selection of coefficients to save memory
	full_samples.clear();
	sample_index.clear();

	//Normalize non-cradled components
	std::vector<float> mean_h, mean_v, var_h, var_v;
	TextureRemoval::normalizeNonCradle(sample_select[0], mean_h, var_h);	//Get normalization of horizontal non-cradle samples
	TextureRemoval::normalizeNonCradle(sample_select[1], mean_v, var_v);	//Get normalization of vertical non-cradle samples

	//Set of arrays used for separation code
	TextureRemoval::cradle_model_fitting model;
	std::vector<std::vector<float>> ncdata;
	std::vector<std::vector<float>> clusters_diffs;
	std::vector<std::vector<float>> diffs;

	//Train horizontal/vertical cradle models for separation
	for (int mod_sel = 2; mod_sel < sample_select.size(); mod_sel++){

		if (sample_select[mod_sel].size() != 0){
			//Non-cradle data samples

			//Normalize the data & choose reference non-cradle data set
			if (sample_type[mod_sel] == CradleFunctions::HORIZONTAL_DIR){
				TextureRemoval::normalizeSamples(sample_select[mod_sel], mean_h, var_h);
				ncdata = sample_select[0];
			}
			else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
				TextureRemoval::normalizeSamples(sample_select[mod_sel], mean_v, var_v);
				ncdata = sample_select[1];
			}
			else{
			}

			//Train the model
			model = TextureRemoval::gibbsSampling(sample_select[mod_sel], ncdata);

			//Use this to store samples
			std::vector<std::vector<float>> clusters(N * M / PSN / PSM * 2);
			int sample_pos = 0;

			//Subsample coefficients for clustering
			for (int i = 0; i < ex - sx; i += PSN){
				for (int j = 0; j < ey - sy; j += PSM){

					int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
					int coeff_size = TARGET_DIM;

					if (pi > 1){
						int ci;
						bool partOfCradle = false;

						if (mod_sel >= 2 + ms.pieceIDh.size()){
							//vertical cradle piece
							ci = mod_sel - 2 - ms.pieceIDh.size();

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDv[ci].size(); temp++){
								if (ms.pieceIDv[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply vertical separation
							if (partOfCradle){
								clusters[sample_pos] = std::vector<float>(coeff_size);

								//Fill up sample - vertical
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_v[l] == 1){
									clusters[sample_pos][lindex] = coeffs[l].at<float>(i, j);
									lindex++;
								}
								sample_pos++;
							}

						}
						else{
							//horizontal cradle piece
							ci = mod_sel - 2;

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDh[ci].size(); temp++){
								if (ms.pieceIDh[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply horizontal separation
							if (partOfCradle){
								clusters[sample_pos] = std::vector<float>(coeff_size);

								//Fill up samle - horizontal
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_h[l] == 1){
									clusters[sample_pos][lindex] = coeffs[l].at<float>(i, j);
									lindex++;
								}
								sample_pos++;
							}
						}
					}
				}
			}

			//Drop unused elements
			clusters.resize(sample_pos);
			bool clustering;
			cv::flann::GenericIndex< cvflann::L2<float> > *kdTree; // The flann KD searching tree

			cv::Mat clusters_mat(clusters.size(), clusters[0].size(), CV_32F);
			if (clusters.size() > 0){
				for (int i = 0; i < clusters_mat.rows; i++){
					for (int j = 0; j < clusters_mat.cols; j++){
						clusters_mat.at<float>(i, j) = clusters[i][j];
					}
				}
				clustering = true;
			}
			else{
				clustering = false;
			}

			//Create KD Tree
			kdTree = new cv::flann::GenericIndex< cvflann::L2<float> >(clusters_mat, cvflann::KDTreeIndexParams(4));

			post_inference(model, clusters, ncdata, clusters_diffs);

			//Look up all coefficients
			std::vector<std::vector<float>> samples(N * M);
			sample_pos = 0;
			for (int i = 0; i < ex - sx; i++){
				for (int j = 0; j < ey - sy; j++){

					int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
					int coeff_size = TARGET_DIM;

					if (pi > 1){
						int ci;
						bool partOfCradle = false;

						if (mod_sel >= 2 + ms.pieceIDh.size()){
							//It's a vertical cradle piece
							ci = mod_sel - 2 - ms.pieceIDh.size();

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDv[ci].size(); temp++){
								if (ms.pieceIDv[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply vertical separation
							if (partOfCradle){
								samples[sample_pos] = std::vector<float>(coeff_size);

								//Fill up sample - vertical
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_v[l] == 1){
									samples[sample_pos][lindex] = coeffs[l].at<float>(i, j);
									lindex++;
								}
								sample_pos++;
							}

						}
						else{
							//It's a horizontal cradle piece
							ci = mod_sel - 2;

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDh[ci].size(); temp++){
								if (ms.pieceIDh[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply horizontal separation
							if (partOfCradle){
								samples[sample_pos] = std::vector<float>(coeff_size);

								//Fill up samle - horizontal
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_h[l] == 1){
									samples[sample_pos][lindex] = coeffs[l].at<float>(i, j);
									lindex++;
								}
								sample_pos++;
							}
						}
					}
				}
			}
			//Drop unused elements
			samples.resize(sample_pos);

			//Normalize the data & choose reference non-cradle data set
			if (samples.size() != 0){
				if (sample_type[mod_sel] == CradleFunctions::HORIZONTAL_DIR){
					TextureRemoval::normalizeSamples(samples, mean_h, var_h);
					ncdata = sample_select[0];
				}
				else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
					TextureRemoval::normalizeSamples(samples, mean_v, var_v);
					ncdata = sample_select[1];
				}
				else{
					//Cross section
				}
			}

			if (samples.size() != 0){
				if (clustering){
					//Convert samples to a cv::Mat
					cv::Mat samples_mat(samples.size(), samples[0].size(), CV_32F);
					for (int a = 0; a < samples.size(); a++){
						for (int b = 0; b < samples[0].size(); b++){
							samples_mat.at<float>(a, b) = samples[a][b];
						}
					}

					//Use clustering for separation
					diffs = std::vector<std::vector<float>>(samples.size());
					for (int i = 0; i < diffs.size(); i++){
						diffs[i] = std::vector<float>(samples[i].size());
					}

					cv::Mat neighborsIdx; //This mat will contain the index of nearest neighbour as returned by Kd-tree
					cv::Mat distances; //In this mat Kd-Tree return the distances for each nearest neighbour

					//Initialize structures
					neighborsIdx.create(cv::Size(NR_NEIGHBOURS, samples_mat.rows), CV_32SC1);
					distances.create(cv::Size(NR_NEIGHBOURS, samples_mat.rows), CV_32FC1);

					//Run KD search
					kdTree->knnSearch(samples_mat, neighborsIdx, distances, NR_NEIGHBOURS, cvflann::SearchParams(8));

					//Use result for separation
					for (int i = 0; i < samples.size(); i++){

						//Get weights
						std::vector<float> weights(NR_NEIGHBOURS);
						float sumweight = 0;
						for (int j = 0; j < NR_NEIGHBOURS; j++){
							weights[j] = 1.0 / (1 + distances.at<float>(i, j));
							sumweight += weights[j];
						}

						//Get interpolated decomposition
						for (int k = 0; k < NR_NEIGHBOURS; k++){
							float cweight = weights[k] / sumweight;
							for (int j = 0; j < samples[i].size(); j++){
								diffs[i][j] += clusters_diffs[neighborsIdx.at<int>(i, k)][j] * cweight;
							}
						}
					}
				}
				else{
					//Do the full post-inference
					post_inference(model, samples, ncdata, diffs);
				}
			}

			//Unnormalize separation data
			if (samples.size() != 0){
				if (sample_type[mod_sel] == CradleFunctions::HORIZONTAL_DIR){
					TextureRemoval::unNormalizeSamples(diffs, mean_h, var_h);
				}
				else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
					TextureRemoval::unNormalizeSamples(diffs, mean_v, var_v);
				}
				else{
					//Cross section
					TextureRemoval::unNormalizeSamplesCrossSection(diffs, mean_h, var_h, mean_v, var_v);
				}
			}

			//Reset index of sample_pos
			sample_pos = 0;

			//Apply separation to the decomposition coefficients
			for (int i = 0; i < ex - sx; i++){
				for (int j = 0; j < ey - sy; j++){

					int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
					int coeff_size = TARGET_DIM;

					if (pi > 1){
						int ci;
						bool partOfCradle = false;

						if (mod_sel >= 2 + ms.pieceIDh.size()){
							//It's a vertical cradle piece
							ci = mod_sel - 2 - ms.pieceIDh.size();

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDv[ci].size(); temp++){
								if (ms.pieceIDv[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply vertical separation
							if (partOfCradle){
								samples[sample_pos] = std::vector<float>(coeff_size);

								//Fill up sample - vertical
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_v[l] == 1){
									coeffs[l].at<float>(i, j) -= diffs[sample_pos][lindex];
									lindex++;
								}
								sample_pos++;
							}

						}
						else{
							//It's a horizontal cradle piece
							ci = mod_sel - 2;

							//Check if segment is part of the cradle
							for (int temp = 0; temp < ms.pieceIDh[ci].size(); temp++){
								if (ms.pieceIDh[ci][temp] == pi - 1)
									partOfCradle = true;
							}

							//Apply horizontal separation
							if (partOfCradle){
								samples[sample_pos] = std::vector<float>(coeff_size);

								//Fill up sample - horizontal
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_h[l] == 1){
									coeffs[l].at<float>(i, j) -= diffs[sample_pos][lindex];
									lindex++;
								}
								sample_pos++;
							}
						}
					}
				}
			}

			//Reconstruct block
			TextureRemoval::reconstructBlock(in, coeffs, sx, sy, sx, sy, ex, ey);
		}
	}
	out = in;

	//Remove padding
	out = out(cv::Range(OVERLAP / 2, N - OVERLAP / 2), cv::Range(OVERLAP / 2, M - OVERLAP / 2));
	out = out(cv::Range(0, input_crop.rows), cv::Range(0, input_crop.cols));
	cv::imwrite("out2.png", out);
	cv::imwrite("cradle2.png", input_crop - out);
}