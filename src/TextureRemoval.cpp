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

#include <platypus/TextureRemoval.h>
#include <platypus/CradleFunctions.h>
#include <platypus/MCA.h>
#include <platypus/FFST.h>
#include <opencv2/flann.hpp>
#include <random>

#define PI 3.1415927

/**
* Collection of all functions that make texture separation, and in particular, wood grain removal possible.
* The main function is textureRemove(), that applies an unsupervised learning algorithm to train the
* statistical model of the wood grain.
**/

namespace TextureRemoval{

	const int block_size = 512;		//Image block size for processing wood-grain separation
	const int overlap = 48;			//Amount of overlap between neighboring blocks
	const int SEED = 1;				//Constant seed for random number generators (to guarantee reproductibility)
	const int SN = 4;				//Sub-sampling factor for rows
	const int SM = 4;				//Sub-sampling factor for columns
	const int PSN = 9;				//Sub-sampling factor for rows for clustering
	const int PSM = 9;				//Sub-sampling factor for columns for clustering
	const int NR_NEIGHBOURS = 5;	//Nr neighbors for Nearest-neighbour (NN) search
	const int max_samples = 10000;	//Maximum number of samples to be processed for the post-inference algo

	//Shearlet decomposition horizontal/vertical angle parameters
	int target_v[] = { 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1 };
	int target_h[] = { 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	int target_dim = 26;

	//Used as buffer for file writing/reading
	float buffer[block_size*block_size];

	//Entry point to texture separation
	void textureRemove(
		cv::Mat &img,								//Input image for wood grain separation
		cv::Mat &mask_orig,							//Mask component, as returned by cradle removal step
		cv::Mat &out,								//Result image is stored here
		const CradleFunctions::MarkedSegments &ms	//Processing information, as returned by cradle removal step
	){

		//Create borders for image
		cv::Mat in, mask, piecemark;
		cv::copyMakeBorder(img, in, overlap / 2, overlap / 2, overlap / 2, overlap / 2, cv::BORDER_REFLECT);

		//Create borders for mask
		cv::copyMakeBorder(mask_orig, mask, overlap / 2, overlap / 2, overlap / 2, overlap / 2, cv::BORDER_REFLECT);

		//Create border for piece mask
		cv::copyMakeBorder(ms.piece_mask, piecemark, overlap / 2, overlap / 2, overlap / 2, overlap / 2, cv::BORDER_REFLECT);

		int N = in.rows;
		int M = in.cols;

		//Dictionaries for texture/cartoon separation
		std::vector<int> dict(2);
		dict[0] = MCA::FDCT;
		dict[1] = MCA::DTWDC;

		int sx, sy, ex, ey;
		cv::Mat cartoon, texture;

		cartoon = cv::Mat(N, M, CV_32F, cv::Scalar(0));
		texture = cv::Mat(N, M, CV_32F, cv::Scalar(0));

		//Parameters used for progress measuring
		int nr_blocks = (N / (block_size - overlap) + 1) * (M / (block_size - overlap) + 1);	//Approximate number of blocks in the image
		int processed = 0;
		int tot_progress = 10 + 1 + ms.pieceIDh.size() + ms.pieceIDv.size();	//10 for MCA, 1 for sampling globally, 1 for each H/V piece
		bool canceled = false;

		//Store integer coordinates of blocks
		std::vector<std::vector<int>> coords;
		sx = sy = 0;

		//Generate coordinates of blocks
		while (sx < N){
			ex = sx + block_size;
			ey = sy + block_size;

			//Do texture/cartoon decomposition
			if (ex > N) ex = N;
			if (ey > M) ey = M;

			//Store coordinates of current block
			std::vector<int> vect(4);
			vect[0] = sx;
			vect[1] = sy;
			vect[2] = ex;
			vect[3] = ey;
			coords.push_back(vect);

			if (ey == M){
				sx = sx + block_size;
				if (sx < N)
					sx -= overlap;
				sy = 0;
			}
			else{
				sy = sy + block_size - overlap;
			}
		}
    puts("TextureRemoval.cpp");
		//MCA decomposition
		#pragma omp parallel for
		for (int l = 0; l < coords.size(); l++){
			//Parallelized texture/cartoon separation loop

			#pragma omp flush (canceled)
			if (!canceled)
			{
				#pragma omp atomic
				processed = (int)(l * 10/ coords.size());

				// progress/abort
				#pragma omp critical
				if (!CradleFunctions::progress(processed, tot_progress))
					canceled = true;

				if (!canceled)
				{
					int sx = coords[l][0];
					int sy = coords[l][1];
					int ex = coords[l][2];
					int ey = coords[l][3];

					//Make tmp a small segment copy of input
					cv::Mat ctn, txt;
					cv::Mat tmp(ex - sx, ey - sy, CV_32F);
					for (int i = sx; i < ex; i++){
						for (int j = sy; j < ey; j++){
							tmp.at<float>(i - sx, j - sy) = in.at<float>(i, j);
						}
					}
					MCA::MCA_Bcr(tmp, dict, txt, ctn);

					//Save out result
					int csx = sx, cex = ex, csy = sy, cey = ey;
					if (sx != 0) csx += overlap / 2;
					if (sy != 0) csy += overlap / 2;
					if (ex != N) cex -= overlap / 2;
					if (ey != M) cey -= overlap / 2;

					for (int i = csx; i < cex; i++){
						for (int j = csy; j < cey; j++){
							texture.at<float>(i, j) = txt.at<float>(i - sx, j - sy);
							cartoon.at<float>(i, j) = ctn.at<float>(i - sx, j - sy);
						}
					}
				}
			}
		}

		if (canceled)
			return;

		cartoon = in - texture;

		cv::Mat new_texture = cv::Mat(texture.rows, texture.cols, CV_32F, cv::Scalar(0));
		processed = 10;

		//Create huge arrays of cradle/non-cradle sampled coefficients
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

		//Mark all blocks as unused
		std::vector<std::vector<int>> block_used(ms.pieceIDh.size() + ms.pieceIDv.size() + 2);
		for (int i = 0; i < block_used.size(); i++){
			block_used[i] = std::vector<int>(coords.size());
		}

		//Sample non-cradle parts for horizontal/vertical separation
		for (int z = 0; z < coords.size(); z++){

			if (!canceled){
				// progress/abort
				#pragma omp critical
				if (!CradleFunctions::progress(processed, tot_progress))
					canceled = true;

				int sx = coords[z][0];
				int sy = coords[z][1];
				int ex = coords[z][2];
				int ey = coords[z][3];

				//Get reference coordinates
				int csx = sx, cex = ex, csy = sy, cey = ey;
				if (sx != 0)
					csx += overlap / 2;
				if (sy != 0)
					csy += overlap / 2;
				if (ex != N)
					cex -= overlap / 2;
				if (ey != M)
					cey -= overlap / 2;

				std::vector<cv::Mat> coeffs;
				//Block to work on
				cv::Mat selection = cv::Mat(block_size, block_size, CV_32F, cv::Scalar(0));
				for (int i = sx; i < ex; i++){
					for (int j = sy; j < ey; j++){
						selection.at<float>(i - sx, j - sy) = texture.at<float>(i, j);
					}
				}

				//Save decomposition results to structure
				coeffs = FFST::shearletTransformSpect(selection);

				//Add points to training set
				for (int i = 0; i < cex - csx; i += SN){
					for (int j = 0; j < cey - csy; j += SM) if ((mask.at<char>(i + csx, j + csy) & CradleFunctions::DEFECT) != CradleFunctions::DEFECT) {

						ushort pi = piecemark.at<ushort>(i + csx, j + csy) + 1;	//Index of the piece
						int coeff_size = target_dim;

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
								block_used[hi][z] = 1;

								//Fill up sample - horizontal
								int lindex = 0;
								for (int l = 0; l < 61; l++) if (target_h[l] == 1){
									full_samples[fsample_pos][lindex] = coeffs[l].at<float>(i, j);
									lindex++;
								}
								fsample_pos++;
							}

							//Find horizontal cradle containing this segment (if any)
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
								block_used[vi][z] = 1;

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
							int coeff_size = target_dim;
							full_samples[fsample_pos] = std::vector<float>(coeff_size);
							sample_index[fsample_pos] = pi;
							block_used[pi][z] = 1;

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
							block_used[pi][z] = 1;

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
			}
		}
		processed++;
		if (canceled)
			return;
		full_samples.resize(fsample_pos);

		std::vector<int> sample_pos(sample_type.size());
		std::vector<std::vector<std::vector<float>>> sample_select(sample_type.size());
    std::cout << "sample_select.size(): " << sample_select.size() << std::endl;
		std::cout << "sample_type.size(): " << sample_type.size() << std::endl;

		//Randomly select samples to reduce computation time
		for (int i = 0; i < sample_select.size(); i++){
			std::cout << "Sample iter: " << i << std::endl;
			int local_pos = 0;
			std::vector<std::vector<float>> local(full_samples.size());
			//Find all samples with corresponding number
			for (int j = 0; j < full_samples.size(); j++){
				std::cout << "Sample index: " << sample_index[j] << ", " << i << " " << std::endl;
				if (sample_index[j] == i){
					std::cout << "sample chosen" << std::endl;
					//Copy sample to local selection
					local[local_pos] = std::vector<float>(full_samples[j].size());
					for (int k = 0; k < full_samples[j].size(); k++){
						local[local_pos][k] = full_samples[j][k];
					}
					local_pos++;
				}
			}
			std::cout << "local size: " << local.size() << std::endl;
      std::cout << "local_pos: " << local_pos << std::endl;
			//Resize
			local.resize(local_pos);
      if (local.size() == 0) {
				std::cout << "local size is 0" << std::endl;
				continue;
			}
			//Sample randomly
			sampleDataset(local, sample_select[i], max_samples);
		}

		//Drop global selection of coefficients to save memory
		full_samples.clear();
		sample_index.clear();

		//Normalize non-cradled components
		std::vector<float> mean_h, mean_v, var_h, var_v;
		normalizeNonCradle(sample_select[0], mean_h, var_h);	//Get normalization of horizontal non-cradle samples
		normalizeNonCradle(sample_select[1], mean_v, var_v);	//Get normalization of vertical non-cradle samples

		//Train separation model on each cradle piece
		cradle_model_fitting model;

		for (int mod_sel = 2; mod_sel < sample_select.size(); mod_sel++){

			if (sample_select[mod_sel].size() != 0){

				#pragma omp atomic
				processed++;

				std::vector<std::vector<float>> ncdata;	//Non-cradle data samples

				//Normalize the data & choose reference non-cradle data set
				if (sample_type[mod_sel] == CradleFunctions::HORIZONTAL_DIR){
					normalizeSamples(sample_select[mod_sel], mean_h, var_h);
					ncdata = sample_select[0];
				}
				else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
					normalizeSamples(sample_select[mod_sel], mean_v, var_v);
					ncdata = sample_select[1];
				}else{
					//Cross section
				}

				//Train the model
				model = gibbsSampling(sample_select[mod_sel], ncdata);

				cv::Mat new_texture;
				texture.copyTo(new_texture);

				// progress/abort
				#pragma omp critical
				if (!CradleFunctions::progress(processed, tot_progress))
					canceled = true;

				#pragma omp parallel for
				//Separate coefficients over entire image
				for (int z = 0; z < coords.size(); z++) if (block_used[mod_sel][z] == 1){

					if (!canceled){

						int sx = coords[z][0];
						int sy = coords[z][1];
						int ex = coords[z][2];
						int ey = coords[z][3];

						//Get reference coordinates
						int csx = sx, cex = ex, csy = sy, cey = ey;
						if (sx != 0)
							csx += overlap / 2;
						if (sy != 0)
							csy += overlap / 2;
						if (ex != N)
							cex -= overlap / 2;
						if (ey != M)
							cey -= overlap / 2;

						std::vector<cv::Mat> coeffs;
						//Block to work on
						cv::Mat selection = cv::Mat(block_size, block_size, CV_32F, cv::Scalar(0));
						for (int i = sx; i < ex; i++){
							for (int j = sy; j < ey; j++){
								selection.at<float>(i - sx, j - sy) = texture.at<float>(i, j);
							}
						}

						//Save decomposition results to structure
						coeffs = FFST::shearletTransformSpect(selection);

						//Use this to store samples
						std::vector<std::vector<float>> clusters(block_size * block_size / PSN / PSM * 2);
						int sample_pos = 0;

						//Subsample coefficients for clustering
						for (int i = 0; i < ex - sx; i += PSN){
							for (int j = 0; j < ey - sy; j += PSM) if ((mask.at<char>(i + sx, j + sy) & CradleFunctions::DEFECT) != CradleFunctions::DEFECT){

								int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
								int coeff_size = target_dim;

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
										//It's a horizontal cradle piece
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
							cv::Mat clusters_mat(clusters.size(), clusters[0].size(), CV_32F);
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

						//Post inference for subsampled coefficients
						std::vector<std::vector<float>> clusters_diffs;
						post_inference(model, clusters, ncdata, clusters_diffs);

						//Look up all coefficients
						sample_pos = 0;
						std::vector<std::vector<float>> samples(block_size * block_size);
						for (int i = 0; i < ex - sx; i++){
							for (int j = 0; j < ey - sy; j++) if ((mask.at<char>(i + sx, j + sy) & CradleFunctions::DEFECT) != CradleFunctions::DEFECT){

								int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
								int coeff_size = target_dim;

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

						std::vector<std::vector<float>> ncdata;	//Non-cradle data samples for post-inference

						//Normalize the data & choose reference non-cradle data set
						if (samples.size() != 0){
							if (sample_type[mod_sel] == CradleFunctions::HORIZONTAL_DIR){
								normalizeSamples(samples, mean_h, var_h);
								ncdata = sample_select[0];
							}
							else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
								normalizeSamples(samples, mean_v, var_v);
								ncdata = sample_select[1];
							}
							else{
								//Cross section
							}
						}

						std::vector<std::vector<float>> diffs;
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
								unNormalizeSamples(diffs, mean_h, var_h);
							}
							else if (sample_type[mod_sel] == CradleFunctions::VERTICAL_DIR){
								unNormalizeSamples(diffs, mean_v, var_v);
							}
							else{
								//Cross section
								unNormalizeSamplesCrossSection(diffs, mean_h, var_h, mean_v, var_v);
							}
						}

						//Reset index of sample_pos
						sample_pos = 0;

						//Apply separation to the decomposition coefficients
						for (int i = 0; i < ex - sx; i++){
							for (int j = 0; j < ey - sy; j++) if ((mask.at<char>(i + sx, j + sy) & CradleFunctions::DEFECT) != CradleFunctions::DEFECT) {

								int pi = piecemark.at<ushort>(i + sx, j + sy) + 1;	//Index of the piece
								int coeff_size = target_dim;

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
												//coeffs[l].at<float>(i, j) = 0;
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
												//coeffs[l].at<float>(i, j) = 0;
												lindex++;
											}
											sample_pos++;
										}
									}
								}
							}
						}

						//Reconstruct block
						reconstructBlock(new_texture, coeffs, sx, sy, csx, csy, cex, cey);
					}
				}
				new_texture.copyTo(texture);
			}
		}
		if (!canceled)
		{
			//Final image
			out = texture + cartoon;

			//Remove padding
			out = out(cv::Range(overlap / 2, N - overlap / 2), cv::Range(overlap / 2, M - overlap / 2));
		}
	}

	void post_inference(cradle_model_fitting &model, std::vector<std::vector<float>> &c, std::vector<std::vector<float>> &nc, std::vector<std::vector<float>> &difference){
		std::default_random_engine generator;

		generator.seed(SEED);
		int p = c[0].size();
		int nsample = model.etac_v.size();

		int k1 = model.Lambda_v[0].cols;
		int k2 = model.Gamma_v[0].cols;

		int Nnoncradle = nc.size();
		int Ncradle = c.size();

		//Convert cradle/noncradle coefficients to matrix
		cv::Mat cradleMat(Ncradle, p, CV_32F);			//z(~x,:)
		cv::Mat noncradleMat(Nnoncradle, p, CV_32F);	//z( x,:)
		for (int i = 0; i < Ncradle; i++){
			for (int j = 0; j < p; j++){
				cradleMat.at<float>(i, j) = c[i][j];
			}
		}
		for (int i = 0; i < Nnoncradle; i++){
			for (int j = 0; j < p; j++){
				noncradleMat.at<float>(i, j) = nc[i][j];
			}
		}
		cv::Mat fits(Ncradle, p, CV_32F, cv::Scalar(0));
		cv::Mat Lmsgt, Lmsg;
		cv::Mat Veta, Veta1;
		cv::Mat Vvect, U, S, Strans;
		cv::Mat Meta;
		cv::Mat eta_nc, eta_nctmp;
		cv::Mat Gammat, tmp;
		cv::Mat lambdat;
		cv::Mat z_nc_c, z_nc_nc, z_c;
		cv::Mat Gmsg, Gmsgt, Vxi1, Vxi, Mxi, xi;

		//Get fitting
		for (int i = 0; i < nsample; i++){
			//Go through all samples

			/*******************
			* Non-cradle signal
			*******************/
			Lmsg = cv::Mat(p, k1, CV_32F);
			for (int k = 0; k < p; k++){
				for (int l = 0; l < k1; l++){
					Lmsg.at<float>(k, l) = model.Lambda_v[i].at<float>(k, l) * model.ps_v[i].at<float>(0, k);
				}
			}
			cv::transpose(Lmsg, Lmsgt);
			//Lmsg = bsxfun(@times,Lambda,ps);

			Veta1 = Lmsgt * model.Lambda_v[i];
			for (int j = 0; j < k1; j++){
				Veta1.at<float>(j, j) += 1.0;
			}
			cv::invert(Veta1, Veta);
			// Veta = inv(eye(k1) + Lmsg'*Lambda{i})

			U = Cholesky(Veta1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Sample eta
			//Eta mean
			Meta = noncradleMat * Lmsg * Veta;

			//Sample zero-mean unit-variance uniform distribution
			/*
			eta_nctmp = cv::Mat(Meta.rows, Meta.cols, CV_32F, 0);
			//normal_distr = std::normal_distribution<float>(0, 1);
			for (int k = 0; k < eta_nctmp.rows; k++){
				for (int l = 0; l < eta_nctmp.cols; l++){
					eta_nctmp.at<float>(k, l) = normal_distr(generator);
				}
			}
			*/
			eta_nc = Meta;// +eta_nctmp*Strans;

			/*******************
			* Cradle signal
			*******************/
			cv::transpose(model.Gamma_v[i], Gammat);
			tmp = model.Gamma_v[i] * Gammat;
			for (int j = 0; j < Gammat.cols; j++){
				tmp.at<float>(j, j) += 1.0 / model.ps_v[i].at<float>(0, j);
			}
			//Gamma{i}*Gamma{i}'+diag(1./ps{i})
			cv::invert(tmp, Lmsg);
			Lmsg = Lmsg * model.Lambda_v[i];
			cv::transpose(Lmsg, Lmsgt);

			Veta1 = model.rho_v[i] * model.rho_v[i] * Lmsgt * model.Lambda_v[i];
			for (int j = 0; j < k1; j++){
				Veta1.at<float>(j, j) += 1.0;
			}
			cv::invert(Veta1, Veta);
			// Veta = inv(eye(k1) + rho(i)^2*Lmsg'*Lambda{i})

			//Mean
			cv::Mat eta_c, eta_ctmp;
			Meta = cradleMat * Lmsg * Veta;

			//Sample zero-mean unit-variance uniform distribution
			/*
			eta_ctmp = cv::Mat(Meta.rows, Meta.cols, CV_32F, 0);
			normal_distr = std::normal_distribution<float>(0, 1);
			for (int k = 0; k < eta_ctmp.rows; k++){
				for (int l = 0; l < eta_ctmp.cols; l++){
					eta_ctmp.at<float>(k, l) = normal_distr(generator);
				}
			}*/

			U = Cholesky(Veta1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Get multivariate normal distribution
			eta_c = Meta;// +eta_ctmp * Strans;

			/*******************
			* Compute z_nc
			*******************/

			cv::transpose(model.Lambda_v[i], lambdat);
			z_nc_nc = eta_nc * lambdat;					//z_nc(x,:) = eta(x,:)*Lambda{i}';
			z_nc_c = eta_c * lambdat * model.rho_v[i];	//z_nc(~x,:) = eta(~x,:)*Lambda{i}'*rho(i);

			/*******************
			* Compute xi
			*******************/
			Gmsg = cv::Mat(model.Gamma_v[i].rows, model.Gamma_v[i].cols, CV_32F);
			for (int k = 0; k < model.Gamma_v[i].rows; k++){
				for (int l = 0; l < model.Gamma_v[i].cols; l++){
					Gmsg.at<float>(k, l) = model.Gamma_v[i].at<float>(k, l) * model.ps_v[i].at<float>(0, k);
				}
			}//Gmsg = bsxfun(@times, Gamma{ i }, ps{ i });
			cv::transpose(Gmsg, Gmsgt);

			Vxi1 = Gmsgt * model.Gamma_v[i];
			for (int k = 0; k < Vxi1.cols; k++){
				Vxi1.at<float>(k, k)++;
			}//Vxi1 = eye(k2) + Gmsg'*Gamma{i};

			cv::invert(Vxi1, Vxi);

			Mxi = (cradleMat - z_nc_c) * Gmsg * Vxi; // (z(~x,:) - z_nc(~x,:))*Gmsg*Vxi
			cv::Mat kappavxi = model.kappa_v[i] * Vxi;
			for (int k = 0; k < Mxi.rows; k++){
				for (int l = 0; l < Mxi.cols; l++){
					Mxi.at<float>(k, l) = Mxi.at<float>(k, l) + kappavxi.at<float>(0, l);
				}
			}//Mxi = bsxfun(@plus, (z(~x, :) - z_nc(~x, :))*Gmsg*Vxi, kappa{ i }*Vxi);

			//Sample multivariate normal distribution
			/*
			cv::Mat samples(Mxi.rows, Mxi.cols, CV_32F, 0);
			normal_distr = std::normal_distribution<float>(0, 1);
			for (int k = 0; k < samples.rows; k++){
				for (int l = 0; l < samples.cols; l++){
					samples.at<float>(k, l) = 0;// normal_distr(generator);
				}
			}*/

			//Get cholesky decomposition of covariance matrix
			U = Cholesky(Vxi1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Get multivariate normal distribution
			xi = Mxi;// +samples * Strans;
			// xi(~x,:) = Mxi + normrnd(0,1,[Ncradle,k2])*S';

			/*******************
			* Compute z_c
			*******************/
			z_c = xi * Gammat;
			fits += z_c;		//Add cradle-component to the sum
		}
		fits /= nsample;

		//Convert back to coefficient array
		difference = std::vector<std::vector<float>>(Ncradle);
		for (int i = 0; i < Ncradle; i++){
			difference[i] = std::vector<float>(p);
			for (int j = 0; j < p; j++){
				difference[i][j] = fits.at<float>(i, j);
			}
		}
	}

	cradle_model_fitting gibbsSampling(std::vector<std::vector<float>> &cradle, std::vector<std::vector<float>> &noncradle){
		//Initialize variables used
		std::default_random_engine generator;
		std::gamma_distribution<float> gamma_distr;
		std::normal_distribution<float> normal_distr;
		std::uniform_real_distribution<double> unif_distr(0.0, 1.0);

		generator.seed(SEED);
		int Ncradle = cradle.size();
		int Nnoncradle = noncradle.size();

		int nrun = 600;
		int burn = 500;
		int thin = 1;
		float sp = (nrun - burn) * 1.0 / thin;

		int p = cradle[0].size();
		int k1 = p;
		int k2 = std::floor(std::log(p) * 3);

		cv::Mat cradleMat(Ncradle, p, CV_32F);
		cv::Mat noncradleMat(Nnoncradle, p, CV_32F);

		for (int i = 0; i < Ncradle; i++){
			for (int j = 0; j < p; j++){
				cradleMat.at<float>(i, j) = cradle[i][j];
			}
		}

		for (int i = 0; i < Nnoncradle; i++){
			for (int j = 0; j < p; j++){
				noncradleMat.at<float>(i, j) = noncradle[i][j];
			}
		}

		float ad, bd;
		float as = 1, bs = 0.3;
		float df = 3;
		float ad1 = 2.1, bd1 = 1;
		float ad2 = 3.1, bd2 = 1;
		float adf = 1, bdf = 1;

		float b0 = 1;
		float b1 = 0.0005;
		float epsilon = 1e-2;		// threshold limit
		float prop = .95;			//proportion of redundant elements within columns

		float mrho = 1;
		float vrho = 0.3*0.3;

		cv::Mat ps(1, p, CV_32F);
		for (int i = 0; i < p; i++){
			ps.at<float>(0, i) = 10;
		}// ps = 10*ones(p,1);

		cv::Mat ps2(1, p, CV_32F);
		for (int i = 0; i < p; i++){
			ps2.at<float>(0, i) = 10;
		}// ps2 = 10*ones(p,1);

		cv::Mat sigma(p, p, CV_32F, cv::Scalar(0));
		for (int i = 0; i < p; i++){
			sigma.at<float>(i, i) = 1.0 / ps.at<float>(0, i);
		}// Sigma=diag(1./ps);

		cv::Mat lambda(p, k1, CV_32F, cv::Scalar(0));	//Lambda = zeros(p,k1);

		cv::Mat psijh1(p, k1, CV_32F);
		gamma_distr = std::gamma_distribution<float>(df / 2, 2.0 / df);
		for (int i = 0; i < p; i++){
			for (int j = 0; j < k1; j++){
				psijh1.at<float>(i, j) = gamma_distr(generator);
			}
		}// psijh1 = gamrnd(df/2,2/df,[p,k1]);

		std::vector<float> delta1(k1);
		gamma_distr = std::gamma_distribution<float>(ad1, bd1);
		delta1[0] = gamma_distr(generator);
		gamma_distr = std::gamma_distribution<float>(ad2, bd2);
		for (int i = 1; i < k1; i++){
			delta1[i] = gamma_distr(generator);
		}// delta1 = [gamrnd(ad1, bd1); gamrnd(ad2, bd2, [k1 - 1, 1])];

		std::vector<float> tauh1(k1);
		tauh1[0] = delta1[0];
		for (int i = 1; i < k1; i++){
			tauh1[i] = delta1[i] * tauh1[i - 1];
		}// tauh1 = cumprod(delta1);

		cv::Mat Plam(p, k1, CV_32F);
		for (int i = 0; i < p; i++){
			for (int j = 0; j < k1; j++){
				Plam.at<float>(i, j) = psijh1.at<float>(i, j) * tauh1[j];
			}
		}// Plam = bsxfun(@times,psijh1,tauh1');

		cv::Mat ttt;

		double rho = mrho;
		cv::Mat Gamma(p, k2, CV_32F, cv::Scalar(0));
		cv::Mat psijh2(p, k2, CV_32F);
		gamma_distr = std::gamma_distribution<float>(df / 2, 2.0 / df);
		for (int i = 0; i < p; i++){
			for (int j = 0; j < k2; j++){
				psijh2.at<float>(i, j) = gamma_distr(generator);
			}
		}// psijh2 = gamrnd(df/2,2/df,[p,k2]);

		std::vector<float> delta2(k2);
		gamma_distr = std::gamma_distribution<float>(ad1, bd1);
		delta2[0] = gamma_distr(generator);
		gamma_distr = std::gamma_distribution<float>(ad2, bd2);
		for (int i = 1; i < k2; i++){
			delta2[i] = gamma_distr(generator);
		}// delta2 = [gamrnd(Ad1, bd1); gamrnd(Ad2, bd2, [k2 - 1, 1])];

		std::vector<float> tauh2(k2);
		tauh2[0] = delta2[0];
		for (int i = 1; i < k2; i++){
			tauh2[i] = delta2[i] * tauh2[i - 1];
		}// tauh2 = cumprod(delta2);

		cv::Mat Pgam(p, k2, CV_32F);
		for (int i = 0; i < p; i++){
			for (int j = 0; j < k2; j++){
				Pgam.at<float>(i, j) = psijh2.at<float>(i, j) * tauh2[j];
			}
		}// Pgam = bsxfun(@times,psijh2,tauh2');

		normal_distr = std::normal_distribution<float>(0, 1);
		cv::Mat kappa(1, k2, CV_32F);
		for (int i = 0; i < k2; i++){
			kappa.at<float>(0, i) = normal_distr(generator);
		}// kappa = normrnd(0,1,[1,k2]);

		cv::Mat xi(Ncradle, k2, CV_32F);
		for (int i = 0; i < Ncradle; i++){
			for (int j = 0; j < k2; j++){
				xi.at<float>(i, j) = normal_distr(generator) + kappa.at<float>(0, j);
			}
		}// xi = bsxfun(@plus, normrnd(0,1,[Ncradle,k2]),kappa);

		/* End of variable definition/initialization */

		//Variables used in the loop
		cv::Mat Lmsg, Lmsgt, Veta, Veta1;
		cv::Mat Meta;
		cv::Mat eta_c, eta_nc, eta_nctmp;
		cv::Mat gammatrans, lambdatrans;
		cv::Mat eta2_nc, eta_nct, alam;
		cv::Mat Gmsg, Gmsgt;
		cv::Mat Vxi, Vxi1, Mxi;
		cv::Mat xi2, xit, agam;
		cv::Mat U, S, L, Strans;

		//Variables used to store results of after-burn iterations
		cradle_model_fitting fitting;
		fitting.rho_v = std::vector<float>(nrun - burn);
		fitting.kappa_v = std::vector<cv::Mat>(nrun - burn);
		fitting.Gamma_v = std::vector<cv::Mat>(nrun - burn);
		fitting.xi_v = std::vector<cv::Mat>(nrun - burn);
		fitting.Lambda_v = std::vector<cv::Mat>(nrun - burn);
		fitting.ps_v = std::vector<cv::Mat>(nrun - burn);
		fitting.etac_v = std::vector<cv::Mat>(nrun - burn);
		fitting.etanc_v = std::vector<cv::Mat>(nrun - burn);

		/*** Start Gibbs sampling ***/
		for (int iter = 0; iter < nrun; iter++){

			// **** Update eta, non-cradle part  ****
			Lmsg = cv::Mat(p, k1, CV_32F);
			for (int i = 0; i < p; i++){
				for (int j = 0; j < k1; j++){
					Lmsg.at<float>(i, j) = lambda.at<float>(i, j) * ps.at<float>(0, i);
				}
			}
			cv::transpose(Lmsg, Lmsgt);
			//Lmsg = bsxfun(@times,Lambda,ps);

			Veta1 = Lmsgt * lambda;
			for (int i = 0; i < Veta1.rows; i++){
				Veta1.at<float>(i, i) += 1;	//+eye(k1)
			}
			cv::invert(Veta1, Veta);

			U = Cholesky(Veta1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Sample eta
			//Eta mean
			Meta = noncradleMat * Lmsg * Veta;

			//Sample zero-mean unit-variance uniform distribution
			eta_nctmp = cv::Mat(Meta.rows, Meta.cols, CV_32F);
			normal_distr = std::normal_distribution<float>(0, 1);
			for (int i = 0; i < eta_nctmp.rows; i++){
				for (int j = 0; j < eta_nctmp.cols; j++){
					eta_nctmp.at<float>(i, j) = normal_distr(generator);
				}
			}

			//Get multivariate normal distribution
			eta_nc = Meta + eta_nctmp * Strans;

			// **** Update eta, cradle part  ****
			Veta1 = rho * rho * Lmsgt * lambda;
			for (int i = 0; i < Veta1.rows; i++){
				Veta1.at<float>(i, i) += 1;	//+eye(k1)
			}
			cv::invert(Veta1, Veta);

			U = Cholesky(Veta1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Eta mean
			cv::transpose(Gamma, gammatrans);
			Meta = (cradleMat - xi * gammatrans) * rho * Lmsg * Veta;
			//Meta = (Y(mask,:) - xi*Gamma')*rho*Lmsg*Veta;

			//Sample zero-mean unit-variance uniform distribution
			eta_nctmp = cv::Mat(Meta.rows, Meta.cols, CV_32F);
			normal_distr = std::normal_distribution<float>(0, 1);
			for (int i = 0; i < eta_nctmp.rows; i++){
				for (int j = 0; j < eta_nctmp.cols; j++){
					eta_nctmp.at<float>(i, j) = normal_distr(generator);
				}
			}
			//eta_nctmp = TextureRemoval::readMatFromFile("C:/Users/localadmin/Documents/MATLAB/code/platypus_matlab/result/eta_c" + std::to_string(iter + 1) + ".txt", eta_nctmp.rows, eta_nctmp.cols);

			//Get multivariate normal distribution
			eta_c = Meta + eta_nctmp * Strans;

			/*** Update beta's --> Do nothing under supervised model ***/

			/*** Update Lambda ***/
			cv::transpose(eta_nc, eta_nct);
			eta2_nc = eta_nct*eta_nc;		//eta2 = eta(~mask,:)'*eta(~mask,:);
			alam = eta_nct * noncradleMat;	//alam = eta(~mask,:)'*Y(~mask,:);

			for (int i = 0; i < p; i++){
				cv::Mat diagPlam(Plam.cols, Plam.cols, CV_32F, cv::Scalar(0));
				for (int j = 0; j < Plam.cols; j++){
					diagPlam.at<float>(j, j) = Plam.at<float>(i, j);
				}
				cv::Mat Qlam = diagPlam + ps.at<float>(0, i) * eta2_nc;
				// Qlam = diag(Plam(j,:)) + ps(j)*eta2;

				cv::Mat blam = ps.at<float>(0, i) * alam(cv::Range(0, alam.rows), cv::Range(i, i + 1));
				// blam = ps(j)*alam(:,j);

				//Cholesky decomposition
				L = CholeskyLower(Qlam);
				cv::Mat Ltransinv, Linv, Ltrans;
				cv::transpose(L, Ltrans);
				cv::invert(Ltrans, Ltransinv);
				cv::invert(L, Linv);

				cv::Mat samples(k1, 1, CV_32F, cv::Scalar(0));
				normal_distr = std::normal_distribution<float>(0, 1);
				for (int j = 0; j < samples.rows; j++){
					samples.at<float>(j, 0) = normal_distr(generator);
				}

				cv::Mat mvndmat = Ltransinv * Linv * blam + Ltransinv * samples;

				for (int j = 0; j < mvndmat.rows; j++){
					lambda.at<float>(i, j) = mvndmat.at<float>(j, 0);
				}//Lambda(i,:) is sampled from multivariate normal distribution with mean inv(Qlam) * blam and covariance matrix inv(Qlam)
			}

			/*** Update psi_{jh}'s ***/
			float dftmp;
			dftmp = df / 2 + 0.5;
			for (int i = 0; i < psijh1.rows; i++){
				for (int j = 0; j < psijh1.cols; j++){
					float tmp = lambda.at<float>(i, j);
					tmp = 1.0 / (df / 2 + tmp*tmp*tauh1[j]);	//tmp = 1./(df/2 + bsxfun(@times,Lambda.^2,tauh1'))

					gamma_distr = std::gamma_distribution<float>(dftmp, tmp);
					psijh1.at<float>(i, j) = gamma_distr(generator);	//gamrnd(df/2 + 0.5,1./(df/2 + bsxfun(@times,Lambda.^2,tauh1')));
				}
			}//  psijh1 = gamrnd(df/2 + 0.5,1./(df/2 + bsxfun(@times,Lambda.^2,tauh1')));

			/*** Update delta & tauh ***/
			cv::Mat matlocal(lambda.rows, lambda.cols, CV_32F);
			for (int i = 0; i < lambda.rows; i++){
				for (int j = 0; j < lambda.cols; j++){
					float tmp = lambda.at<float>(i, j);
					matlocal.at<float>(i, j) = psijh1.at<float>(i, j) * tmp * tmp;
				}
			}//matlocal = mat = bsxfun(@times,psijh1,Lambda.^2);

			std::vector<float> tmpsummat(tauh1.size());
			for (int i = 0; i < tauh1.size(); i++){
				tmpsummat[i] = 0;
				for (int j = 0; j < matlocal.rows; j++){
					tmpsummat[i] += matlocal.at<float>(j, i);
				}
			}// tmpsummat = sum(mat)';

			float tmpmss = 0;
			for (int i = 0; i < tauh1.size(); i++){
				tmpmss += tauh1[i] * tmpsummat[i];
			}// tmpmss = sum(tauh1.*sum(mat)')

			ad = ad1 + 0.5 * p * k1;	//ad = ad1 + 0.5*p*k1;
			bd = bd1 + 0.5 * (1.0 / delta1[0]) * tmpmss;		//bd = bd1 + 0.5*(1/delta1(1))*sum(tauh1.*sum(mat)');

			//Resample
			gamma_distr = std::gamma_distribution<float>(ad, 1.0 / bd);
			delta1[0] = gamma_distr(generator);	//delta1(1) = gamrnd(ad,1/bd);

			//Update tauh1
			tauh1[0] = delta1[0];
			for (int i = 1; i < delta1.size(); i++){
				tauh1[i] = tauh1[i - 1] * delta1[i];
			}//tauh1 = cumprod(delta1);

			for (int h = 2; h <= k1; h++){
				ad = ad2 + 0.5*p*(k1 - h + 1);

				tmpmss = 0;
				for (int i = h - 1; i < tauh1.size(); i++){
					tmpmss += tauh1[i] * tmpsummat[i];
				}//tmpmss = sum(tauh1(h:end).*sum(mat(:,h:end))')

				bd = bd1 + 0.5 * (1.0 / delta1[h - 1]) * tmpmss; //bd = bd2 + 0.5*(1/delta1(h))*sum(tauh1(h:end).*sum(mat(:,h:end))');

				//Resample
				gamma_distr = std::gamma_distribution<float>(ad, 1.0 / bd);
				delta1[h - 1] = gamma_distr(generator);	//delta1(h) = gamrnd(ad,1/bd);

				//Update tauh1
				tauh1[0] = delta1[0];
				for (int i = 1; i < delta1.size(); i++){
					tauh1[i] = tauh1[i - 1] * delta1[i];
				}//tauh1 = cumprod(delta1);
			}

			/*** Update xi, xiz ***/
			Gmsg = cv::Mat(Gamma.rows, Gamma.cols, CV_32F);
			for (int i = 0; i < Gamma.rows; i++){
				for (int j = 0; j < Gamma.cols; j++){
					Gmsg.at<float>(i, j) = Gamma.at<float>(i, j) * ps.at<float>(0, i);
				}
			}//Gmsg = bsxfun(@times, Gamma, ps);
			cv::transpose(Gmsg, Gmsgt);

			Vxi1 = Gmsgt * Gamma;
			for (int i = 0; i < Vxi1.cols; i++){
				Vxi1.at<float>(i, i) = Vxi1.at<float>(i, i) + 1;
			}//Vxi1 = eye(k2) + Gmsg'*Gamma;

			cv::invert(Vxi1, Vxi);
			cv::transpose(lambda, lambdatrans);

			Mxi = (cradleMat - rho * eta_c * lambdatrans) * Gmsg * Vxi; // (Y(mask,:) - rho*eta(mask,:)*Lambda')*Gmsg*Vxi

			cv::Mat kappavxi = kappa * Vxi;
			for (int i = 0; i < Mxi.rows; i++){
				for (int j = 0; j < Mxi.cols; j++){
					Mxi.at<float>(i, j) = Mxi.at<float>(i, j) + kappavxi.at<float>(0, j);
				}
			}//  Mxi = bsxfun(@plus , (Y(mask,:) - rho*eta(mask,:)*Lambda')*Gmsg*Vxi , kappa*Vxi);

			//Sample multivariate normal distribution
			cv::Mat samples(Mxi.rows, Mxi.cols, CV_32F);
			normal_distr = std::normal_distribution<float>(0, 1);
			for (int i = 0; i < samples.rows; i++){
				for (int j = 0; j < samples.cols; j++){
					samples.at<float>(i, j) = normal_distr(generator);
				}
			}
			//samples = TextureRemoval::readMatFromFile("C:/Users/localadmin/Documents/MATLAB/code/platypus_matlab/result/xi" + std::to_string(iter + 1) + ".txt", samples.rows, samples.cols);

			//Get cholesky decomposition of covariance matrix
			U = Cholesky(Vxi1);
			cv::invert(U, S);
			cv::transpose(S, Strans);

			//Get multivariate normal distribution
			xi = Mxi + samples * Strans;

			/*** Update kappa ***/
			std::vector<float> xisum(xi.cols);
			for (int j = 0; j < xi.cols; j++){
				for (int i = 0; i < xi.rows; i++){
					xisum[j] += xi.at<float>(i, j);
				}
				xisum[j] /= Ncradle;
			}//xisum = sum(xi,1)/Ncradle

			for (int i = 0; i < kappa.cols; i++){
				normal_distr = std::normal_distribution<float>(xisum[i], 1.0 / Ncradle);
				kappa.at<float>(0, i) = normal_distr(generator);
			}//kappa = arrayfun(@(x)normrnd(x,1/Ncradle,[1,1]),sum(xi,1)/Ncradle);

			/*** Update gamma ***/
			cv::transpose(xi, xit);
			xi2 = xit * xi;
			cv::transpose(lambda, lambdatrans);
			agam = xit * (cradleMat - rho * eta_c * lambdatrans);

			for (int i = 0; i < p; i++){
				cv::Mat diagPgam(Pgam.cols, Pgam.cols, CV_32F, cv::Scalar(0));
				for (int j = 0; j < Pgam.cols; j++){
					diagPgam.at<float>(j, j) = Pgam.at<float>(i, j);
				}

				cv::Mat Qgam = diagPgam + ps2.at<float>(0, i) * xi2;

				cv::Mat bgam = ps2.at<float>(0, i) * agam(cv::Range(0, agam.rows), cv::Range(i, i + 1));
				cv::Mat QgamInv;

				cv::invert(Qgam, QgamInv);

				//Spectral decomposition
				L = CholeskyLower(Qgam);
				cv::Mat Ltransinv, Linv, Ltrans;
				cv::transpose(L, Ltrans);
				cv::invert(Ltrans, Ltransinv);
				cv::invert(L, Linv);

				cv::Mat samples(k2, 1, CV_32F, cv::Scalar(0));
				normal_distr = std::normal_distribution<float>(0, 1);
				for (int j = 0; j < samples.rows; j++){
					samples.at<float>(j, 0) = normal_distr(generator);
				}
				//samples = TextureRemoval::readMatFromFile("C:/Users/localadmin/Documents/MATLAB/code/platypus_matlab/result/zlamG" + std::to_string(iter + 1) + "_" + std::to_string(i + 1) + ".txt", samples.rows, samples.cols);

				cv::Mat mvndmat = Ltransinv * Linv * bgam + Ltransinv * samples;

				for (int j = 0; j < mvndmat.rows; j++){
					Gamma.at<float>(i, j) = mvndmat.at<float>(j, 0);
				}//Gamma(i,:) is sampled from multivariate normal distribution with mean inv(Qgam) * bgam and covariance matrix inv(Qgam)
			}
			cv::transpose(Gamma, gammatrans);

			/*** Update psi_{jh}'s ***/
			dftmp = df / 2 + 0.5;
			for (int i = 0; i < psijh2.rows; i++){
				for (int j = 0; j < psijh2.cols; j++){
					float tmp = Gamma.at<float>(i, j);
					tmp = 1.0 / (df / 2 + tmp*tmp*tauh2[j]);	//tmp = 1./(df/2 + bsxfun(@times,Gamma.^2,tauh2'))

					gamma_distr = std::gamma_distribution<float>(dftmp, tmp);
					psijh2.at<float>(i, j) = gamma_distr(generator);	//gamrnd(df/2 + 0.5,1./(df/2 + bsxfun(@times,Gamma.^2,tauh2')));
				}
			}

			/*** Update delta2 & tauh2 ***/
			matlocal = cv::Mat(Gamma.rows, Gamma.cols, CV_32F);
			for (int i = 0; i < Gamma.rows; i++){
				for (int j = 0; j < Gamma.cols; j++){
					float tmp = Gamma.at<float>(i, j);
					matlocal.at<float>(i, j) = psijh2.at<float>(i, j) * tmp * tmp;
				}
			}//mat = bsxfun(@times, psijh2, Gamma. ^ 2);

			tmpsummat = std::vector<float>(tauh2.size());
			for (int i = 0; i < tauh2.size(); i++){
				tmpsummat[i] = 0;
				for (int j = 0; j < matlocal.rows; j++){
					tmpsummat[i] += matlocal.at<float>(j, i);
				}
			}// tmpsummat = sum(mat)';

			tmpmss = 0;
			for (int i = 0; i < tauh2.size(); i++){
				tmpmss += tauh2[i] * tmpsummat[i];
			}// tmpmss = sum(tauh2.*sum(mat)')

			ad = ad1 + 0.5 * p * k2;	//ad = ad1 + 0.5*p*k2;
			bd = bd1 + 0.5 * (1.0 / delta2[0]) * tmpmss;		//bd = bd1 + .5*(1 / delta2(1))*sum(tauh2.*sum(mat)');

			//Resample
			gamma_distr = std::gamma_distribution<float>(ad, 1.0 / bd);
			delta2[0] = gamma_distr(generator);	//delta2(1) = gamrnd(ad,1/bd);

			//Update tauh2
			tauh2[0] = delta2[0];
			for (int i = 1; i < delta2.size(); i++){
				tauh2[i] = tauh2[i - 1] * delta2[i];
			}//tauh2 = cumprod(delta2);

			for (int h = 2; h <= k2; h++){
				ad = ad2 + 0.5*p*(k2 - h + 1);

				tmpmss = 0;
				for (int i = h - 1; i < tauh2.size(); i++){
					tmpmss += tauh2[i] * tmpsummat[i];
				}//tmpmss = sum(tauh2(h:end).*sum(mat(:,h:end))')

				bd = bd1 + 0.5 * (1.0 / delta2[h - 1]) * tmpmss; //bd = bd2 + 0.5*(1/delta2(h))*sum(tauh2(h:end).*sum(mat(:,h:end))');

				//Resample
				gamma_distr = std::gamma_distribution<float>(ad, 1.0 / bd);
				delta2[h - 1] = gamma_distr(generator);	//delta2(h) = gamrnd(ad,1/bd);

				//Update tauh2
				tauh2[0] = delta2[0];
				for (int i = 1; i < delta2.size(); i++){
					tauh2[i] = tauh2[i - 1] * delta2[i];
				}//tauh2 = cumprod(delta2);
			}

			tauh2[0] = delta2[0];
			for (int i = 1; i < delta2.size(); i++){
				tauh2[i] = tauh2[i - 1] * delta2[i];
			}

			Pgam = cv::Mat(p, k2, CV_32F);
			for (int i = 0; i < p; i++){
				for (int j = 0; j < k2; j++){
					Pgam.at<float>(i, j) = psijh2.at<float>(i, j) * tauh2[j];
				}
			}// Pgam = bsxfun(@times,psijh2,tauh2');

			/*** Update precision parameters ***/
			Plam = cv::Mat(p, k1, CV_32F);
			for (int i = 0; i < p; i++){
				for (int j = 0; j < k1; j++){
					Plam.at<float>(i, j) = psijh1.at<float>(i, j) * tauh1[j];
				}
			}// Plam = bsxfun(@times, psijh1, tauh1');

			//Split, in function of burn reached/not reached
			if (iter < burn){

				// make adaptations for non - cradle parameters
				float prob = 1.0 / std::exp(b0 + b1*iter);
				float uu = unif_distr(generator);

				std::vector<float> lind(lambda.cols);
				for (int i = 0; i < lambda.rows; i++){
					for (int j = 0; j < lambda.cols; j++){
						if (std::abs(lambda.at<float>(i, j)) < epsilon){
							lind[j] += 1.0 / p;
						}
					}
				}// lind = sum(abs(Lambda) < epsilon)/p;

				int num = 0;
				for (int i = 0; i < lind.size(); i++){
					if (lind[i] >= prop)
						num++;
				}// vec = lind >=prop; num = sum(vec);

				if (uu < prob){

					if (iter > 20 && num == 0){
						//Expand
						k1++;

						//Extend lambda
						cv::Mat zerocol(p, 1, CV_32F, cv::Scalar(0));
						cv::hconcat(lambda, zerocol, lambda);
						//Lambda(:,k1) = zeros(p,1);

						//Extend eta_c
						cv::Mat colextendc(eta_c.rows, 1, CV_32F);
						normal_distr = std::normal_distribution<float>(0, 1);
						for (int i = 0; i < colextendc.rows; i++){
							colextendc.at<float>(i, 0) = normal_distr(generator);
						}
						cv::hconcat(eta_c, colextendc, eta_c);
						//eta(mask,k1) = normrnd(0,1,[ncradle,1]);

						//Extend eta_nc
						cv::Mat colextendnc(eta_nc.rows, 1, CV_32F);
						normal_distr = std::normal_distribution<float>(0, 1);
						for (int i = 0; i < colextendnc.rows; i++){
							colextendnc.at<float>(i, 0) = normal_distr(generator);
						}
						cv::hconcat(eta_nc, colextendnc, eta_nc);
						//eta(~mask,k1) = normrnd(0,1,[nnoncradle,1]);

						//Extend psijh1
						cv::Mat colextend(p, 1, CV_32F);
						gamma_distr = std::gamma_distribution<float>(df / 2, 2.0 / df);
						for (int i = 0; i < p; i++){
							colextend.at<float>(i, 0) = gamma_distr(generator);
						}
						cv::hconcat(psijh1, colextend, psijh1);
						// psijh1(:,k1) = gamrnd(df/2,2/df,[p,1]);

						//Extend delta1
						gamma_distr = std::gamma_distribution<float>(ad2, 1.0 / bd2);
						delta1.push_back(gamma_distr(generator));
						//delta1 = [delta1;gamrnd(ad2,1/bd2)];

						//Extend tauh1
						tauh1.push_back(tauh1[k1 - 2] * delta1[k1 - 1]);
						//tauh1 = cumprod(delta1);

						//Update Plam
						Plam = cv::Mat(p, k1, CV_32F);
						for (int i = 0; i < p; i++){
							for (int j = 0; j < k1; j++){
								Plam.at<float>(i, j) = psijh1.at<float>(i, j) * tauh1[j];
							}
						}// Plam = bsxfun(@times, psijh1, tauh1');
					}
					else if (num > 0 && num < k1){
						//Contract
						k1 -= num;

						//Contract lambda
						int index;
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < lambda.rows; j++){
									lambda.at<float>(j, index) = lambda.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k1 columns
						lambda = lambda(cv::Range(0, lambda.rows), cv::Range(0, k1));
						// Lambda = Lambda(:,nonred);

						//Contract psijh1
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < psijh1.rows; j++){
									psijh1.at<float>(j, index) = psijh1.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k1 columns
						psijh1 = psijh1(cv::Range(0, psijh1.rows), cv::Range(0, k1));
						// psijh1 = psijh1(:,nonred);

						//Contract eta_c
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < eta_c.rows; j++){
									eta_c.at<float>(j, index) = eta_c.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k1 columns
						eta_c = eta_c(cv::Range(0, eta_c.rows), cv::Range(0, k1));

						//Contract eta_c
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < eta_nc.rows; j++){
									eta_nc.at<float>(j, index) = eta_nc.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k1 columns
						eta_nc = eta_nc(cv::Range(0, eta_nc.rows), cv::Range(0, k1));

						//Contract delta1
						std::vector<float> dtmp(k1);
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								dtmp[index] = delta1[i];
								index++;
							}
						}
						delta1 = dtmp;
						// delta1 = delta1(nonred);

						//Contract tauh1
						tauh1 = std::vector<float>(k1);
						tauh1[0] = delta1[0];
						for (int i = 1; i < delta1.size(); i++){
							tauh1[i] = tauh1[i - 1] * delta1[i];
						}// tauh1 = cumprod(delta1);

						//Update Plam
						Plam = cv::Mat(p, k1, CV_32F);
						for (int i = 0; i < p; i++){
							for (int j = 0; j < k1; j++){
								Plam.at<float>(i, j) = psijh1.at<float>(i, j) * tauh1[j];
							}
						}// Plam = bsxfun(@times, psijh1, tauh1');
					}
				}

				// make adaptations for cradle parameters
				prob = 1.0 / std::exp(b0 + b1*iter);
				uu = unif_distr(generator);

				lind = std::vector<float>(Gamma.cols);
				for (int i = 0; i < Gamma.rows; i++){
					for (int j = 0; j < Gamma.cols; j++){
						if (std::abs(Gamma.at<float>(i, j)) < epsilon){
							lind[j] += 1.0 / p;
						}
					}
				}// lind = sum(abs(Gamma) < epsilon)/p;
				num = 0;
				for (int i = 0; i < lind.size(); i++){
					if (lind[i] >= prop)
						num++;
				}// vec = lind >=prop;num = sum(vec);

				if (uu < prob){
					if (iter > 20 && num == 0){
						//Expand
						k2++;

						//Extend gamma
						cv::Mat zerocol(p, 1, CV_32F, cv::Scalar(0));
						cv::hconcat(Gamma, zerocol, Gamma);
						// Gamma(:,k2) = zeros(p,1);

						//Extend kappa
						cv::Mat colextend(1, 1, CV_32F);
						normal_distr = std::normal_distribution<float>(0, 1);
						colextend.at<float>(0, 0) = normal_distr(generator);
						cv::hconcat(kappa, colextend, kappa);
						// kappa(k2) = normrnd(0,1,[1,1]);

						//Extend xi
						colextend = cv::Mat(Ncradle, 1, CV_32F);
						normal_distr = std::normal_distribution<float>(kappa.at<float>(0, k2 - 1), 1);
						for (int i = 0; i < Ncradle; i++){
							colextend.at<float>(i, 0) = normal_distr(generator);
						}
						cv::hconcat(xi, colextend, xi);
						// xi(:,k2) = normrnd(kappa(k2),1,[Ncradle,1]);

						//Extend psijh2
						colextend = cv::Mat(p, 1, CV_32F);
						gamma_distr = std::gamma_distribution<float>(df / 2, 2.0 / df);
						for (int i = 0; i < p; i++){
							colextend.at<float>(i, 0) = gamma_distr(generator);
						}
						cv::hconcat(psijh2, colextend, psijh2);
						// psijh2(:,k2) = gamrnd(df/2,2/df,[p,1]);

						//Extend delta2
						gamma_distr = std::gamma_distribution<float>(ad2, 1.0 / bd2);
						delta2.push_back(gamma_distr(generator));
						//delta2 = [delta2;gamrnd(ad2,1/bd2)];

						//Extend tauh1
						tauh2.push_back(tauh2[k2 - 2] * delta2[k2 - 1]);
						//tauh2 = cumprod(delta2);

						//Update Pgam
						Pgam = cv::Mat(p, k2, CV_32F);
						for (int i = 0; i < p; i++){
							for (int j = 0; j < k2; j++){
								Pgam.at<float>(i, j) = psijh2.at<float>(i, j) * tauh2[j];
							}
						}// Pgam = bsxfun(@times,psijh2,tauh2');
					}
					else if (num > 0 && num < k2){
						//Contract
						k2 -= num;

						//Contract gamma
						int index;
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < Gamma.rows; j++){
									Gamma.at<float>(j, index) = Gamma.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k2 columns
						Gamma = Gamma(cv::Range(0, Gamma.rows), cv::Range(0, k2));
						// Gamma = Gamma(:,nonred);

						//Contract psijh2
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < psijh2.rows; j++){
									psijh2.at<float>(j, index) = psijh2.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k2 columns
						psijh2 = psijh2(cv::Range(0, psijh2.rows), cv::Range(0, k2));
						// psijh2 = psijh2(:,nonred);

						//Contract xi
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < xi.rows; j++){
									xi.at<float>(j, index) = xi.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k2 columns
						xi = xi(cv::Range(0, xi.rows), cv::Range(0, k2));
						// xi = xi(:,nonred);

						//Contract kappa
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								//Keep data
								for (int j = 0; j < kappa.rows; j++){
									kappa.at<float>(j, index) = kappa.at<float>(j, i);
								}
								index++;
							}
						}
						//Only keep copied k2 columns
						kappa = kappa(cv::Range(0, kappa.rows), cv::Range(0, k2));
						// kappa = kappa(:,nonred);

						//Contract delta2
						std::vector<float> dtmp(k2);
						index = 0;
						for (int i = 0; i < lind.size(); i++){
							if (lind[i] < prop){
								dtmp[index] = delta2[i];
								index++;
							}
						}
						delta2 = dtmp;
						// delta2 = delta2(nonred);

						//Contract tauh2
						tauh2 = std::vector<float>(k2);
						tauh2[0] = delta2[0];
						for (int i = 1; i < delta2.size(); i++){
							tauh2[i] = tauh2[i - 1] * delta2[i];
						}// tauh2 = cumprod(delta2);

						//Update Pgam
						Pgam = cv::Mat(p, k2, CV_32F);
						for (int i = 0; i < p; i++){
							for (int j = 0; j < k2; j++){
								Pgam.at<float>(i, j) = psijh2.at<float>(i, j) * tauh2[j];
							}
						}// Pgam = bsxfun(@times,psijh2,tauh2');
					}
				}
			}
			else{
				//After burn period
				//Save out current matrices
				fitting.rho_v[iter - burn] = rho;
				fitting.kappa_v[iter - burn] = kappa.clone();
				fitting.Gamma_v[iter - burn] = Gamma.clone();
				fitting.xi_v[iter - burn] = xi.clone();
				fitting.Lambda_v[iter - burn] = lambda.clone();
				fitting.ps_v[iter - burn] = ps.clone();
				fitting.etac_v[iter - burn] = eta_c.clone();
				fitting.etanc_v[iter - burn] = eta_nc.clone();
			}
		}

		//Return fitted model
		return fitting;
	}

	void normalizeSamples(std::vector<std::vector<float>> &cradle, std::vector<float> &mean, std::vector<float> &var){
		int s = cradle[0].size();

		//Mean and variance of cradle and non-cradle data are given!
		for (int j = 0; j < cradle.size(); j++){
			for (int i = 0; i < s; i++){
				if (var[i] != 0)
					cradle[j][i] = (cradle[j][i] - mean[i]) / var[i];
			}
		}
	}

	void normalizeNonCradle(std::vector<std::vector<float>> &noncradle, std::vector<float> &mean, std::vector<float> &var){
		int s = noncradle[0].size();

		//Get mean on non-cradle data, for each resolution level
		mean = std::vector<float>(s);
		var = std::vector<float>(s);

		for (int j = 0; j < noncradle.size(); j++){
			for (int i = 0; i < s; i++){
				mean[i] += noncradle[j][i];
			}
		}
		for (int i = 0; i < s; i++){
			mean[i] /= noncradle.size();
		}

		//Get variance
		for (int j = 0; j < noncradle.size(); j++){
			for (int i = 0; i < s; i++){
				float val = noncradle[j][i] - mean[i];
				var[i] += val*val;
			}
		}
		for (int i = 0; i < s; i++){
			var[i] /= noncradle.size();
			var[i] = std::sqrt(var[i]); //sqrt(Var)
		}

		//Get data points to zero mean, unit variance
		for (int j = 0; j < noncradle.size(); j++){
			for (int i = 0; i < s; i++){
				if (var[i] != 0)
					noncradle[j][i] = (noncradle[j][i] - mean[i]) / var[i];
			}
		}
	}

	void unNormalizeSamples(std::vector<std::vector<float>> &cradle, std::vector<float> &mean, std::vector<float> &var){
		int s = cradle[0].size();

		//Mean and variance of cradle and non-cradle data are given!
		for (int j = 0; j < cradle.size(); j++){
			for (int i = 0; i < s; i++){
				if (var[i] != 0)
					cradle[j][i] = cradle[j][i] * var[i];// +mean[i];
			}
		}
	}

	void unNormalizeSamplesCrossSection(std::vector<std::vector<float>> &cradle, std::vector<float> &meanh, std::vector<float> &varh, std::vector<float> &meanv, std::vector<float> &varv){
		int s = cradle[0].size() / 2;

		//Vertical samples
		for (int j = 0; j < cradle.size(); j++){
			for (int i = 0; i < s; i++){
				if (varv[i] != 0)
					cradle[j][i] = cradle[j][i] * varv[i];
			}
		}

		//Horizontal samples
		for (int j = 0; j < cradle.size(); j++){
			for (int i = 0; i < s; i++){
				if (varh[i] != 0)
					cradle[j][s + i] = cradle[j][s + i] * varh[i];
			}
		}
	}

	void sampleDataset(std::vector<std::vector<float>> &dts, std::vector<std::vector<float>> &samples, int cnt){
		//Check if empty
		if (dts.size() == 0){
			samples = std::vector<std::vector<float>>(0);
			return;
		}
		//Get true size of dataset
		int maxind = dts.size(), s = dts[0].size();

		if (maxind < cnt){
			//Select all samples once
			samples = std::vector<std::vector<float>>(maxind);
			for (int i = 0; i < maxind; i++){
				//Copy picked index
				samples[i] = std::vector<float>(s);
				for (int j = 0; j < s; j++){
					samples[i][j] = dts[i][j];
				}
			}
			return;
		}

		samples = std::vector<std::vector<float>>(cnt);

		//Sample cnt samples from specified dataset
		for (int i = 0; i < cnt; i++){
			int ind = std::rand() % maxind;

			//Copy picked index
			samples[i] = std::vector<float>(s);
			for (int j = 0; j < s; j++){
				samples[i][j] = dts[ind][j];
			}
		}
	}

	void reconstructBlock(cv::Mat &texture, std::vector<cv::Mat> &coeffs, int sx, int sy, int csx, int csy, int cex, int cey){

		cv::Mat img;

		//Reconstruct coefficients
		img = FFST::inverseShearletTransformSpect(coeffs);

		//Copy pixels to right location
		for (int j = csx - sx; j < cex - sx; j++){
			for (int k = csy - sy; k < cey - sy; k++){
				texture.at<float>(j + sx, k + sy) = img.at<float>(j, k);
			}
		}
	}

	std::vector<float> mvnpdf(std::vector<std::vector<float>> &X, std::vector<float> &mean, cv::Mat &covar){
		int n = X.size();
		int d = X[0].size();

		//Extract mean
		cv::Mat X0(n, d, CV_32F);
		for (int i = 0; i < n; i++){
			for (int j = 0; j < d; j++){
				X0.at<float>(i, j) = X[i][j] - mean[j];
			}
		}

		//Decompose covariance
		cv::Mat invR, R = Cholesky(covar);
		cv::invert(R, invR);

		//Create standardized data
		X0 = X0 * invR;

		//logSqrtDetSigma = sum(log(diag(R)));
		float logSqrtDetSigma = 0;
		for (int i = 0; i < R.rows; i++)
			logSqrtDetSigma += std::log(R.at<float>(i, i));

		//The quadratic form is the inner products of the standardized data
		std::vector<float> quadform(n);
		for (int i = 0; i < n; i++){
			for (int j = 0; j < d; j++){
				float val = X0.at<float>(i, j);
				quadform[i] += val * val;
			}
		}


		//Get mvnpdf output
		std::vector<float> res(n);
		float tmp = d*std::log(2 * PI) / 2;
		for (int i = 0; i < n; i++){
			res[i] = std::exp(-0.5 * quadform[i] - logSqrtDetSigma - tmp);
		}

		return res;
	}

	std::vector<float> mvnpdf(std::vector<std::vector<float>> &X, cv::Mat &mean, cv::Mat &covar){
		int n = X.size();
		int d = X[0].size();

		//Extract mean
		cv::Mat X0(n, d, CV_32F);
		for (int i = 0; i < n; i++){
			for (int j = 0; j < d; j++){
				X0.at<float>(i, j) = X[i][j] - mean.at<float>(0, j);
			}
		}

		//Decompose covariance
		cv::Mat invR, R = Cholesky(covar);
		cv::invert(R, invR);

		//Create standardized data
		X0 = X0 * invR;

		//logSqrtDetSigma = sum(log(diag(R)));
		float logSqrtDetSigma = 0;
		for (int i = 0; i < R.rows; i++)
			logSqrtDetSigma += std::log(R.at<float>(i, i));

		//The quadratic form is the inner products of the standardized data
		std::vector<float> quadform(n);
		for (int i = 0; i < n; i++){
			for (int j = 0; j < d; j++){
				float val = X0.at<float>(i, j);
				quadform[i] += val * val;
			}
		}


		//Get mvnpdf output
		std::vector<float> res(n);
		float tmp = d*std::log(2 * PI) / 2;
		for (int i = 0; i < n; i++){
			res[i] = std::exp(-0.5 * quadform[i] - logSqrtDetSigma - tmp);
		}

		return res;
	}

	/* Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S */
	cv::Mat Cholesky(cv::Mat &A){
		CV_Assert(A.type() == CV_32F);

		int dim = A.rows;
		cv::Mat S(dim, dim, CV_32F);

		int i, j, k;

		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < i; j++)
				S.at<float>(i, j) = 0.f;

			float sum = 0.f;
			for (k = 0; k < i; k++)
			{
				float val = S.at<float>(k, i);
				sum += val*val;
			}

			S.at<float>(i, i) = std::sqrt(std::max(A.at<float>(i, i) - sum, 0.f));
			float ival = 1.f / S.at<float>(i, i);

			for (j = i + 1; j < dim; j++)
			{
				sum = 0;
				for (k = 0; k < i; k++)
					sum += S.at<float>(k, i) * S.at<float>(k, j);

				S.at<float>(i, j) = (A.at<float>(i, j) - sum)*ival;
			}
		}

		return S;
	}

	cv::Mat CholeskyLower(cv::Mat &A){
		cv::Mat U = Cholesky(A);
		cv::Mat L;
		cv::transpose(U, L);
		return L;
	}
}
