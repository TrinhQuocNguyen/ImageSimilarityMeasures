//
// Copyright(c) Multimedia Signal Processing Group (MMSPG),
//              Ecole Polytechnique Fédérale de Lausanne (EPFL)
//              http://mmspg.epfl.ch
// All rights reserved.
// Author: Philippe Hanhart (philippe.hanhart@epfl.ch)
//
// Permission is hereby granted, without written agreement and without
// license or royalty fees, to use, copy, modify, and distribute the
// software provided and its documentation for research purpose only,
// provided that this copyright notice and the original authors' names
// appear on all copies and supporting documentation.
// The software provided may not be commercially distributed.
// In no event shall the Ecole Polytechnique Fédérale de Lausanne (EPFL)
// be liable to any party for direct, indirect, special, incidental, or
// consequential damages arising out of the use of the software and its
// documentation.
// The Ecole Polytechnique Fédérale de Lausanne (EPFL) specifically
// disclaims any warranties.
// The software provided hereunder is on an "as is" basis and the Ecole
// Polytechnique Fédérale de Lausanne (EPFL) has no obligation to provide
// maintenance, support, updates, enhancements, or modifications.
//

/**************************************************************************

 Usage:
  VQMT.exe OriginalVideo ProcessedVideo Wi Heightdth NumberOfFrames ChromaFormat Output Metrics

  OriginalVideo: the original video as raw YUV video file, progressively scanned, and 8 bits per sample
  ProcessedVideo: the processed video as raw YUV video file, progressively scanned, and 8 bits per sample
  Height: the height of the video
  Width: the width of the video
  NumberOfFrames: the number of frames to process
  ChromaFormat: the chroma subsampling format. 0: YUV400, 1: YUV420, 2: YUV422, 3: YUV444
  Output: the name of the output file(s)
  Metrics: the list of metrics to use
   available metrics:
   - PSNR: Peak Signal-to-Noise Ratio (PNSR)
   - SSIM: Structural Similarity (SSIM)
   - MSSSIM: Multi-Scale Structural Similarity (MS-SSIM)
   - VIFP: Visual Information Fidelity, pixel domain version (VIFp)
   - PSNRHVS: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF) (PSNR-HVS)
   - PSNRHVSM: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF) and between-coefficient contrast masking of DCT basis functions (PSNR-HVS-M)

 Example:
  VQMT.exe original.yuv processed.yuv 1088 1920 250 1 results PSNR SSIM MSSSIM VIFP
  will create the following output files in CSV (comma-separated values) format:
  - results_pnsr.csv
  - results_ssim.csv
  - results_msssim.csv
  - results_vifp.csv

 Notes:
 - SSIM comes for free when MSSSIM is computed (but you still need to specify it to get the output)
 - PSNRHVS and PSNRHVSM are always computed at the same time (but you still need to specify both to get the two outputs)
 - When using MSSSIM, the height and width of the video have to be multiple of 16
 - When using VIFP, the height and width of the video have to be multiple of 8

 Changes in version 1.1 (since 1.0) on 30/3/13
 - Added support for large files (>2GB)
 - Added support for different chroma sampling formats (YUV400, YUV420, YUV422, and YUV444)

**************************************************************************/

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2\opencv.hpp>
#include "VideoYUV.hpp"
#include "PSNR.hpp"
#include "SSIM.hpp"
#include "MSSSIM.hpp"
#include "VIFP.hpp"
#include "PSNRHVS.hpp"
#include "qm.cpp"

using namespace std;

//#define C1 (float) (0.01 * 255 * 0.01  * 255)
//#define C2 (float) (0.03 * 255 * 0.03  * 255)

//
////sigma on block_size
//double sigma(Mat & m, int i, int j, int block_size)
//{
//	double sd = 0;
//
//	Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
//	Mat m_squared(block_size, block_size, CV_64F);
//
//	multiply(m_tmp, m_tmp, m_squared);
//
//	// E(x)
//	double avg = mean(m_tmp)[0];
//	// E(x²)
//	double avg_2 = mean(m_squared)[0];
//
//
//	sd = sqrt(avg_2 - avg * avg);
//
//	return sd;
//}
//
//// Covariance
//double cov(Mat & m1, Mat & m2, int i, int j, int block_size)
//{
//	Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
//	Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
//	Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));
//
//
//	multiply(m1_tmp, m2_tmp, m3);
//
//	double avg_ro = mean(m3)[0]; // E(XY)
//	double avg_r = mean(m1_tmp)[0]; // E(X)
//	double avg_o = mean(m2_tmp)[0]; // E(Y)
//
//
//	double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)
//
//	return sd_ro;
//}
//
//// Mean squared error
//double eqm(Mat & img1, Mat & img2)
//{
//	int i, j;
//	double eqm = 0;
//	int height = img1.rows;
//	int width = img1.cols;
//
//	for (i = 0; i < height; i++)
//		for (j = 0; j < width; j++)
//			eqm += (img1.at<double>(i, j) - img2.at<double>(i, j)) * (img1.at<double>(i, j) - img2.at<double>(i, j));
//
//	eqm /= height * width;
//
//	return eqm;
//}
//
//
//
///**
//*	Compute the PSNR between 2 images
//*/
//double psnr(Mat & img_src, Mat & img_compressed, int block_size)
//{
//	int D = 255;
//	return (10 * log10((D*D) / eqm(img_src, img_compressed)));
//}
//
//
///**
//* Compute the SSIM between 2 images
//*/
//double ssim(Mat & img_src, Mat & img_compressed, int block_size, bool show_progress = false)
//{
//	double ssim = 0;
//
//	int nbBlockPerHeight = img_src.rows / block_size;
//	int nbBlockPerWidth = img_src.cols / block_size;
//
//	for (int k = 0; k < nbBlockPerHeight; k++)
//	{
//		for (int l = 0; l < nbBlockPerWidth; l++)
//		{
//			int m = k * block_size;
//			int n = l * block_size;
//
//			double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
//			double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
//			double sigma_o = sigma(img_src, m, n, block_size);
//			double sigma_r = sigma(img_compressed, m, n, block_size);
//			double sigma_ro = cov(img_src, img_compressed, m, n, block_size);
//
//			ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
//
//		}
//		// Progress
//		if (show_progress)
//			cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]";
//	}
//	ssim /= nbBlockPerHeight * nbBlockPerWidth;
//
//	if (show_progress)
//	{
//		cout << "\r>>SSIM [100%]" << endl;
//		cout << "SSIM : " << ssim << endl;
//	}
//
//	return ssim;
//}
//
//void compute_quality_metrics(char * file1, char * file2, int block_size)
//{
//
//	Mat img_src;
//	Mat img_compressed;
//
//	// Loading pictures
//	img_src = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
//	img_compressed = imread(file2, CV_LOAD_IMAGE_GRAYSCALE);
//
//
//	img_src.convertTo(img_src, CV_64F);
//	img_compressed.convertTo(img_compressed, CV_64F);
//
//	int height_o = img_src.rows;
//	int height_r = img_compressed.rows;
//	int width_o = img_src.cols;
//	int width_r = img_compressed.cols;
//
//	// Check pictures size
//	if (height_o != height_r || width_o != width_r)
//	{
//		cout << "Images must have the same dimensions" << endl;
//		return;
//	}
//
//	// Check if the block size is a multiple of height / width
//	if (height_o % block_size != 0 || width_o % block_size != 0)
//	{
//		cout << "WARNING : Image WIDTH and HEIGHT should be divisible by BLOCK_SIZE for the maximum accuracy" << endl
//			<< "HEIGHT : " << height_o << endl
//			<< "WIDTH : " << width_o << endl
//			<< "BLOCK_SIZE : " << block_size << endl
//			<< endl;
//	}
//
//	double ssim_val = ssim(img_src, img_compressed, block_size);
//	double psnr_val = psnr(img_src, img_compressed, block_size);
//
//	cout << "SSIM : " << ssim_val << endl;
//	cout << "PSNR : " << psnr_val << endl;
//}


enum Params {
	PARAM_ORIGINAL = 1,	// Original video stream (YUV)
	PARAM_PROCESSED,	// Processed video stream (YUV)
	PARAM_HEIGHT,		// Height
	PARAM_WIDTH,		// Width
	PARAM_NBFRAMES,		// Number of frames
	PARAM_CHROMA,		// Chroma format
	PARAM_RESULTS,		// Output file for results
	PARAM_METRICS,		// Metric(s) to compute
	PARAM_SIZE
};

enum Metrics {
	METRIC_PSNR = 0,
	METRIC_SSIM,
	METRIC_MSSSIM,
	METRIC_VIFP,
	METRIC_PSNRHVS,
	METRIC_PSNRHVSM,
	METRIC_SIZE
};

int main(int argc, const char *argv[])
{
	//int height = 640;
	//int width = 640;
	///*SSIM *ssim = new SSIM(height, width);
	//MSSSIM *msssim = new MSSSIM(height, width);
	//PSNR *psnr = new PSNR(height, width);*/

	//cv::Mat originalImage = cv::imread("./data/raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat noiseImage = cv::imread("./data/noise_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::resize(noiseImage, noiseImage, cv::Size(width, height));

	//cv::Mat m_image = cv::imread("./data/m_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat g_image = cv::imread("./data/g_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);

	////float result = cv::PSNR(originalImage, g_image);
	////float from_code = ssim->compute(originalImage, noiseImage);
	//
	////printf("%f", result);

	//double result = psnr(originalImage, noiseImage, 64);
	//cout <<"result: " << result << endl;
	////cout << "from code: "<< from_code << endl;

	//////////////////////////////////////////////////
	// default settings
	double C1 = 6.5025, C2 = 58.5225;





	IplImage
		*img1 = NULL, *img2 = NULL, *img1_img2 = NULL,
		*img1_temp = NULL, *img2_temp = NULL,
		*img1_sq = NULL, *img2_sq = NULL,
		*mu1 = NULL, *mu2 = NULL,
		*mu1_sq = NULL, *mu2_sq = NULL, *mu1_mu2 = NULL,
		*sigma1_sq = NULL, *sigma2_sq = NULL, *sigma12 = NULL,
		*ssim_map = NULL, *temp1 = NULL, *temp2 = NULL, *temp3 = NULL;


	/***************************** INITS **********************************/
	cv::Mat originalImage = cv::imread("./data/raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat noiseImage = cv::imread("./data/noise_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	cv::resize(noiseImage, noiseImage, cv::Size(640, 640));

	cv::Mat m_image = cv::imread("./data/m_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat g_image = cv::imread("./data/g_raindrop0113.jpg", CV_LOAD_IMAGE_COLOR);

	img1_temp = cvLoadImage("./data/raindrop0113.jpg");
	img2_temp = cvLoadImage("./data/m_raindrop0113.jpg");

	if (img1_temp == NULL || img2_temp == NULL)
		return -1;

	int x = img1_temp->width, y = img1_temp->height;
	int nChan = img1_temp->nChannels, d = IPL_DEPTH_32F;
	CvSize size = cvSize(x, y);

	img1 = cvCreateImage(size, d, nChan);
	img2 = cvCreateImage(size, d, nChan);

	cvConvert(img1_temp, img1);
	cvConvert(img2_temp, img2);
	cvReleaseImage(&img1_temp);
	cvReleaseImage(&img2_temp);


	img1_sq = cvCreateImage(size, d, nChan);
	img2_sq = cvCreateImage(size, d, nChan);
	img1_img2 = cvCreateImage(size, d, nChan);

	cvPow(img1, img1_sq, 2);
	cvPow(img2, img2_sq, 2);
	cvMul(img1, img2, img1_img2, 1);

	mu1 = cvCreateImage(size, d, nChan);
	mu2 = cvCreateImage(size, d, nChan);

	mu1_sq = cvCreateImage(size, d, nChan);
	mu2_sq = cvCreateImage(size, d, nChan);
	mu1_mu2 = cvCreateImage(size, d, nChan);


	sigma1_sq = cvCreateImage(size, d, nChan);
	sigma2_sq = cvCreateImage(size, d, nChan);
	sigma12 = cvCreateImage(size, d, nChan);

	temp1 = cvCreateImage(size, d, nChan);
	temp2 = cvCreateImage(size, d, nChan);
	temp3 = cvCreateImage(size, d, nChan);

	ssim_map = cvCreateImage(size, d, nChan);
	/*************************** END INITS **********************************/


	//////////////////////////////////////////////////////////////////////////
	// PRELIMINARY COMPUTING
	cvSmooth(img1, mu1, CV_GAUSSIAN, 11, 11, 1.5);
	cvSmooth(img2, mu2, CV_GAUSSIAN, 11, 11, 1.5);

	cvPow(mu1, mu1_sq, 2);
	cvPow(mu2, mu2_sq, 2);
	cvMul(mu1, mu2, mu1_mu2, 1);


	cvSmooth(img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5);
	cvAddWeighted(sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq);

	cvSmooth(img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5);
	cvAddWeighted(sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq);

	cvSmooth(img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5);
	cvAddWeighted(sigma12, 1, mu1_mu2, -1, 0, sigma12);


	//////////////////////////////////////////////////////////////////////////
	// FORMULA

	// (2*mu1_mu2 + C1)
	cvScale(mu1_mu2, temp1, 2);
	cvAddS(temp1, cvScalarAll(C1), temp1);

	// (2*sigma12 + C2)
	cvScale(sigma12, temp2, 2);
	cvAddS(temp2, cvScalarAll(C2), temp2);

	// ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	cvMul(temp1, temp2, temp3, 1);

	// (mu1_sq + mu2_sq + C1)
	cvAdd(mu1_sq, mu2_sq, temp1);
	cvAddS(temp1, cvScalarAll(C1), temp1);

	// (sigma1_sq + sigma2_sq + C2)
	cvAdd(sigma1_sq, sigma2_sq, temp2);
	cvAddS(temp2, cvScalarAll(C2), temp2);

	// ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
	cvMul(temp1, temp2, temp1, 1);

	// ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
	cvDiv(temp3, temp1, ssim_map, 1);


	CvScalar index_scalar = cvAvg(ssim_map);

	// through observation, there is approximately 
	// 1% error max with the original matlab program

	cout << "(R, G & B SSIM index)" << endl;
	cout << index_scalar.val[2] * 100 << "%" << endl;
	cout << index_scalar.val[1] * 100 << "%" << endl;
	cout << index_scalar.val[0] * 100 << "%" << endl;

	// if you use this code within a program
	// don't forget to release the IplImages
	return 0;


}

int old_main(int argc, const char *argv[]) {
	// Check number of input parameters
	if (argc < PARAM_SIZE) {
		fprintf(stderr, "Check software usage: at least %d parameters are required.\n", PARAM_SIZE);
		exit(EXIT_FAILURE);
	}

	double duration;
	duration = static_cast<double>(cv::getTickCount());

	// Input parameters
	unsigned int height = atoi(argv[PARAM_HEIGHT]);
	unsigned int width = atoi(argv[PARAM_WIDTH]);
	unsigned int nbframes = atoi(argv[PARAM_NBFRAMES]);
	unsigned int chroma = atoi(argv[PARAM_CHROMA]);

	// Input video streams
	VideoYUV *original = new VideoYUV(argv[PARAM_ORIGINAL], height, width, nbframes, chroma);
	VideoYUV *processed = new VideoYUV(argv[PARAM_PROCESSED], height, width, nbframes, chroma);

	// Output files for results
	FILE *result_file[METRIC_SIZE] = { NULL };
	char *str = new char[256];
	for (int i = 7; i < argc; i++) {
		if (strcmp(argv[i], "PSNR") == 0) {
			sprintf(str, "%s_psnr.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_PSNR] = fopen(str, "w");
		}
		else if (strcmp(argv[i], "SSIM") == 0) {
			sprintf(str, "%s_ssim.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_SSIM] = fopen(str, "w");
		}
		else if (strcmp(argv[i], "MSSSIM") == 0) {
			sprintf(str, "%s_msssim.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_MSSSIM] = fopen(str, "w");
		}
		else if (strcmp(argv[i], "VIFP") == 0) {
			sprintf(str, "%s_vifp.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_VIFP] = fopen(str, "w");
		}
		else if (strcmp(argv[i], "PSNRHVS") == 0) {
			sprintf(str, "%s_psnrhvs.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_PSNRHVS] = fopen(str, "w");
		}
		else if (strcmp(argv[i], "PSNRHVSM") == 0) {
			sprintf(str, "%s_psnrhvsm.csv", argv[PARAM_RESULTS]);
			result_file[METRIC_PSNRHVSM] = fopen(str, "w");
		}
	}
	delete[] str;

	// Check size for VIFp downsampling
	if (result_file[METRIC_VIFP] != NULL && (height % 8 != 0 || width % 8 != 0)) {
		fprintf(stderr, "VIFp: 'height' and 'width' have to be multiple of 8.\n");
		exit(EXIT_FAILURE);
	}
	// Check size for MS-SSIM downsampling
	if (result_file[METRIC_MSSSIM] != NULL && (height % 16 != 0 || width % 16 != 0)) {
		fprintf(stderr, "MS-SSIM: 'height' and 'width' have to be multiple of 16.\n");
		exit(EXIT_FAILURE);
	}

	// Print header to file
	for (int m = 0; m < METRIC_SIZE; m++) {
		if (result_file[m] != NULL) {
			fprintf(result_file[m], "frame,value\n");
		}
	}

	//PSNR *psnr = new PSNR(height, width);
	SSIM *ssim = new SSIM(height, width);
	MSSSIM *msssim = new MSSSIM(height, width);
	VIFP *vifp = new VIFP(height, width);
	PSNRHVS *phvs = new PSNRHVS(height, width);

	cv::Mat original_frame(height, width, CV_32F), processed_frame(height, width, CV_32F);
	float result[METRIC_SIZE] = { 0 };
	double result_avg[METRIC_SIZE] = { 0 };

	for (unsigned int frame = 0; frame < nbframes; frame++) {
		// Grab frame
		if (!original->readOneFrame()) exit(EXIT_FAILURE);
		original->getLuma(original_frame, CV_32F);
		if (!processed->readOneFrame()) exit(EXIT_FAILURE);
		processed->getLuma(processed_frame, CV_32F);

		// Compute PSNR
		if (result_file[METRIC_PSNR] != NULL) {
			//result[METRIC_PSNR] = psnr->compute(original_frame, processed_frame);
		}

		// Compute SSIM and MS-SSIM
		if (result_file[METRIC_SSIM] != NULL && result_file[METRIC_MSSSIM] == NULL) {
			result[METRIC_SSIM] = ssim->compute(original_frame, processed_frame);
		}
		if (result_file[METRIC_MSSSIM] != NULL) {
			msssim->compute(original_frame, processed_frame);
			if (result_file[METRIC_SSIM] != NULL) {
				result[METRIC_SSIM] = msssim->getSSIM();
			}
			result[METRIC_MSSSIM] = msssim->getMSSSIM();
		}

		// Compute VIFp
		if (result_file[METRIC_VIFP] != NULL) {
			result[METRIC_VIFP] = vifp->compute(original_frame, processed_frame);
		}

		// Compute PSNR-HVS and PSNR-HVS-M
		if (result_file[METRIC_PSNRHVS] != NULL || result_file[METRIC_PSNRHVSM] != NULL) {
			phvs->compute(original_frame, processed_frame);
			if (result_file[METRIC_PSNRHVS] != NULL) {
				result[METRIC_PSNRHVS] = phvs->getPSNRHVS();
			}
			if (result_file[METRIC_PSNRHVSM] != NULL) {
				result[METRIC_PSNRHVSM] = phvs->getPSNRHVSM();
			}
		}

		// Print quality index to file
		for (int m = 0; m < METRIC_SIZE; m++) {
			if (result_file[m] != NULL) {
				result_avg[m] += result[m];
				fprintf(result_file[m], "%d,%0.6f\n", frame, result[m]);
			}
		}
	}

	// Print average quality index to file
	for (int m = 0; m < METRIC_SIZE; m++) {
		if (result_file[m] != NULL) {
			result_avg[m] /= double(nbframes);
			fprintf(result_file[m], "average,%0.6f", result_avg[m]);
			fclose(result_file[m]);
		}
	}

	//delete psnr;
	delete ssim;
	delete msssim;
	delete vifp;
	delete phvs;
	delete original;
	delete processed;

	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	printf("Time: %0.3fs\n", duration);

	return EXIT_SUCCESS;
}
