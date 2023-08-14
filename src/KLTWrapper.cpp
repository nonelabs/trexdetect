// Copyright (c) 2016 Kwang Moo Yi.
// All rights reserved.

// This  software  is  strictly   for  non-commercial  use  only.  For
// commercial       use,       please        contact       me       at
// kwang.m<dot>yi<AT>gmail<dot>com.   Also,  when  used  for  academic
// purposes, please cite  the paper "Detection of  Moving Objects with
// Non-stationary Cameras in 5.8ms:  Bringing Motion Detection to Your
// Mobile Device,"  Yi et  al, CVPRW 2013  Redistribution and  use for
// non-commercial purposes  in source  and binary forms  are permitted
// provided that  the above  copyright notice  and this  paragraph are
// duplicated  in   all  such   forms  and  that   any  documentation,
// advertising  materials,   and  other  materials  related   to  such
// distribution and use acknowledge that the software was developed by
// the  Perception and  Intelligence Lab,  Seoul National  University.
// The name of the Perception  and Intelligence Lab and Seoul National
// University may not  be used to endorse or  promote products derived
// from this software without specific prior written permission.  THIS
// SOFTWARE IS PROVIDED ``AS IS''  AND WITHOUT ANY WARRANTIES.  USE AT
// YOUR OWN RISK!

#include "KLTWrapper.hpp"

#include <vector>
#include <opencv2/highgui.hpp>

KLTWrapper::KLTWrapper(void)
{
	// For LK funciton in opencv
	win_size = 10;
	count = 0;
	flags = 0;
}


void KLTWrapper::Init(cv::Mat imgGray)
{
	int ni = imgGray.cols;
	int nj = imgGray.rows;

	// Allocate Maximum possible + some more for safety
	MAX_COUNT = (float (ni) / float (GRID_SIZE_W) + 1.0)*(float (nj) / float (GRID_SIZE_H) + 1.0);

	// Pre-allocate
	image = cv::Mat(imgGray.size(), CV_8UC3);
	imgPrevGray = cv::Mat(imgGray.size(), CV_8UC1);
	pyramid = cv::Mat(imgGray.size(), CV_8UC1);
	prev_pyramid = cv::Mat(imgGray.size(), CV_8UC1);

	points = std::vector<std::vector<cv::Point2f>> (2, std::vector<cv::Point2f>(MAX_COUNT));
	swap_points = std::vector<std::vector<cv::Point2f>> (2, std::vector<cv::Point2f>(MAX_COUNT));
	flags = 0;

	eig = cv::Mat(imgGray.size(), CV_32FC1);
	temp = cv::Mat(imgGray.size(), CV_32FC1);
	maskimg = cv::Mat(imgGray.size(), CV_8UC1);

	// Gen mask
	//BYTE *pMask = (BYTE *) maskimg.data;
	int widthStep = maskimg.step;
	for (int j = 0; j < nj; ++j) {
		for (int i = 0; i < ni; ++i) {
			maskimg.data[i + j * widthStep] = (i >= ni / 5) && (i <= ni * 4 / 5) && (j >= nj / 5) && (j <= nj * 4 / 5) ? (BYTE) 255 : (BYTE) 255;
		}
	}

	// Init homography
	for (int i = 0; i < 9; i++)
		matH[i] = i / 3 == i % 3 ? 1 : 0;
}

void KLTWrapper::InitFeatures(cv::Mat imgGray)
{
	/* automatic initialization */
	double quality = 0.01;
	double min_distance = 10;

	int ni = imgGray.cols;
	int nj = imgGray.rows;

	count = ni / GRID_SIZE_W * nj / GRID_SIZE_H;

	int cnt = 0;
	for (int i = 0; i < ni / GRID_SIZE_W - 1; ++i) {
		for (int j = 0; j < nj / GRID_SIZE_H - 1; ++j) {
			points[1][cnt].x = i * GRID_SIZE_W + GRID_SIZE_W / 2;
			points[1][cnt++].y = j * GRID_SIZE_H + GRID_SIZE_H / 2;
		}
	}

	SwapData(imgGray);
}

void KLTWrapper::RunTrack(cv::Mat imgGray, cv::Mat prevGray)
{
	int i, k;
	int nMatch[MAX_COUNT];

	if (prevGray.empty()) {
		prevGray = imgPrevGray;
	} else {
		flags = 0;
	}
	image.setTo(cv::Scalar(0));
	if (count > 0) {
		cv::calcOpticalFlowPyrLK(
				prevGray, imgGray,
				points[0], points[1], status,cv::noArray(),
				cv::Size(win_size, win_size), 3,
				cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.03),
				flags
		);
		for (i = k = 0; i < count; i++) {
			if (!status[i]) {
				continue;
			}

			nMatch[k++] = i;
		}
		count = k;
	}

	if (count >= 10) {
		// Make homography matrix with correspondences
		MakeHomoGraphy(nMatch, count);
	} else {
		for (int ii = 0; ii < 9; ++ii) {
			matH[ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}

	InitFeatures(imgGray);
}

void KLTWrapper::SwapData(cv::Mat imgGray)
{
	imgGray.copyTo(imgPrevGray);
	std::swap(points[0], points[1]);
}

void KLTWrapper::GetHomography(double *pmatH)
{
	memcpy(pmatH, matH, sizeof(matH));
}

void KLTWrapper::MakeHomoGraphy(int *pnMatch, int nCnt)
{
	std::vector<cv::Point2f> pt1, pt2;
	cv::Mat _h;

	pt1.resize(nCnt);
	pt2.resize(nCnt);
	for (int i = 0; i < nCnt; i++) {
		// REVERSE HOMOGRAPHY
		pt1[i] = points[1][pnMatch[i]];
		pt2[i] = points[0][pnMatch[i]];
	}

	_h = cv::findHomography(pt1, pt2, cv::RANSAC, 1);
	// You can also use cv::LMEDS instead of cv::RANSAC if you prefer.

	if (_h.empty()) {
		return;
	}

	// Copy the homography to your destination (e.g., matH).
	// Assuming matH is a double array with space for 9 elements.
	for (int i = 0; i < 9; i++) {
		matH[i] = _h.at<double>(i / 3, i % 3);
	}
}