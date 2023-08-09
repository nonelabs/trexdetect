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

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#define GRID_SIZE_W (32)
#define GRID_SIZE_H (24)

typedef unsigned char BYTE;

class KLTWrapper {
 private:
	cv::Mat eig;
	cv::Mat temp;
	cv::Mat maskimg;

	// For LK
	cv::Mat image;
	cv::Mat imgPrevGray, pyramid, prev_pyramid, swap_temp;
	int win_size;
	int MAX_COUNT;
	std::vector<std::vector<cv::Point2f>> points, swap_points;
	std::vector<uchar> status;
	int count;
	int flags;

	// For Homography Matrix
	double matH[9];

 private:
	void SwapData(cv::Mat imgGray);
	void MakeHomoGraphy(int *pnMatch, int nCnt);

 public:
	 KLTWrapper(void);
	~KLTWrapper(void);

	void Init(cv::Mat imgGray);
	void InitFeatures(cv::Mat imgGray);
	void RunTrack(cv::Mat imgGray, cv::Mat prevGray);	// with MakeHomography
	void GetHomography(double *pmatH);
};
