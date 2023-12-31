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

#ifndef	_MCDWRAPPER_CPP_
#define	_MCDWRAPPER_CPP_

#include <ctime>
#include <cstring>
#include "MCDWrapper.hpp"
#include "params.hpp"

#if defined _WIN32 || defined _WIN64
int gettimeofday(struct timeval *tp, int *tz)
{
	LARGE_INTEGER tickNow;
	static LARGE_INTEGER tickFrequency;
	static BOOL tickFrequencySet = FALSE;
	if (tickFrequencySet == FALSE) {
		QueryPerformanceFrequency(&tickFrequency);
		tickFrequencySet = TRUE;
	}
	QueryPerformanceCounter(&tickNow);
	tp->tv_sec = (long)(tickNow.QuadPart / tickFrequency.QuadPart);
	tp->tv_usec = (long)(((tickNow.QuadPart % tickFrequency.QuadPart) * 1000000L) / tickFrequency.QuadPart);

	return 0;
}
#else
#include <sys/time.h>
#endif


void
 MCDWrapper::Init(cv::Mat in_imgIpl)
{

	frm_cnt = 0;
	imgIpl = in_imgIpl;
	cv::Size imageSize(in_imgIpl.cols, in_imgIpl.rows);
	imgIplTemp    = cv::Mat (imageSize, CV_8UC1);
	imgGray 	  = cv::Mat (imageSize, CV_8UC1);
	imgGrayPrev   = cv::Mat (imageSize, CV_8UC1);
	imgGaussLarge = cv::Mat (imageSize, CV_8UC1);
	imgGaussSmall = cv::Mat (imageSize, CV_8UC1);
	imgDOG 	      = cv::Mat (imageSize, CV_8UC1);
	detect_img    = cv::Mat (imageSize, CV_8UC1);

	cv::cvtColor(imgIpl, imgIplTemp, cv::COLOR_BGR2GRAY);
	cv::medianBlur(imgIplTemp, imgGray, 5);


	m_LucasKanade.Init(imgGray);
	BGModel.init(imgGray);
	imgGray.copyTo(imgGrayPrev);
}

void MCDWrapper::Run()
{

	frm_cnt++;

	timeval tic, toc, tic_total, toc_total;
	float rt_preProc;	// pre Processing time
	float rt_motionComp;	// motion Compensation time
	float rt_modelUpdate;	// model update time
	float rt_total;		// Background Subtraction time

	//--TIME START
	cv::cvtColor(imgIpl, imgIplTemp, cv::COLOR_BGR2GRAY);
	gettimeofday(&tic, NULL);
	// Smmothign using median filter
	cv::medianBlur(imgIplTemp, imgGray, 5);

	//--TIME END
	gettimeofday(&toc, NULL);
	rt_preProc = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Calculate Backward homography
	// Get H
	double h[9];
	m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
	m_LucasKanade.GetHomography(h);
	BGModel.motionCompensate(h);

	//--TIME END
	gettimeofday(&toc, NULL);
	rt_motionComp = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Update BG Model and Detect
	BGModel.update(detect_img);
	//--TIME END
	gettimeofday(&toc, NULL);
	rt_modelUpdate = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	rt_total = rt_preProc + rt_motionComp + rt_modelUpdate;

	// Debug display of individual maps
	// cv::Mat mean = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Mean[0]);
	// cv::imshow("mean",mean/255.0);
	// cv::Mat var = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Var[0]);
	// cv::imshow("var",var/255.0);
	// cv::Mat age = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Age[0]);
	// cv::imshow("age",age/255.0);

	//////////////////////////////////////////////////////////////////////////
	// Debug Output
	for (int i = 0; i < 100; ++i) {
		printf("\b");
	}
	printf("PP: %.2f(ms)\tOF: %.2f(ms)\tBGM: %.2f(ms)\tTotal time: \t%.2f(ms)", MAX(0.0, rt_preProc), MAX(0.0, rt_motionComp), MAX(0.0, rt_modelUpdate), MAX(0.0, rt_total));

	// Uncomment this block if you want to save runtime to txt
	// if(rt_preProc >= 0 && rt_motionComp >= 0 && rt_modelUpdate >= 0 && rt_total >= 0){
	//      FILE* fileRunTime = fopen("runtime.txt", "a");
	//      fprintf(fileRunTime, "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", rt_preProc, rt_motionComp, rt_modelUpdate, 0.0, rt_total);
	//      fclose(fileRunTime);
	// }

	imgGray.copyTo(imgGrayPrev);

}

#endif				// _MCDWRAPPER_CPP_
