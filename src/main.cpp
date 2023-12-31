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

#include "main.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

int main(int argc, char *argv[])
{

	char test_file[100];
	char abs_file[1000];

	if (argc != 3) {
		printf("Usage: Binary [VideoFileFullPathWithExt] [bool DumpOutputImgs (1/0)]\n");
		return -1;
	}

	/************************************************************************/
	/*  Get MP4 Name exluding all paths                                     */
	/************************************************************************/
	strncpy(abs_file, (char *)(argv[1]), 1000);
	//first run to find '/' token
	int last_token_pos = 0;
	for (int i = 0; abs_file[i] != '\0'; ++i) {
		if (abs_file[i] == '/' || abs_file[i] == '\\')
			last_token_pos = i;
	}
	int tmpidx = 0;
	for (int i = last_token_pos; abs_file[i] != '.'; ++i) {
		test_file[tmpidx++] = abs_file[i];
	}
	test_file[tmpidx] = '\0';

	/************************************************************************/
	/*  Initialize Variables                                                */
	/************************************************************************/

	// wrapper class for mcd
	MCDWrapper *mcdwrapper = new MCDWrapper();

	// OPEN CV VARIABLES
	const char window_name[] = "OUTPUT";
	cv::Mat frame;
    cv::Mat frame_copy;
    cv::Mat vil_conv;
    cv::Mat raw_img;
    cv::Mat model_copy;
    cv::Mat model_img;
	cv::Mat edge;

	// File name strings
	string infile_name;
	infile_name.append(abs_file);
	// Create output dir
	//system("mkdir ./results");
	string mp4file_name("./results/");
	mp4file_name.append(test_file);
	mp4file_name.append("_result");
	mp4file_name.append(".mp4");

	// Initialize capture
	cv::VideoCapture capture("video2.mp4");
	if(!capture.isOpened()) {
		std::cerr << "Error: Could not open video file!" << std::endl;
		return -1;
	}


	// Init frame number and exit condition
	int frame_num = 1;
	bool bRun = true;
    cv::Mat ff;
    for (int i=0;i<1000;i++)
        capture.read(ff);

	/************************************************************************/
	/*  The main process loop                                               */
	/************************************************************************/
	while (true) {	// the main loop

		cv::Mat frame;

		// Capture a frame
		if (!capture.read(frame)) {
			std::cout << "Reached the end of the video." << std::endl;
			break;
		}
        frame.copyTo(raw_img);
        frame.copyTo(frame_copy);

		if (frame_num == 1) {

			// Init the wrapper for first frame
			mcdwrapper->Init(raw_img);

		} else {

			// Run detection
			mcdwrapper->Run();

		}

        cv::Mat thresholded;
        cv::threshold(mcdwrapper->detect_img, thresholded, 1, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double minAreaSize = 1; // You can adjust this value
        double maxAreaSize = 40000; // You can adjust this value

        // Remove small blobs by filling them with black color
        for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < minAreaSize || cv::contourArea(contours[i]) > maxAreaSize) {
            cv::drawContours(thresholded, contours, (int) i, cv::Scalar(0), -1, cv::LINE_8, hierarchy, 0);
        }
        }


        cv::findContours(thresholded, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat result = thresholded.clone();
        result.setTo(0);
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect boundingBox = cv::boundingRect(contours[i]);
            cv::rectangle(result, boundingBox, 1, 2);
        }



		for (int j = 0; j < frame_copy.rows; ++j) {
			for (int i = 0; i < frame_copy.cols; ++i) {

				float draw_orig = 1.0;

				int widthstepMsk = mcdwrapper->detect_img.step;

				//int mask_data = mcdwrapper->detect_img.data[i + j * widthstepMsk];
                //int mask_data = mcdwrapper->detect_img.data[i + j * widthstepMsk];
                int mask_data = result.data[i + j * widthstepMsk];


				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2];
				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 1] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 1];
				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 0] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 0];

				if (frame_num > 1) {
					((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2] = mask_data > 0 ? 255 * (1.0 ) : ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2];
				}

			}
		}

        cv::imshow("Video", frame_copy);
        cv::waitKey(1);
		++frame_num;

	}

	return 0;
}
