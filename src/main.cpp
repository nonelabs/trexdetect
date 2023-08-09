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
    std::cout << "Hello";

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
	cv::VideoCapture capture("woman.mp4");
	if(!capture.isOpened()) {
		std::cerr << "Error: Could not open video file!" << std::endl;
		return -1;
	}


	// Init frame number and exit condition
	int frame_num = 1;
	bool bRun = true;

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

		// Display detection results as overlay
		for (int j = 0; j < frame_copy.cols; ++j) {
			for (int i = 0; i < frame_copy.rows; ++i) {

				float draw_orig = 0.5;

				BYTE *pMaskImg = (BYTE *) (mcdwrapper->detect_img.data);
				int widthstepMsk = mcdwrapper->detect_img.step;

				int mask_data = pMaskImg[i + j * widthstepMsk];

				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2];
				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 1] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 1];
				((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 0] = draw_orig * ((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 0];

				if (frame_num > 1) {
					((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2] += mask_data > 0 ? 255 * (1.0 - draw_orig) : 0;
				}

			}
		}

        cv::imshow("Video", frame);

		// // DISPLAY---- and write to png
		// cvWriteFrame(pVideoOut, frame_copy);

		if (atoi((char *)(argv[2])) > 0 && frame_num >= 1) {

			for (int j = 0; j < frame_copy.cols; ++j) {
				for (int i = 0; i < frame_copy.rows; ++i) {

					float draw_orig = 0.5;

					BYTE *pMaskImg = (BYTE *) (mcdwrapper->detect_img.data);
					int widthstepMsk = mcdwrapper->detect_img.step;

					int mask_data = pMaskImg[i + j * widthstepMsk];

					((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 2] = mask_data > 0 ? 255 : 0;
					((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 1] = mask_data > 0 ? 255 : 0;
					((BYTE *) (frame_copy.data))[i * 3 + j * frame_copy.step + 0] = mask_data > 0 ? 255 : 0;
				}
			}

			char bufbuf[1000];
			sprintf(bufbuf, "./results/%s_frm%05d.png", test_file, frame_num);
            cv::imwrite(bufbuf, frame_copy);
		}
        cv::waitKey(1000);
		++frame_num;

	}

	return 0;
}
