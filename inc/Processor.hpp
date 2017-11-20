#ifndef PROCESSOR_hpp
#define PROCESSOR_hpp

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

using namespace std;

class  Processor {
public:
	Processor();
	void Processor::processImages(string path1, string path2, string fileResult);

};

#endif