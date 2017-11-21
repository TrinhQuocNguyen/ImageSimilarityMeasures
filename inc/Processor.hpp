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
#include <fstream>

using namespace std;

class  Processor{
public:
	Processor(int h, int w);
	std::ofstream Processor::createFileIfNotExist(string fileResultPath);
	void Processor::processImages(int width, int height, string folderPath1, string folderPath2, string fileResultPath, bool previewOn);
	void Processor::processFolders(int width, int height, string origialFolderPath, list<string> processedFolderPaths, string resultFolderPath);
	void Processor::CreateFolder(const char * path);

};

#endif