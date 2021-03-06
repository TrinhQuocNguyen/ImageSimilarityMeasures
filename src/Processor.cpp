#include "Processor.hpp"
#include <windows.h>
#include <stdio.h>

Processor::Processor(int h, int w){

}
/**
Create a folder if it was not existed
*/
void Processor::CreateFolder(const char * path)
{
	if (!CreateDirectoryA(path, NULL))
	{
		return;
	}
}
/**
Create the file if it was not existed (In this case, make result-report files)
*/
std::ofstream Processor::createFileIfNotExist(string fileResultPath) {

	CreateFolder("./../results");
	std::ofstream fp;
	fp.open(fileResultPath, ofstream::out);
	if (!fp.is_open()) {
		cout << "Can not create the file in path: " << fileResultPath << endl;
		return fp;
	}
	else{
		return fp;
	}
}

/**
Get the string of current time in format: yyMMdd_HHmmss
*/
string getOutputFileName() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
	string str = oss.str();

	return str;
}
/**
Get the last string of the path
*/
string splitFilename( string str)
{
	string result[2];
	size_t found;
	found = str.find_last_of("/\\");
	result[0] = str.substr(0, found); // folder
	result[1] = str.substr(found + 1); // file
	return str.substr(found + 1);
}
/**
The function compares the similarity of pairs of images in originalFolder to the other folders
*/
void Processor::processFolders(int width, int height, string originalFolderPath , list<string> processedFolderPaths, string resultFolderPath) {
	// compare images in origianl folder to every folders in the list of processedFolderPaths
	for (list<string>::iterator it = processedFolderPaths.begin(); it != processedFolderPaths.end(); it++) {
		processImages(width, height, originalFolderPath, *it, resultFolderPath +"/"+ splitFilename(originalFolderPath) + "_" + splitFilename(*it)+ ".txt", false );
	}
}

/** @processImages 

The function compares the similarity of pairs of images in 2 folders
@param width 
@param height 
@param folderPath1 
@param folderPath2 
@param fileResultPath 
@param previewOn
*/
void Processor::processImages(int width, int height, string folderPath1, string folderPath2, string fileResultPath, bool previewOn) {

	// create file to write
	std:ofstream fp = createFileIfNotExist(fileResultPath+"."+ getOutputFileName());

	// Init methods
	SSIM *ssim = new SSIM(height, width);
	MSSSIM *msssim = new MSSSIM(height, width);
	PSNR *psnr = new PSNR(height, width);
	VIFP *vifp = new VIFP(height, width);

	float ssimMes = 0, msssimMes = 0, psnrMes = 0, vifpMes = 0;
	float ssimSum = 0, msssimSum = 0, psnrSum = 0, vifpSum = 0;

	// get all the image in folderPath1 (normally it should be the original folder)
	vector<cv::String> originalFiles;
	cv::glob(folderPath1, originalFiles);
	int count = 0;

	for (auto & file : originalFiles)
	{
		//std::cout << "Loading drop file " << file << std::endl;
		cv::Mat originalImage = cv::imread(file);

		// get the mask (predict)
		size_t pos = file.find("raindrop");
		cv::String iFileName = file.substr(pos);

		// load corresponding image
		cv::Mat noiseImage = cv::imread(folderPath2 + "/" + iFileName, CV_LOAD_IMAGE_COLOR);
		if (noiseImage.empty()) {
			cout << "Can not load the corresponding image" << endl;
		}

		// resize to nomalize the images
		cv::resize(originalImage, originalImage, cv::Size(width, height));
		cv::resize(noiseImage, noiseImage, cv::Size(width, height));

		// caculate 
		psnrMes= cv::PSNR(originalImage, noiseImage);
		ssimMes = ssim->compute(originalImage, noiseImage);
		msssimMes = msssim->compute(originalImage, noiseImage);
		originalImage.convertTo(originalImage, CV_32F);
		noiseImage.convertTo(noiseImage, CV_32F);
		vifpMes= vifp->compute(originalImage, noiseImage);

		// add to sum
		ssimSum += ssimMes;
		msssimSum += msssimMes;
		psnrSum += psnrMes;
		vifpSum +=  vifpMes;

		// compare 2 corresponding images
		cout << "----------------------" << endl;
		cout << "Similarity average of images in 2 folders: " << folderPath1 << " and " << folderPath2 << endl;
		cout << "Similarity of 2 images: " << iFileName << endl;
		cout << "PSNR   : " << psnrMes << endl;
		cout << "SSIM   : " << ssimMes<< endl;
		cout << "MSSSIM : " << msssimMes << endl;
		cout << "VIFP   : " << vifpMes << endl;

		// write to file
		fp << "----------------------" << endl;
		fp << "Similarity average of images in 2 folders: " << folderPath1 << " and " << folderPath2 << endl;
		fp << "Similarity of 2 images: " << iFileName << endl;
		fp << "PSNR   : " << psnrMes << endl;
		fp << "SSIM   : " << ssimMes << endl;
		fp << "MSSSIM : " << msssimMes << endl;
		fp << "VIFP   : " << vifpMes << endl;

		/// create windows
		if (previewOn)
		{
			cv::namedWindow("origin", 1);
			cv::imshow("origin", originalImage);
			cv::namedWindow("noise", 1);
			cv::imshow("noise", noiseImage);
			cv::waitKey();
		}
		count++;
	}
	// compare 2 corresponding images
	cout << "###############################" << endl;
	cout << "Similarity average of images in 2 folders: " << folderPath1 <<" and " << folderPath2 << endl;
	cout << "PSNR   : " << psnrSum / count << endl;
	cout << "SSIM   : " << ssimSum / count << endl;
	cout << "MSSSIM : " << msssimSum / count << endl;
	cout << "VIFP   : " << vifpSum / count << endl;
	cout << "###############################" << endl;
	cout << "Done!\n";

	// write to file
	fp << "###############################" << endl;
	fp << "Similarity average of images in 2 folders: " << folderPath1 << " and " << folderPath2 << endl;
	fp << "PSNR   : " << psnrSum / count << endl;
	fp << "SSIM   : " << ssimSum / count << endl;
	fp << "MSSSIM : " << msssimSum / count << endl;
	fp << "VIFP   : " << vifpSum / count << endl;
	fp << "###############################" << endl;
	fp << "Done!\n";
	fp.close();
}

