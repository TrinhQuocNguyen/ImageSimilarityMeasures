Compare the similarity from image to image in 2 separated folders (Images have corresponding images as the same file name).
The program will print out and save to file as the log file:
1. Compare each image to corresponding image
2. Evaluate as the mean value to all the images in 2 folders.



-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------

Video Quality Measurement Tool

---------------------------
 INTRODUCTION
---------------------------

This software provides fast implementations of the following objective metrics:
- PSNR: Peak Signal-to-Noise Ratio,
- SSIM: Structural Similarity,
- MS-SSIM: Multi-Scale Structural Similarity,
- VIFp: Visual Information Fidelity, pixel domain version,
- PSNR-HVS: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF),
- PSNR-HVS-M: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF) and between-coefficient contrast masking of DCT basis functions.

In this software, the above metrics are implemented in OpenCV (C++) based on the original Matlab implementations provided by their developers.
The source code of this software can be compiled on any platform and only requires the OpenCV library (core and imgproc modules).
This software allows performing video quality assessment without using Matlab and shows better performance than Matlab in terms of run time.


---------------------------
 PREREQUISITE
---------------------------

The OpenCV library needs to be installed to be able to compile this code. Only the core and imgproc modules are required.
This software was developed using OpenCV 3.2.0.

The OpenCV dlls are provided directly with the Windows binaries, such that the software can work directly without installing the OpenCV library.


---------------------------
 USAGE
---------------------------
Metrics: the list of metrics to use
 available metrics:
 - PSNR: Peak Signal-to-Noise Ratio (PNSR)
 - SSIM: Structural Similarity (SSIM)
 - MSSSIM: Multi-Scale Structural Similarity (MS-SSIM)
 - VIFP: Visual Information Fidelity, pixel domain version (VIFp)
 - PSNRHVS: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF) (PSNR-HVS)
 - PSNRHVSM: Peak Signal-to-Noise Ratio taking into account Contrast Sensitivity Function (CSF) and between-coefficient contrast masking of DCT basis functions (PSNR-HVS-M)


Example:




Notes:
- SSIM comes for free when MSSSIM is computed (but you still need to specify it to get the output)
- PSNRHVS and PSNRHVSM are always computed at the same time (but you still need to specify both to get the two outputs)
- When using MSSSIM, the height and width of the video have to be multiple of 16
- When using VIFP, the height and width of the video have to be multiple of 8


