#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Non maxima suppression implementation
// Input is harris corner image and was thresholded
void nms(cv::Mat corner_img, std::vector<cv::KeyPoint> &keypoints, int swSize, int blockSize, unsigned char minResponse){
    int sw_dist = floor(swSize / 2); 
    // loop over all pixels in the corner image
    for (int r = sw_dist; r < corner_img.rows - sw_dist - 1; r++){ // rows
        for (int c = sw_dist; c < corner_img.cols - sw_dist - 1; c++){ // cols
            // loop over all pixels within sliding window around the current pixel
            unsigned char max_val{0}; // keeps track of strongest response
            int r_max = -1;
            int c_max = -1;
            for (int rs = r - sw_dist; rs <= r + sw_dist; rs++){
                for (int cs = c - sw_dist; cs <= c + sw_dist; cs++){
                    // check wether max_val needs to be updated
                    unsigned char new_val = (unsigned char)corner_img.at<unsigned char>(rs, cs);
                    // max_val = max_val < new_val ? new_val : max_val;
                    if(max_val < new_val && new_val > minResponse){
                        max_val = new_val;
                        r_max = rs;
                        c_max = cs;
                    }
                }
            }

            // check wether current pixel is local maximum
            if (r == r_max && c == c_max){
                // add to the keypoint vector
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(c, r);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }  
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    // compute detector parameters based on image size
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    unsigned char minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    
    double k = 0.04; // Harris parameter (see equation for details)
    int swSize = 7; // size of NMS window

    // Apply corner detection
    double t = (double)cv::getTickCount();

     // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1 );
    cv::cornerHarris( img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT ); 
    double ret, thresh1, max, min;
    cv::minMaxLoc(dst, &min, &max);
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_8U, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    /*ret,thresh1 = cv::threshold(dst_norm_scaled, 
                                dst_norm_scaled, 
                                minResponse, 
                                255, 
                                cv::THRESH_BINARY);*/
    nms(dst_norm_scaled, keypoints, swSize, blockSize, minResponse);
    //imshow("harris scaled", dst_norm_scaled);
    // cv::Mat visImageScaled = dst_norm_scaled.clone();
    // cv::drawKeypoints(dst_norm_scaled, keypoints, visImageScaled, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // imshow("harris keypoints", visImageScaled);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    if(detectorType.compare("FAST") == 0){
        int threshold = 50;
        bool bNMS = true; // use NMS in FAST
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = (double)cv::getTickCount() - t;
        
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, 
                          keypoints, 
                          visImage, 
                          cv::Scalar::all(-1), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "FAST Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }else if(detectorType.compare("BRISK") == 0){
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        detector->detect(img, keypoints);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, 
                          keypoints, 
                          visImage, 
                          cv::Scalar::all(-1), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "BRISK Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);  
    }else if(detectorType.compare("ORB") == 0){
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detector->detect(img, keypoints);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, 
                            keypoints, 
                            visImage, 
                            cv::Scalar::all(-1), 
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "ORB Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);     
    }else if(detectorType.compare("AKAZE") == 0){
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        detector->detect(img, keypoints);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, 
                            keypoints, 
                            visImage, 
                            cv::Scalar::all(-1), 
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "AKAZE Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);     
    }else if(detectorType.compare("SIFT") == 0){
        cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
        siftPtr->detect(img, keypoints);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, 
                          keypoints, 
                          visImage, 
                          cv::Scalar::all(-1), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "SIFT Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);  
    }
}