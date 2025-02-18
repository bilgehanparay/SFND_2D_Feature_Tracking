/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

/**/
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix   = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex  = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex    = 9;   // last file index to load
    int imgFillWidth   = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size() < dataBufferSize){
            dataBuffer.push_back(frame);
        }else{
            dataBuffer.erase(dataBuffer.begin());
            dataBuffer.push_back(frame);
        }
        
        //// EOF STUDENT ASSIGNMENT
        //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */
        double t = (double)cv::getTickCount();
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "AKAZE"; // -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
        string descriptorTypeM = "DES_HOG"; // DES_BINARY, DES_HOG
        string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and 
        //// enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        if(detectorType.compare("SHITOMASI") == 0){
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }else if(detectorType.compare("HARRIS") == 0){
            detKeypointsHarris(keypoints, imgGray, true);
        }else{
            detKeypointsModern(keypoints, imgGray,  detectorType, false);
        }

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        vector<cv::KeyPoint>boundedKeyPoints;
        if (bFocusOnVehicle){
            for(auto it = keypoints.begin(); it <= keypoints.end();it++){
                if(vehicleRect.contains((*it).pt))
                    boundedKeyPoints.push_back(*it);
            } 
            keypoints.erase(keypoints.begin(), keypoints.end());
            keypoints = boundedKeyPoints;   
        }
        float meanKeySize = 0;
        float varKeysize = 0; 
        for(auto it = keypoints.begin(); it <= keypoints.end();it++)
            meanKeySize = meanKeySize + (*it).size;
        meanKeySize = meanKeySize/keypoints.size();
        for(auto it = keypoints.begin(); it <= keypoints.end();it++)
            varKeysize = varKeysize + ((*it).size - meanKeySize) * ((*it).size - meanKeySize);
        varKeysize = varKeysize / keypoints.size();
        //cout << "TASK7: " << detectorType << " Number of Keypoints: " << keypoints.size() 
        //     << " size mean: " << meanKeySize << " size var: " << varKeysize << "\n";
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        //cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;
        t = (double)cv::getTickCount() - t;
        double f = cv::getTickFrequency();
        //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
        t = t/f;
        cout << "TASK9: Detector-" << detectorType << " Descriptor-" << descriptorType 
             << " Total time for detection+description: " << 1000 * t/1.0 << "ms" <<"\n";

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            /*
            DES_BINARY is for discrete descriptors, such as BRISK, BRIEF, FREAK. 
            DES_HOG is for continuous descriptors such as SIFT and SURF.
            */

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp

            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorTypeM, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            // cout << "TASK8: Detector-" << detectorType << " Descriptor-" << descriptorType 
            //     << " Number of Matched Points: " << matches.size() << "\n";
            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}