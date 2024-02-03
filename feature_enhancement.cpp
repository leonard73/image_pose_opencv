#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Read two images
    Mat img1 = imread("./1.ppm", IMREAD_GRAYSCALE);
    Mat img2 = imread("./2.ppm", IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images." << endl;
        return -1;
    }

    // Detect ORB features and descriptors
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    // Match features using BFMatcher
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches based on their distances
    sort(matches.begin(), matches.end());

    // Take the top N matches (you can adjust this based on your requirements)
    int numMatches = 50;
    vector<DMatch> topMatches(matches.begin(), matches.begin() + numMatches);

    // Draw matches
    Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, topMatches, imgMatches);



    // Extract matched keypoints
    vector<Point2f> points1, points2;
    for (const DMatch& match : topMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Find homography matrix
    Mat H = findHomography(points1, points2, RANSAC);

    // // Compute rotation and translation from homography matrix
    double fx = 300; // Focal length (adjust based on your camera)
    double fy = 300;
    double cx = img1.cols / 2; // Principal point (center of the image)
    double cy = img1.rows / 2;

    Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    Mat rot, trans;
    decomposeHomographyMat(H, K, rot, trans, noArray());

    // // Print rotation and translation
    // cout << "Rotation Matrix:\n" << rot << endl;
    // cout << "Translation Vector:\n" << trans << endl;

    // Display the matches
    imshow("Matches", imgMatches);
    waitKey(0);

    return 0;
}