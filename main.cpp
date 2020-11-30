#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING


#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <vector>
#include <random>
#include <chrono>


using namespace cv;
using namespace std;

string dir = "D:/Dataset";
vector<string> paths;
vector<string> paths2;

Mat addGaussian(Mat image, int n); // n-> 횟수
Mat motionBlurs(Mat image);
Mat RandomResize(Mat image, int type); //type -> 크게 or 작게
Mat RandomCrop(Mat image, int type); //type -> 왼쪽, 오른쪽
Mat RandomRotate(Mat image);
Mat Distort(Mat image);
Mat addSaltPeper(Mat image, int n); // n-> 노이즈 개수

int main() {
	srand((int)time(NULL));

	int numberofImage[5]{ 0 };
	int ti = 0;
	for (auto& p : experimental::filesystem::directory_iterator(dir)) {
		for (auto& ps : experimental::filesystem::directory_iterator(p.path().string())) {
			string sc = ps.path().string();
			if (sc.find(".jpg") == string::npos) {
				paths.push_back(sc);
			}
			else { paths2.push_back(sc); numberofImage[ti]++; }
		}
		ti++;
	}
	cout << "Folder Name" << endl;
	for (string s : paths) {
		cout << s << endl;
	}
	cout << "Image Name" << endl;
	for (string s : paths2) {
		cout << s << endl;
	}

	chrono::system_clock::time_point start = chrono::system_clock::now();
	int number = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < numberofImage[i]; j++) {
			string readPath = paths2[number];
			Mat ori = imread(readPath);
			Mat tmp;
			string imagePath = paths[9 * i + 1]+"\\"+to_string(j)+".jpg"; // gaussian
			tmp = addGaussian(ori, 3);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 3] + "\\" + to_string(j) + ".jpg"; // motionB
			tmp = motionBlurs(ori);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 8] + "\\" + to_string(j) + ".jpg"; // scale-up
			tmp = RandomResize(ori, 0);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 7] + "\\" + to_string(j) + ".jpg"; // scale-down
			tmp = RandomResize(ori, 1);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 2] + "\\" + to_string(j) + ".jpg"; // left-crop
			tmp = RandomCrop(ori, 0);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 4] + "\\" + to_string(j) + ".jpg"; // right-crop
			tmp = RandomCrop(ori, 1);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 5] + "\\" + to_string(j) + ".jpg"; // rotation
			tmp = RandomRotate(ori);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 6] + "\\" + to_string(j) + ".jpg"; // salt&peper
			tmp = addSaltPeper(ori, 1000);
			imwrite(imagePath, tmp);
			imagePath = paths[9 * i + 0] + "\\" + to_string(j) + ".jpg"; // distort
			tmp = Distort(ori);
			imwrite(imagePath, tmp);

			cout << paths2[number] <<" image processing done" << endl;
			number++;
		}
	}
	chrono::system_clock::time_point end = chrono::system_clock::now();
	chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(end - start);
	double time = micro.count() / 1000.;

	cout << "All " << number << "Image Processing Done Time [" << time << "] ms" << endl;
}

Mat motionBlurs(Mat image) {
	Mat res;
	Mat mbKer = Mat::zeros(31, 31, CV_32FC1);
	
	for (int i = 0; i < 31; i++) {
		mbKer.ptr<float>(15)[i] = 1;
	}
	mbKer /= 31;

	filter2D(image, res, -1, mbKer);
	return res;
}

Mat addGaussian(Mat image, int n) {
	Mat res;
	
	GaussianBlur(image, res, Size(5,5), 1.6);
	for(int i = 0; i<n-1; i++) GaussianBlur(res, res, Size(5, 5), 1.6);

	return res;
}

Mat RandomResize(Mat image, int type) {
	Mat res;

	float resizeFactorX = (rand() % 1000) / 1000.; //0~1
	float resizeFactorY = (rand() % 1000) / 1000.; //0~1
	if (type == 0) {//bigger case
		resizeFactorX += 1; //1~2
		resizeFactorY += 1; //1~2
	}
	else {
		resizeFactorX = resizeFactorX /2 + 0.5; //0.5~1
		resizeFactorY = resizeFactorY /2 + 0.5; //0.5~1
	}
	resize(image, res, Size(image.cols * resizeFactorX, image.rows * resizeFactorY));

	return res;
}
Mat RandomCrop(Mat image, int type) {
	Mat res;

	Rect bounds(0, 0, image.cols, image.rows);
	Rect r;
	float cropFactor = (rand()%1000) / 2000 + 0.5; //0.5~1
	if (type == 0) r = Rect(0, 0, cropFactor * image.cols, image.rows);
	else if (type == 1) r = Rect((1 - cropFactor) * image.cols, 0, image.cols, image.rows);
	image(r&bounds).copyTo(res);
	return res;
}

Mat RandomRotate(Mat image) {
	Mat res;

	float rotateFactor = (rand() % 360);
	Mat rotationMatrix = getRotationMatrix2D(Point(image.cols / 2, image.rows / 2), rotateFactor, 1);

	warpAffine(image, res, rotationMatrix, image.size());

	return res;
}
Mat Distort(Mat image) {
	Mat res;
	image.copyTo(res);

	int cx = res.cols / 2, cy = res.rows / 2;

	Mat mapx(image.rows, image.cols, CV_32FC1), mapy(image.rows, image.cols, CV_32FC1);
	Mat r, theta;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			mapx.at<float>(i, j) = i;
		}
	}
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			mapy.at<float>(i, j) = j;
		}
	}

	mapx = 2 * mapx / (image.rows - 1.) - 1;
	mapy = 2 * mapy / (image.cols - 1.) - 1;


	cartToPolar(mapx, mapy, r, theta);

	for (int i = 0; i < r.total(); i++) {
		if(r.ptr<float>(0)[i] < 1)r.ptr<float>(0)[i] = powf(r.ptr<float>(0)[i], 1.4);
	}

	polarToCart(r, theta, mapx, mapy);

	mapx = ((mapx + 1.) * image.rows - 1.) / 2;
	mapy = ((mapy + 1.) * image.cols - 1.) / 2;

	remap(image, res, mapy, mapx, INTER_LINEAR);

	return res;
}
Mat addSaltPeper(Mat image, int n) {
	Mat res;
	image.copyTo(res);
	for (int k = 0; k < n; k++)
	{
		int i = rand() % image.cols; 
		int j = rand() % image.rows; 
		int salt_or_pepper = (rand() % 2) * 255; 

		if (image.type() == CV_8UC1) 
		{
			res.at<uchar>(j, i) = salt_or_pepper;
		}
		else if (image.type() == CV_8UC3)
		{
			res.at<cv::Vec3b>(j, i)[0] = salt_or_pepper;
			res.at<cv::Vec3b>(j, i)[1] = salt_or_pepper;
			res.at<cv::Vec3b>(j, i)[2] = salt_or_pepper;
		}
	}

	return res;
}