#include <pcl/visualization/cloud_viewer.h>
#include <iostream>  
#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h>  
#include <opencv2/opencv.hpp>  
#include <string>
#include "PFMReadWrite.h"
#include "fadiff.h"
#include "badiff.h"
#include <cmath>
using namespace std;
using namespace cv;
using namespace pcl;
using namespace Eigen;
using namespace fadbad;

int user_data;
//相机内参，根据输入改动
const double u0 = 1118.559;
const double v0 = 965.441;
const double fx = 7125.31;
const double fy = 7125.31;
const double Tx = 169.748;
const double doffs = 461.123;
const double d1 = 20;
Matrix<double, 3, 3> CamMatrix;
bool L = true;
//inline Matrix<double, 4, 4> countT2(double u, double v)
//{
//	Matrix<double, 4, 4> ans;
//	if (u == 0 && v == 0)
//	{
//		ans << 1, 0, 0, 0,
//			0, 1, 0, 0,
//			0, 0, 0, 0,
//			0, 0, 0, 0;
//		return ans;
//	}
//	else
//	{
//		double U = u * u;
//		double V = v * v;
//		double tmp = ((U + V)) / (1.776889 * (1 + U + V) - (U + V));
//		double alfa = sqrt(tmp) / sqrt(U + V);
//		ans << 1, 0, 0, 0,
//			0, 1, 0, 0,
//			0, 0, alfa, 0,
//			0, 0, 0, alfa;
//	}
//}
F<double> func(const F<double>& x, const F<double>& y, double& Zw)
{
	double miu = 1.333*1.333;

	F<double> z = sqrt(miu + (miu - 1)*(x*x + y * y));

	return (d1 + (Zw - d1) / z)*(x - u0) / fx - x;
}
void viewerOneOff(visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
}

int main()
{
	CamMatrix << fx, 0, u0, 0, fy, v0, 0, 0, 1;
	
	PointCloud<PointXYZ> cloud_a;
	PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

	Mat color = imread("E:/2021/depth2pointcloud/Sword2-perfect/im0.png");
	Mat depth = loadPFM(string("E:/2021/depth2pointcloud/Sword2-perfect/disp0.pfm"));

	int rowNumber = color.rows;
	int colNumber = color.cols;

	cloud_a.height = rowNumber;
	cloud_a.width = colNumber;
	cloud_a.points.resize(cloud_a.width * cloud_a.height);
	Mat mapL_x = Mat(rowNumber, colNumber, CV_32FC1); //保存方向图像的映射关系信息
	Mat mapL_y = Mat(rowNumber, colNumber, CV_32FC1);

	float *x = (float*)mapL_x.data;
	float *y = (float*)mapL_y.data;
	//unsigned short *y = (unsigned short *)malloc(sizeof(unsigned short)*rowNumber*colNumber);
	//unsigned short *rx = (unsigned short *)malloc(sizeof(unsigned short)*rowNumber*colNumber);
	//unsigned short *ry = (unsigned short *)malloc(sizeof(unsigned short)*rowNumber*colNumber);
	for (unsigned int u = 0; u < rowNumber; ++u)
	{
		for (unsigned int v = 0; v < colNumber; ++v)
		{
			/*unsigned int num = rowNumber*colNumber-(u*colNumber + v)-1;*/
			unsigned int num = u * colNumber + v;
			double Xw = 0, Yw = 0, Zw = 0;
			double eps = 0.01;

			Zw = fx * Tx / ((depth.ptr<float>(u, v)[0]) + doffs);
			Xw = (v + 1 - u0) * Zw / fx;
			Yw = (u + 1 - v0) * Zw / fy;
			F<double> xx, yy, f;     // Declare variables x,y,f
			f = func(xx, yy,Zw);         // Evaluate function and derivatives
			for (int i = 0; i < 10; i++)
			{
				xx = u;                 // Initialize variable x
				xx.diff(0, 2);         // Differentiate with respect to x (index 0 of 2)
				yy = v;                 // Initialize variable y
				yy.diff(1, 2);         // Differentiate with respect to y (index 1 of 2)
				double fval = f.x();   // Value of function
				double dfdxx = f.d(0);  // Value of df/dx (index 0 of 2)
				double dfdyy = f.d(1);  // Value of df/dy (index 1 of 2)
				xx = xx - fval / dfdxx;
				yy = yy - fval / dfdyy;
				if (fabs(Xw - xx.val()) < eps && fabs(Yw - yy.val()) < eps)
				{
					//xx.val
					x[num] = xx.val(); y[num] = yy.val(); break;
				}
				cout << "f(x,y)=" << fval << endl;
				cout << "df/dx(x,y)=" << dfdxx << endl;
				cout << "df/dy(x,y)=" << dfdyy << endl;
			}


			//cloud_a.points[num].b = color.at<Vec3b>(u, v)[0];
			//cloud_a.points[num].g = color.at<Vec3b>(u, v)[1];
			//cloud_a.points[num].r = color.at<Vec3b>(u, v)[2];
			cloud_a.points[num].x = Xw;
			cloud_a.points[num].y = Yw;
			cloud_a.points[num].z = Zw;
		}
	}
	//for (unsigned int u = 0; u < rowNumber; ++u)
	//{
	//	for (unsigned int v = 0; v < colNumber; ++v)
	//	{
	//		unsigned int num = u * colNumber + v;
	//		double Xw = 0, Yw = 0, Zw = 0;

	//		//cloud_a.points[num].x = Xw;
	//		//cloud_a.points[num].y = Yw;
	//		//cloud_a.points[num].z = Zw;
	//	}
	//}

	//for (unsigned int u = 0; u < rowNumber; ++u)
	//{
	//	for (unsigned int v = 0; v < colNumber; ++v)
	//	{
	//		unsigned int num = u * colNumber + v;
	//		Matrix<double, 3, 1> cor_pixel;
	//		Matrix<double, 3, 1> cor_cam;
	//		cor_pixel << u, v,1;
	//		cor_cam = CamMatrix.inverse()*cor_pixel;
	//		Matrix<double, 4, 1> L_cor1;
	//		Matrix<double, 4, 1> L_cor2;
	//		Matrix<double, 4, 1> L_cor3;
	//		Matrix<double, 4, 1> L_cor4;
	//		Matrix<double, 2, 1> err;
	//		Matrix<double, 4, 4> T1;
	//		Matrix<double, 4, 4> T2;
	//		Matrix<double, 4, 4> T3;
	//		L_cor1 << 0, 0, cor_cam(0) / cor_cam(2), cor_cam(1) / cor_cam(2);
	//		T1 << 1, 0, d1, 0, 0, 1, 0, d1, 0, 0, 1, 0, 0, 0, 0, 1;
	//		L_cor2 = T1 * L_cor1;
	//		T2 = countT2(L_cor1(2), L_cor1(3));
	//		L_cor3 = T2 * L_cor2;
	//		T3<< 1, 0,  cloud_a.points[num].z-d1, 0, 0, 1, 0, cloud_a.points[num].z - d1, 0, 0, 1, 0, 0, 0, 0, 1;
	//		L_cor4 = T2 * L_cor3;
	//		err << L_cor4(0) - cloud_a.points[num].x, L_cor4(1) - cloud_a.points[num].y;

	//		//cloud_a.points[num].x = Xw;
	//		//cloud_a.points[num].y = Yw;
	//		//cloud_a.points[num].z = Zw;
	//	}
	//}

	//*cloud = cloud_a;

	//visualization::CloudViewer viewer("Cloud Viewer");

	//viewer.showCloud(cloud);

	//viewer.runOnVisualizationThreadOnce(viewerOneOff);

	//while (!viewer.wasStopped())
	//{
	//	user_data = 9;
	//}

	return 0;
}