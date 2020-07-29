#include<iostream>
#include<opencv2\opencv.hpp>
#include<Windows.h>

using namespace std;
using namespace cv;


struct Ellipse_struct
{
	double A, B, C, D, E, F;
};

//随机选一个点
Point2d GetRandomPoint(const vector<Point2d>pts)
{
	const int idx = rand() % pts.size();
	return pts.at(idx);
}

//随机选n个点
vector<Point2d> GetNrandomPoints(const vector<Point2d>pts, const int nPoints)
{
	vector<Point2d> randomPoints;
	for (int i = 0; i < nPoints; i++)
	{
		randomPoints.push_back(GetRandomPoint(pts));
	}
	return randomPoints;
}

//返回由n个点拟合的椭圆的参数
Ellipse_struct RansacFitEllipse(vector<Point2d>select_points)
{
	Ellipse_struct ellipses_null = { 0 };
	if (select_points.size() < 5)
	{
		cout << "No enough points to fit ellipse." << endl;
		return ellipses_null;
	}

	/*============================================================
		  椭圆的一般方程：Ax2 + Bxy + Cy2 + Dx + Ey + F = 0     
	==============================================================*/

	Mat leftm = Mat_<double>(select_points.size(), 5);
	Mat rightm = Mat_<double>(select_points.size(), 1);
	for (int i = 0; i < select_points.size(); i++)
	{
		leftm.at<double>(i, 0) = select_points[i].x*select_points[i].y;
		leftm.at<double>(i, 1) = select_points[i].y*select_points[i].y;
		leftm.at<double>(i, 2) = select_points[i].x;
		leftm.at<double>(i, 3) = select_points[i].y;
		leftm.at<double>(i, 4) = 1;

		rightm.at<double>(i, 0) = -(select_points[i].x*select_points[i].x);
	}
	Mat cof_mat = (leftm.t()*leftm).inv()*leftm.t()*rightm;//求解AX=B
	double A, B, C, D, E, F;
	A = 1;
	B = cof_mat.at<double>(0);
	C = cof_mat.at<double>(1);
	D = cof_mat.at<double>(2);
	E = cof_mat.at<double>(3);
	F = cof_mat.at<double>(4);
	Ellipse_struct ellipse_model = { A, B, C, D, E, F };
	return ellipse_model;
}

//迭代n次 RANSAC拟合椭圆
//dataPts：所有边缘点集；error：代数距离误差；num：记录拟合椭圆是用到了点的个数
//use_points：拟合过程中用到的点；noise_points：拟合过程中没用到的点
vector<Ellipse_struct> IterFit(vector<Point2d>dataPts, const int iter_num,const double error,vector<int>&num, 
	    vector<vector<Point2d>>&use_points, vector<vector<Point2d>>&noise_points)
{
	Ellipse_struct ellipse_fit; //每次迭代过程中不断改变的椭圆模型
	vector<Ellipse_struct>ellipse_model_pool;//存储每次迭代的椭圆拟合参数
	use_points.resize(iter_num);
	noise_points.resize(iter_num);
	for (int i = 0; i < iter_num; i++)//迭代
	{
		
		vector<Point2d>select_points = GetNrandomPoints(dataPts, 6);//每次迭代先选6个初始点进行拟合，选5个点有时候不能刚好拟合成椭圆
		ellipse_fit = RansacFitEllipse(select_points);//椭圆拟合

		//椭圆拟合过程中只有0解，则该6个点不能拟合成椭圆，重新选6个点
		if (ellipse_fit.B == 0 && ellipse_fit.C == 0 && ellipse_fit.D == 0 && ellipse_fit.E == 0 && ellipse_fit.F == 0)
		{
			select_points.clear();
			--i;
			continue;
		}

		int points_num = 0;//记录用于拟合椭圆的点的个数
		for (int j = 0; j < dataPts.size(); j++)//边缘点集中的点遍历
		{
			double x = dataPts[j].x;
			double x2 = dataPts[j].x*dataPts[j].x;
			double y = dataPts[j].y;
			double y2 = dataPts[j].y*dataPts[j].y;
			double xy = dataPts[j].x*dataPts[j].y;
			//代数距离
			double distance = ellipse_fit.A*x2 + ellipse_fit.B*xy + ellipse_fit.C*y2 + ellipse_fit.D*x + ellipse_fit.E*y + ellipse_fit.F;
			//代数距离小于误差，参与拟合
			if (abs(distance)<=error)
			{
				select_points.push_back(dataPts[j]);
				ellipse_fit = RansacFitEllipse(select_points);
				points_num++;
				continue;
			}
			//代数距离大于误差，不参与拟合，记录不参与拟合的点
			else
			{
				noise_points[i].push_back(dataPts[j]);
			}
		}

		use_points[i] = select_points;//记录参与椭圆拟合的点
		select_points.clear();//将所选的点清空，准备进行下一次迭代
		num.push_back(points_num);//记录该椭圆拟合时的点的数量
		ellipse_model_pool.push_back(ellipse_fit);//记录该椭圆模型
	}

	return ellipse_model_pool;//返回所有椭圆模型
}

//获取拟合后的椭圆参数
void GetEllipseParam(Ellipse_struct best_ellipse, double&x, double&y, double&alpha, double&majorAxis, double&minorAxis)
{
	double A, B, C, D, E, F;
	//F归一
	A = best_ellipse.A/ best_ellipse.F;
	B = best_ellipse.B/ best_ellipse.F;
	C = best_ellipse.C/ best_ellipse.F;
	D = best_ellipse.D/ best_ellipse.F;
	E = best_ellipse.E/ best_ellipse.F;
	F = best_ellipse.F/ best_ellipse.F;

	// 中心坐标
	x = ((B * E) - (2 * C * D)) / ((4 * A * C) - (B * B));
	y = ((B * D) - (2 * A * E)) / ((4 * A * C) - (B * B));

	// 长短轴
	majorAxis = max(sqrt(2 * (A*x*x + C*y*y + B*x*y - 1) / (A + C - sqrt((A - C)*(A - C) + B*B))), sqrt(2 * (A*x*x + C*y*y + B*x*y - 1) / (A + C + sqrt((A - C)*(A - C) + B*B))));
	minorAxis = min(sqrt(2 * (A*x*x + C*y*y + B*x*y - 1) / (A + C - sqrt((A - C)*(A - C) + B*B))), sqrt(2 * (A*x*x + C*y*y + B*x*y - 1) / (A + C + sqrt((A - C)*(A - C) + B*B))));
	// 角度
	alpha =  (-atan( B/ (A - C)))/2 ;
	
}

//计算误差
vector<double> CalculateError(Ellipse_struct ellipse_model, string best_model_path)
{
	vector<double>error_vec;
	fstream best_model(best_model_path);
	if (!best_model.is_open())
	{
		cout << "未能打开文件！" << endl;
	}
	double x, y, R1, R2, Phi;
	GetEllipseParam(ellipse_model, x, y, Phi, R1, R2);
	double n;
	while (best_model >> n)
	{
		error_vec.push_back(n);
	}
	error_vec[0] = y - error_vec[0];
	error_vec[1] = x - error_vec[1];
	error_vec[2] = Phi - error_vec[2];
	error_vec[3] = R1 - error_vec[3];
	error_vec[4] = R2 - error_vec[4];
	return error_vec;
}

//读取txt文件中的边缘点
Point2d SubTemp(string&temp_col,string&temp_row)
{
	Point2d bound;
	int n;
	istringstream is_c(temp_col);
	istringstream is_r(temp_row);
	string scols, srows;
	is_c >> n >> scols;
	is_r >> n >> srows;
	double fcols = stof(scols);
	double frows = stof(srows);
	bound.x = fcols;
	bound.y = frows;
	return bound;
}

int main()
{
	ifstream myfile_c("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5_col.txt");
	ifstream myfile_r("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5_row.txt");

	vector<Point2d>bound;//全部边缘点集

	//读取Halcon提取到的亚像素边缘
	string temp_c;
	string temp_r;
	if (!myfile_c.is_open()|| !myfile_r.is_open())
	{
		cout << "未成功打开文件" << endl;
	}
	while (getline(myfile_c, temp_c)&&getline(myfile_r,temp_r))
	{
		Point2d bound_points = SubTemp(temp_c,temp_r);
		bound.push_back(bound_points);
	}
	myfile_c.close();
	myfile_r.close();

	vector<int>num_all;//用于记录用于拟合若干个椭圆模型的个数
	vector<vector<Point2d>>use_points_pool;//用于拟合椭圆的点
	vector<vector<Point2d>>noise_points_pool;//没用来拟合椭圆的点

	double t1 = getTickCount();
	vector<Ellipse_struct>ellipse_all = IterFit(bound, 50, 15, num_all, use_points_pool, noise_points_pool);//迭代次数50次，代数距离小于10的点不参与拟合
	double t2 = getTickCount();
	cout << (t2-t1)/getTickFrequency() << " ms" << endl;
	
	vector<int>::iterator max_num = max_element(num_all.begin(), num_all.end());//用于拟合椭圆最多的点的个数
	Ellipse_struct best_ellipse_model = ellipse_all[distance(num_all.begin(), max_num)];//基于最大点的个数，选择最优的椭圆模型

	vector<Point2d>use_points = use_points_pool[distance(num_all.begin(), max_num)];//最优椭圆中用到的点
	vector<Point2d>noise_points = noise_points_pool[distance(num_all.begin(), max_num)];//最优椭圆中没用到的点

	for (int j = 0; j < noise_points.size(); j++)
	{
		double x = noise_points[j].x;
		double x2 = noise_points[j].x*noise_points[j].x;
		double y = noise_points[j].y;
		double y2 = noise_points[j].y*noise_points[j].y;
		double xy = noise_points[j].x*noise_points[j].y;
		//代数距离
		double distance = best_ellipse_model.A*x2 + best_ellipse_model.B*xy + best_ellipse_model.C*y2 + best_ellipse_model.D*x + best_ellipse_model.E*y + best_ellipse_model.F;

		cout << distance << endl;
	}

	//保存有用、无用点
	//ofstream outfile_col;
	//ofstream outfile_row;
	//outfile_col.open("bad_points(5)15_col.txt");
	//outfile_row.open("bad_points(5)15_row.txt");
	//outfile_col << noise_points.size() << endl;
	//outfile_row << noise_points.size() << endl;
	//for (int i = 0; i < noise_points.size(); i++)
	//{
	//	outfile_col << setprecision(10) << "2 " << noise_points[i].x << endl;
	//	outfile_row << setprecision(10) << "2 " << noise_points[i].y << endl;
	//}
	//outfile_col.close();
	//outfile_row.close();

	//椭圆参数 
	double x, y, R1, R2, Phi;
	GetEllipseParam(best_ellipse_model, x, y, Phi, R1, R2);
	Point2f center(x, y);


	//FileStorage fs("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5计算参数结果改进.txt", FileStorage::WRITE);

	//fs << "center" << center;
	//fs << "Phi" << Phi;
	//fs << "R1" << R1;
	//fs << "R2" << R2;
	//fs.release(); 

	vector<double>error_vec = CalculateError(best_ellipse_model, "C:/Users/SK-Dan/Desktop/自建椭圆数据集/real_src_ellipse_param.txt");

	//FileStorage fs2("C:/Users/SK-Dan/Desktop/自建椭圆数据集/5-9/5计算误差结果改进.txt", FileStorage::WRITE);
	//fs2 << "X" << error_vec[0];
	//fs2 << "Y" << error_vec[1];
	//fs2 << "phi" << error_vec[2];
	//fs2 << "R1" << error_vec[3];
	//fs2 << "R2" << error_vec[4];
	//fs2.release();
	return 0;
}