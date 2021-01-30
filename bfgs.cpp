//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <assert.h>
/*
Using BFGS to optimize arbitrary quadratic function
Author: Glinttsd  email:913995397@qq.com
Reference: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
*/
using namespace std;
typedef std::vector<int> vi;
template <typename T>
struct coord_2D;
template <typename T>
struct matrix_2x2;
template <typename T>
struct coord_2D_trans;

double n = 0.01;
double threshold = 0.01;

template <typename T>
struct quadra_func {
	quadra_func<T>(T _a11, T _a1, T _b, T _a22 = 0, T _a12 = 0, T _a2 = 0) : a11(_a11), a22(_a22), a12(_a12), a1(_a1), a2(_a2), b(_b) {}
	T operator()(T _x, T _y = 0) {
		x_val = _x;
		y_val = _y;
		return a11 * x_val * x_val + a22 * y_val * y_val + a12 * x_val * y_val + a1 * x_val + a2 * y_val + b;
	}
public:
	T a11,a12,a22,a1,a2,b;
	T x_val,y_val;
};

template <typename T>
struct coord_2D {
	coord_2D(T _x1, T _x2, vi _shape = vi{ 2, 1 }) :x1(_x1), x2(_x2), shape(_shape) {}
	coord_2D() :shape(vi{ 2,1 }) {}
	coord_2D& operator=(coord_2D& coord) {
		//assert(!shape_check(coord));
		x1 = coord.x1;
		x2 = coord.x2;
		return *this;
	}
	coord_2D<T> operator*(T a) {
		return coord_2D<T>(x1 * a, x2 * a);
	}
	coord_2D<T> operator*(matrix_2x2<T>& mat) {
		//assert((shape != vi{ 1,2 }));
		return coord_2D<T>(mat.a11 * x1 + mat.a21 * x2, mat.a12 * x1 + mat.a22 * x2);
	}
	matrix_2x2<T> operator*(coord_2D_trans<T>& coord) { // (2,1)*(1,2)
		return matrix_2x2<T>(x1 * coord.x1, x1 * coord.x2, x2 * coord.x1, x2 * coord.x2);
	}
	coord_2D<T> operator+(coord_2D& coord) {
		return coord_2D<T>(x1 + coord.x1, x2 + coord.x2);
	}
	coord_2D<T> operator-(coord_2D& coord) {
		return coord_2D<T>(x1 - coord.x1, x2 - coord.x2);
	}

public:
	T x1;
	T x2;
	std::vector<int> shape;

	coord_2D_trans<T> Tran() {
		return coord_2D_trans<T>(x1, x2);
	}
	bool shape_check(coord_2D& coord) {
		return coord.shape == shape ? true : false;
	}
};

template <typename T>
struct coord_2D_trans :public coord_2D<T> {
	coord_2D_trans() :shape(vi{ 2,1 }) {}
	coord_2D_trans(T _x1, T _x2, vi _shape = vi{ 1,2 }) {
		x1 = _x1;
		x2 = _x2;
		shape = _shape;
	}
	T operator*(coord_2D<T>& coord) { // (1,2)*(2,1)
		return x1 * coord.x1 + x2 * coord.x2;
	}
	coord_2D_trans<T> operator*(matrix_2x2<T>& mat) { // (1,2)*(2,2)
		return coord_2D_trans<T>(x1 * mat.a11 + x2 * mat.a21,
			x1 * mat.a12 + x2 * mat.a22);
	}
};

template <typename T>
struct matrix_2x2 {
	matrix_2x2() {}
	matrix_2x2(T _a11, T _a12, T _a21, T _a22) :a11(_a11), a12(_a12), a21(_a21), a22(_a22) {}
	matrix_2x2& operator=(matrix_2x2& mat) {
		a11 = mat.a11;
		a12 = mat.a12;
		a21 = mat.a21;
		a22 = mat.a22;
		return *this;
	}
	matrix_2x2<T> operator+(matrix_2x2& mat) {
		return matrix_2x2<T>(a11 + mat.a11, a12 + mat.a12, a21 + mat.a21, a22 + mat.a22);
	}
	matrix_2x2<T> operator*(T a) {
		return matrix_2x2<T>(a11 * a, a12 * a, a21 * a, a22 * a);
	}
	coord_2D<T> operator*(coord_2D<T>& coord) {// (2,2) * (2,1)
		return coord_2D<T>(a11 * coord.x1 + a21 * coord.x2, a12 * coord.x1 + a22 * coord.x2);
	}
	matrix_2x2<T> operator*(matrix_2x2<T>& mat) {
		T A = a11 * mat.a11 + a12 * mat.a21;
		T B = a11 * mat.a12 + a12 * mat.a22;
		T C = a21 * mat.a11 + a22 * mat.a21;
		T D = a21 * mat.a12 + a22 * mat.a22;
		return matrix_2x2<T>(A, B, C, D);
	}

public:
	T a11,a12,a21,a22;
};


template<typename quadra_func, typename T>
T dx(quadra_func func, T x, T y) {
	return (func((x + static_cast<T> (n)), y) - func((x - static_cast<T> (n)), y)) / (2 * static_cast<T> (n));
}

template<typename quadra_func, typename T>
T dy(quadra_func func, T x, T y) {
	return (func(x, (y + static_cast<T> (n))) - func(x, (y - static_cast<T> (n)))) / (2 * static_cast<T> (n));
}

template<typename quadra_func, typename T>
T dxx(quadra_func func, T x, T y) {
	return (dx(func, (x + static_cast<T> (n)), y) - dx(func, (x - static_cast<T> (n)), y)) / (2 * static_cast<T> (n));
}

template<typename quadra_func, typename T>
T dyy(quadra_func func, T x, T y) {
	return (dy(func, x, (y + static_cast<T> (n))) - dy(func, x, (y - static_cast<T> (n)))) / (2 * static_cast<T> (n));
}

//template<typename quadra_func, typename T>
//T dxy(quadra_func func, T x, T y) {
//	return (dx(func, x, (y + static_cast<T> (n))) - dx(func,x, (y - static_cast<T> (n)))) / (2 * static_cast<T> (n));
//}
//
//template<typename quadra_func, typename T>
//T dyx(quadra_func func, T x, T y) {
//	return (dy(func, (x + static_cast<T> (n)), y) - dy(func, (x - static_cast<T> (n)), y)) / (2 * static_cast<T> (n));
//}

template<typename T>
T line_search(quadra_func<T> func, coord_2D<T> x, coord_2D<T> g) {
	T a = x.x1;
	T b = g.x1;
	T c = x.x2;
	T d = g.x2;
	T alpha_a11 = func.a11 * b * b + func.a22 * d * d + func.a12 * b * d;
	T alpha_a1 = 2 * func.a11 * a * b + 2 * func.a22 * c * d + func.a12 * (a * d + c * b) + func.a1 * b + func.a2 * d;
	T alpha_b = func.a11 * a * a + func.a22 * c * c + func.a12 * a * c + func.a1 * a + func.a2 * c + func.b;
	quadra_func<T> func_of_alpha(alpha_a11, alpha_a1, alpha_b);
	return -alpha_a1 / (2 * alpha_a11);
}

template<typename quadra_func, typename T>
coord_2D<T> bfgs(quadra_func func, coord_2D<T> x0) {
	int max_steps = 1000;
	int step;
	//T x_k;
	//T y_k;
	T alpha_k;
	coord_2D<T> x_k;
	coord_2D<T> x_k_new;
	coord_2D<T> y_k;
	//coord_2D<T> p_0(0, 0);
	coord_2D<T> p_k;
	coord_2D<T> s_k;
	coord_2D<T> g_0(dx(func, x0.x1, x0.x2), dy(func, x0.x1, x0.x2));
	coord_2D<T> g_k;
	coord_2D<T> g_k_new;
	matrix_2x2<T> I(1, 0, 0, 1);
	matrix_2x2<T> B_k_inverse;
	matrix_2x2<T> B_k_inverse_new;
	for (step = 0; step < max_steps; step++) {
		if (step == 0) {
			B_k_inverse = I;
			x_k = x0;
			g_k = g_0;
		}
		else {
			B_k_inverse = B_k_inverse_new;
			g_k = g_k_new;
			x_k = x_k_new;
			if ((g_k.x1 < threshold) && (g_k.x2 < threshold)) {
				cout << "number of steps:" << step << endl;
				return x_k;
			}
		}
		p_k = B_k_inverse * (g_k * (-1));
		alpha_k = line_search(func, x_k, p_k);
		s_k = (p_k * alpha_k);
		x_k_new = (x_k + s_k);
		g_k = coord_2D<T>(dx(func, x_k.x1, x_k.x2), dy(func, x_k.x1, x_k.x2));
		g_k_new = coord_2D<T>(dx(func, x_k_new.x1, x_k_new.x2), dy(func, x_k_new.x1, x_k_new.x2));
		y_k = g_k_new - g_k;
		double a = y_k.Tran() * B_k_inverse * y_k;
		double b = s_k.Tran() * y_k;
		matrix_2x2<T> c = s_k * s_k.Tran();
		matrix_2x2<T> d = B_k_inverse * y_k * s_k.Tran();
		matrix_2x2<T> e = s_k * y_k.Tran() * B_k_inverse;
		matrix_2x2<T> f = s_k * s_k.Tran();
		B_k_inverse_new = (B_k_inverse + (f * (b + a)) * (1 / (b * b)) + ((d + e) * (1 / b)) * (-1));
	}
	cout << "number of steps:" << step << endl;
	return x_k;
}


//main()
//{
//	coord_2D<double> res_point;
//	double x;
//	double y;
//	double a11, a22, a12, a1, a2, b;
//	cout << "Enter your quadratic function: <template> a11*x*x + a22*y*y + a12*x*y + a1*x + a2*y +b" << endl;
//	cout << "a11:";
//	cin >> a11;
//	cout << "a22:";
//	cin >> a22;
//	cout << "a12:";
//	cin >> a12;
//	cout << "a1:";
//	cin >> a1;
//	cout << "a2:";
//	cin >> a2;
//	cout << "b:";
//	cin >> b;
//	quadra_func<double> my_func(a11, a1, b, a22, a12, a2);//f(x,y) = a11x^2 + a22y^2 + a12xy + a1x + a2y + b
//	//quadra_func<double> my_func(4, -16, 31, 1, 0, -6);
//	while (1) {
//		cout << "Chose your initial point: <template> (x,y):" << endl;
//		cout << "x:";
//		cin >> x;
//		cout << "y:";
//		cin >> y;
//		coord_2D<double> x0(x, y);
//		cout << "Processing BFGS...." << endl;
//		res_point = bfgs(my_func, x0);
//		cout << "Optimal point is:" << "(" << res_point.x1 << "," << res_point.x2 << ")" << endl;
//		double result = my_func(res_point.x1, res_point.x2);
//		cout << "Optimal value is:" << result << endl;
//	}
//
//	system("pause");
//}