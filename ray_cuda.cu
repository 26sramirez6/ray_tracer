/*
 * ray_tracing.cpp
 *
 *  Created on: Aug 18, 2019
 *      Author: 26sra
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stack>
#include <set>
#include <cstdlib>
#include <omp.h>
#include <cuda.h>

using std::cout;
using std::endl;
using std::stoll;
using std::stack;
using std::set;
using std::vector;

using cd_t = const double;
using cld_t = const long double;
using llu_t = long long unsigned;
using cll_t = const long long;
using cllu_t = const unsigned long long;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr int nThreadsPerBlock = 256;
constexpr int nBlocks = 128;

__global__ void 
dot( int *a, int *b, int *c ) {
	__shared__ int temp[nThreadsPerBlock];
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	temp[threadIdx.x] = a[index] * b[index];
	__syncthreads();
	if( 0 == threadIdx.x ) {
		int sum = 0;
		for( int i = 0; i < nThreadsPerBlock; i++ )
			sum += temp[i];
		atomicAdd(c,sum);
	}
}

__global__ double
random_double( double min,  double max ) {
	return min + (double)rand()/(double)RAND_MAX * (max - min);
}

__global__ double
norm(int x, int y, int z) {
	return sqrt(x_*x_+y_*y_+z_*z_);
}

__global__ double
direction_sampling() {
	cd_t phi = random_double(0, 2*pi);
	cd_t cosTheta = random_double(-1, 1);
	cd_t sinTheta = sqrt(1-(cosTheta*cosTheta));
	x_ = sinTheta*cos(phi);
	y_ = sinTheta*sin(phi);
	z_ = cosTheta;
}

void
ray_serial(Config & cfg) {
	double * G = (double *)calloc(cfg.G2_, sizeof(double));
	Vec3d W(0,10,0), V, C(0,12,0), N, S, I, L(4,4,-1), IC, LI;
	constexpr int R = 6;
	constexpr int Wmax = 10;
	cd_t ref = cfg.GD2_/Wmax;
	double t0 = omp_get_wtime();
	for (llu_t i=0; i<cfg.N_; i++) {
		do {
			V.direction_sampling();
			cd_t q = W.y_/V.y_;
			W.x_ = q*V.x_;
			W.y_ = q*V.y_;
			W.z_ = q*V.z_;
		} while(!(abs(W.x_) < Wmax && abs(W.z_) < Wmax &&
				  pow(V.dot(C),2) + R*R - C.dot(C) > 0));

		cd_t vc = V.dot(C);
		cd_t t = vc - sqrt(vc*vc + R*R - C.dot(C));
		I.Set(t*V.x_, t*V.y_, t*V.z_);
		IC.Set(I.x_-C.x_, I.y_-C.y_, I.z_-C.z_);
		cd_t icNorm = IC.norm();
		N.Set(IC.x_/icNorm, IC.y_/icNorm, IC.z_/icNorm);
		LI.Set(L.x_-I.x_, L.y_-I.y_, L.z_-I.z_);
		cd_t liNorm = LI.norm();
		S.Set(LI.x_/liNorm, LI.y_/liNorm, LI.z_/liNorm);
		cd_t sn = S.dot(N);
		cd_t b = sn < 0 ? 0 : sn;

		cll_t x = (cll_t)floor(W.x_*ref + cfg.GD2_);
		cll_t y = (cll_t)floor(W.z_*ref + cfg.GD2_);
		G[x*cfg.G_ + y] += b;
	}
	double t1 = omp_get_wtime();
	cout << "Total Runtime (seconds): " << t1-t0 << endl;

	FILE * f = fopen("hw4.out", "wb");
	fwrite(G, sizeof(double), cfg.G2_, f);
	fclose(f);
}

int
main(int argc, char ** argv) {
	if (argc!=3) {
		cout << "Usage ./ray_tracing <number of rays> <gridpoints>" << endl;
	}
	srand((unsigned int)time(NULL));
	Config cfg(argv);
	ray_serial(cfg);
}
