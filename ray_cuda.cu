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
#include <curand.h>
#include <curand_kernel.h>

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
constexpr int N = 100;
constexpr int R = 6;
constexpr int Wmax = 10;


__global__ void 
setup_kernel(curandState * state, cllu_t seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__device__ double
random_double( curandState * state, double min, double max ) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[id];
	double ret = min + curand_uniform(state+id)*(max - min);
	state[id] = localState;
	return ret;
}

struct Config {
	cllu_t N_;
	cllu_t G_;
	cllu_t G2_;
	cd_t GD2_;

	Config(char ** argv) : N_(stoll(argv[1])),
			G_(stoll(argv[2])), G2_(G_*G_), GD2_(G_/2.) {
	}
};

struct Vec3d {
	double x_ = 0;
	double y_ = 0;
	double z_ = 0;

	__device__
	Vec3d() {}
	
	__device__
	Vec3d(double x, double y, double z) : x_(x), y_(y), z_(z) {}

	__device__ double
	norm() const {
		return sqrt(x_*x_+y_*y_+z_*z_);
	}

	__device__ double
	dot(Vec3d other) {
		return other.x_*x_ + other.y_*y_ + other.z_*z_;
	}

	__device__ void
	Set(double x, double y, double z) {
		x_ = x;
		y_ = y;
		z_ = z;
	}

	__device__ void
	Set(Vec3d & other) {
		x_ = other.x_;
		y_ = other.y_;
		z_ = other.z_;
	}

	__device__ void
	direction_sampling(curandState * state) {
		cd_t phi = random_double(state, 0, 2*pi);
		cd_t cosTheta = random_double(state, -1, 1);
		cd_t sinTheta = sqrt(1-(cosTheta*cosTheta));
		x_ = sinTheta*cos(phi);
		y_ = sinTheta*sin(phi);
		z_ = cosTheta;
	}

};



//__device__ double 
//dot(double * a, double * b, cllu_t N) {
//	double ret = 0;
//	for (llu_t i=0; i<N; i++) {
//		ret += a[i]*b[i];
//	}
//	return ret;
//}
//
//__device__ double 
//copy_vec3d(double * a, double * b) {
//	a[0] = b[0];
//	a[1] = b[1];
//	a[2] = b[2];
//}


//
//__device__ double
//norm(double x, double y, double z) {
//	return sqrt(x*x+y*y+z*z);
//}

//__device__ void
//direction_sampling( curandState * state, double * vec ) {
//	cd_t phi = random_double(state, 0, 2*pi);
//	cd_t cosTheta = random_double(state, -1, 1);
//	cd_t sinTheta = sqrt(1-(cosTheta*cosTheta));
//	vec[0] = sinTheta*cos(phi);
//	vec[1] = sinTheta*sin(phi);
//	vec[2] = cosTheta
//	return;
//}

__global__ void
ray_cuda(double * G, cllu_t G, cllu_t G2) {
	Vec3d W(0,10,0), V, C(0,12,0), N, S, I, L(4,4,-1), IC, LI;
	
	cd_t ref = cfg.GD2_/Wmax;
	
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
	
}

int
main(int argc, char ** argv) {
	if (argc!=3) {
		cout << "Usage ./ray_tracing <number of rays> <gridpoints>" << endl;
	}
	
	double * a;
	double * dev_a;
    
    curandState * d_states;
    cudaMalloc(&d_states, sizeof(curandState)*N);
      
    a = (double *)malloc( N*sizeof(double) );
//    b = (int*)malloc( size );
//    c = (int*)malloc( sizeof(int) );

//    for (i=0;i<N;++i){
//        a[i] = 1;
//        b[i] = i;
//    }
//	*c = 0;
    cudaMalloc( (void**)&dev_a, N*sizeof( double ) );
    double t0 = omp_get_wtime();
//    cudaMalloc( (void**)&dev_b, size );
//    cudaMalloc( (void**)&dev_c, sizeof(int) );

//    cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
//    cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
//    cudaMemcpy( dev_c, c, sizeof(int), cudaMemcpyHostToDevice );
    
    // launch add() kernel on GPU, passing parameters
//    dot<<< N/nThreadsPerBlock, nThreadsPerBlock >>>( dev_a, dev_b, dev_c );
	cllu_t threadsLaunch = N < nThreadsPerBlock ? N : nThreadsPerBlock;
    setup_kernel<<<N/nThreadsPerBlock, threadsLaunch>>>( d_states, time(NULL) );
    random_double<<<N/nThreadsPerBlock + 1, threadsLaunch>>>( d_states, dev_a, 10, 15 );
    // copy device result back to host copy of c
//    cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( a, dev_a, sizeof(double)*N, cudaMemcpyDeviceToHost);
    
    double t1 = omp_get_wtime();
	cout << "Total Runtime (seconds): " << t1-t0 << endl;

	FILE * f = fopen("hw4.out", "wb");
	fwrite(G, sizeof(double), cfg.G2_, f);
	fclose(f);

	cudaFree( dev_a );
    free(a);
//    cudaFree( dev_b );
//    cudaFree( dev_c );
//    printf("c:%d\n", *c);
    
    return 0;
}
