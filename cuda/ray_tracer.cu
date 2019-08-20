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

using cf_t = const float;
using llu_t = long long unsigned;
using cll_t = const long long;
using cllu_t = const unsigned long long;
constexpr float pi = 3.141592653589793238462643383279502884;
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

__device__ float
random_float( curandState * state, float min, float max ) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[id];
	float ret = min + curand_uniform(state+id)*(max - min);
	state[id] = localState;
	return ret;
}

struct Config {
	cllu_t N_;
	cllu_t G_;
	cllu_t G2_;
	cf_t GD2_;

	Config(char ** argv) : N_(stoll(argv[1])),
			G_(stoll(argv[2])), G2_(G_*G_), GD2_(G_/2.) {
	}
};

struct Vec3d {
	float x_ = 0;
	float y_ = 0;
	float z_ = 0;

	__device__
	Vec3d() {}
	
	__device__
	Vec3d(float x, float y, float z) : x_(x), y_(y), z_(z) {}

	__device__ float
	norm() const {
		return sqrt(x_*x_+y_*y_+z_*z_);
	}

	__device__ float
	dot(Vec3d other) {
		return other.x_*x_ + other.y_*y_ + other.z_*z_;
	}

	__device__ void
	set(float x, float y, float z) {
		x_ = x;
		y_ = y;
		z_ = z;
	}

	__device__ void
	set(Vec3d & other) {
		x_ = other.x_;
		y_ = other.y_;
		z_ = other.z_;
	}

	__device__ void
	direction_sampling(curandState * state) {
		cf_t phi = random_float(state, 0, 2*pi);
		cf_t cosTheta = random_float(state, -1, 1);
		cf_t sinTheta = sqrt(1-(cosTheta*cosTheta));
		x_ = sinTheta*cos(phi);
		y_ = sinTheta*sin(phi);
		z_ = cosTheta;
	}

};



//__device__ float 
//dot(float * a, float * b, cllu_t N) {
//	float ret = 0;
//	for (llu_t i=0; i<N; i++) {
//		ret += a[i]*b[i];
//	}
//	return ret;
//}
//
//__device__ float 
//copy_vec3d(float * a, float * b) {
//	a[0] = b[0];
//	a[1] = b[1];
//	a[2] = b[2];
//}


//
//__device__ float
//norm(float x, float y, float z) {
//	return sqrt(x*x+y*y+z*z);
//}

//__device__ void
//direction_sampling( curandState * state, float * vec ) {
//	cf_t phi = random_float(state, 0, 2*pi);
//	cf_t cosTheta = random_float(state, -1, 1);
//	cf_t sinTheta = sqrt(1-(cosTheta*cosTheta));
//	vec[0] = sinTheta*cos(phi);
//	vec[1] = sinTheta*sin(phi);
//	vec[2] = cosTheta
//	return;
//}

__global__ void
ray_tracer(curandState * states, float * devG, 
		cllu_t N, cllu_t G, cllu_t G2, cf_t GD2) {
	Vec3d W(0,10,0), V, C(0,12,0), N, S, I, L(4,4,-1), IC, LI;
	cf_t ref = GD2/Wmax;
	do {
		V.direction_sampling(states);
		cf_t q = W.y_/V.y_;
		W.x_ = q*V.x_;
		W.y_ = q*V.y_;
		W.z_ = q*V.z_;
	} while(!(abs(W.x_) < Wmax && abs(W.z_) < Wmax &&
			  pow(V.dot(C),2) + R*R - C.dot(C) > 0));
	
	cf_t vc = V.dot(C);
	cf_t t = vc - sqrt(vc*vc + R*R - C.dot(C));
	I.set(t*V.x_, t*V.y_, t*V.z_);
	IC.set(I.x_-C.x_, I.y_-C.y_, I.z_-C.z_);
	cf_t icNorm = IC.norm();
	N.set(IC.x_/icNorm, IC.y_/icNorm, IC.z_/icNorm);
	LI.set(L.x_-I.x_, L.y_-I.y_, L.z_-I.z_);
	cf_t liNorm = LI.norm();
	S.set(LI.x_/liNorm, LI.y_/liNorm, LI.z_/liNorm);
	cf_t sn = S.dot(N);
	cf_t b = sn < 0 ? 0 : sn;

	cll_t x = (cll_t)floor(W.x_*ref + GD2);
	cll_t y = (cll_t)floor(W.z_*ref + GD2);
	atomicAdd(devG[x*G + y], b);
}

int
main(int argc, char ** argv) {
	if (argc!=3) {
		cout << "Usage ./ray_tracing <number of rays> <gridpoints>" << endl;
	}   
	Config(argv) cfg;
    curandState * devStates;
    cudaMalloc(&devStates, sizeof(curandState)*cfg.N_);
    float * G;
	float * devG;
    
    G = (float *)calloc(cfg.G2_, sizeof(float));
    cudaMalloc( (void**)&devG, cfg.G2_*sizeof( float ) );
    cudaMemSet(devG, 0, cfg.G2_*sizeof( float ));
    
    cf_t t0 = omp_get_wtime();    
	cllu_t threadsLaunch = N < nThreadsPerBlock ? N : nThreadsPerBlock;
    setup_kernel<<<N/nThreadsPerBlock, threadsLaunch>>>( devStates, time(NULL) );
    ray_tracer<<<N/nThreadsPerBlock + 1, threadsLaunch>>>( 
    		devStates, devG, cfg.G_, cfg.G2_, cfg.GD2_);
    
    cudaMemcpy(G, devG, cfg.G2_*sizeof(float), cudaMemcpyDeviceToHost);
    cf_t t1 = omp_get_wtime();
	cout << "Total Runtime (seconds): " << t1-t0 << endl;
	
	FILE * f = fopen("hw4.out", "wb");
	fwrite(G, sizeof(float), cfg.G2_, f);
	fclose(f);

	cudaFree(devG);
	cudaFree(devStates);
    free(G);    
    return 0;
}
