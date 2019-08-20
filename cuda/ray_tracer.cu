/*
 * ray_tracing.cpp
 *
 *  Created on: Aug 18, 2019
 *      Author: 26sra
 */

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using std::cout;
using std::endl;
using std::stoll;

using cf_t = const float;
using llu_t = long long unsigned;
using cll_t = const long long;
using cllu_t = const unsigned long long;
constexpr float pi = 3.141592653589793238462643383279502884;
constexpr int nThreadsPerBlock = 256;
//constexpr int nBlocks = 128;
constexpr int R = 6;
constexpr int Wmax = 10;


__global__ void 
setup_kernel(curandState * state, cllu_t seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__device__ float
random_float(uint64_t * seed, float min, float max) {
	
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
		return sqrtf(x_*x_+y_*y_+z_*z_);
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
	direction_sampling(uint64_t * seed) {
		cf_t phi = random_float(seed, 0, 2*pi);
		cf_t cosTheta = random_float(seed, -1, 1);
		cf_t sinTheta = sqrtf(1-(cosTheta*cosTheta));
		x_ = sinTheta*cosf(phi);
		y_ = sinTheta*sinf(phi);
		z_ = cosTheta;
	}

};

__global__ void
ray_tracer(curandState * states, float * devG, 
		cllu_t G, cllu_t G2, cf_t GD2) {
	Vec3d W(0,10,0), V, C(0,12,0), N, S, I, L(4,4,-1), IC, LI;
	cf_t ref = GD2/Wmax;
	uint64_t seed = (threadIdx.x + blockIdx.x*blockDim.x)*4238811;
	
	do {
		V.direction_sampling(&seed);
		cf_t q = W.y_/V.y_;
		W.x_ = q*V.x_;
		W.y_ = q*V.y_;
		W.z_ = q*V.z_;
		if (k>10000) {
			atomicAdd(&devG[2], random_float(states, -1, 1));
			atomicAdd(&devG[3], random_float(states, -1, 1));
			atomicAdd(&devG[4], random_float(states, -1, 1));
			return;
		}
		k++;
	} while(!(fabsf(W.x_) < Wmax && fabsf(W.z_) < Wmax &&
			  powf(V.dot(C),2) + R*R - C.dot(C) > 0));
	cf_t vc = V.dot(C);
	cf_t t = vc - sqrtf(vc*vc + R*R - C.dot(C));
	I.set(t*V.x_, t*V.y_, t*V.z_);
	IC.set(I.x_-C.x_, I.y_-C.y_, I.z_-C.z_);
	cf_t icNorm = IC.norm();
	N.set(IC.x_/icNorm, IC.y_/icNorm, IC.z_/icNorm);
	LI.set(L.x_-I.x_, L.y_-I.y_, L.z_-I.z_);
	cf_t liNorm = LI.norm();
	S.set(LI.x_/liNorm, LI.y_/liNorm, LI.z_/liNorm);
	cf_t sn = S.dot(N);
	cf_t b = sn < 0 ? 0 : sn;
	cll_t x = (cll_t)floorf(W.x_*ref + GD2);
	cll_t y = (cll_t)floorf(W.z_*ref + GD2);
	atomicAdd(&devG[x*G + y], b);
	return;
}

int
main(int argc, char ** argv) {
	if (argc!=3) {
		cout << "Usage ./ray_tracing <number of rays> <gridpoints>" << endl;
	}   
	Config cfg(argv);
    curandState * devStates;
    cudaMalloc((void**)&devStates, sizeof(curandState)*cfg.N_);
    float * G;
	float * devG;
    
    G = (float *)calloc(cfg.G2_, sizeof(float));
    cudaMalloc((void**)&devG, cfg.G2_*sizeof(float));
    cudaMemset(devG, 0, cfg.G2_*sizeof(float));
    
//    cf_t t0 = omp_get_wtime();    
    cout << "here1" << endl;
	cllu_t threadsLaunch = cfg.N_ < nThreadsPerBlock ? cfg.N_ : nThreadsPerBlock;
	cllu_t blocksLaunch = cfg.N_/nThreadsPerBlock + 1;
	cout << threadsLaunch << endl;
	cout << blocksLaunch << endl;
    setup_kernel<<<blocksLaunch, threadsLaunch>>>(devStates, time(NULL));
    cout << "here2" << endl;
    ray_tracer<<<blocksLaunch, threadsLaunch>>>( 
    		devStates, devG, cfg.G_, cfg.G2_, cfg.GD2_);
    cout << "here3" << endl;
    cudaMemcpy(G, devG, cfg.G2_*sizeof(float), cudaMemcpyDeviceToHost);
//    cf_t t1 = omp_get_wtime();
//	cout << "Total Runtime (seconds): " << t1-t0 << endl;
    cout << "here4" << endl;
    cout << G[2] << endl;
    cout << G[3] << endl;
    cout << G[4] << endl;
	FILE * f = fopen("hw4.out", "wb");
	fwrite(G, sizeof(float), cfg.G2_, f);
	fclose(f);
	cout << "here5" << endl;
	cudaFree(devG);
	cudaFree(devStates);
    free(G);  
    cout << "here6" << endl;
    return 0;
}
