/*
* This program uses the host CURAND API to generate 100
* pseudorandom floats .
*/

//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <cmath>

//#define DENG_NOT_USING_CUBLAS_V2

//CUDA headers
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
////CUDA thrust
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/for_each.h>

#include "RK.cu"

int main()
{
	const size_t N_sample = 5000;
	const size_t dim = 2;
	const size_t N_t = 5000;
	const float tau = 100.0;

	//create CUDA PRNG
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	time_t seed;
	time(&seed);
	std::cout << "here?" << std::endl;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	std::cout << "start randomize" << std::endl;
	
	RungeKutta_CUDA::Runge_Kutta a(N_sample, dim, N_t, tau);

	


	for (int i = 0; i < 2; ++i)
	{
		curandGenerateNormal(gen, a.coord_dev[i], N_sample, 0, 1);
		curandGenerateNormal(gen, a.momtm_dev[i], N_sample, 0, 1);
	}
	std::cout << "finish randomize" << std::endl;


	a.run();

	return 0;
}