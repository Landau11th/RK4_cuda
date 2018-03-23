#include <iostream>

#include <cuda.h>
#include <cublas_v2.h>
//#include <cuda_runtime.h>
#include "device_launch_parameters.h"

namespace RungeKutta_CUDA
{
	typedef float real;
	cublasHandle_t cublas_hd;
	cublasStatus_t cublas_stat = cublasCreate(&cublas_hd);
	const size_t thrd_per_blk = 512;

	__global__ void elementwise_multi(real* multi, real* res, int length)
	{
		int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;
		while (thrd_id < length)
		{
			res[thrd_id] *= multi[thrd_id];
			thrd_id += blockDim.x* gridDim.x;
		}
	}
	__global__ void zeros(real *res, int size)
	{
		int thrd_id = threadIdx.x + blockIdx.x * blockDim.x;
		while (thrd_id < size)
		{
			res[thrd_id] = 0;
			thrd_id += blockDim.x* gridDim.x;
		}
	}


	real ** alloc_mem(size_t dim_, size_t N_sample_)
	{
		real**temp = new real*[dim_];
		real *first = nullptr;
		cudaMalloc((void **)& first, dim_ * N_sample_ * sizeof(float));
		for (int i = 0; i < dim_; ++i)
		{
			temp[i] = first + i*N_sample_;
		}
		//real **temp = nullptr;
		//cudaMalloc((void **)& temp, dim_ * sizeof(float*));
		//for(int i = 0; i < dim_; ++i)
		//{
		//    cudaMalloc((void **)& temp[i], N_sample_ * sizeof(float));
		//}
		return temp;
	}

	real ** free_mem(real **temp, int dim_)
	{
		for (int i = 0; i < dim_; ++i)
		{
			cudaFree(temp[i]);
		}
		delete[] temp;
		return nullptr;
	}


	void copy(real **from, int rows, int size, real **res)
	{
		for (int i = 0; i < rows; ++i)
		{
			//std::cout << __func__ << " here?" << i<< std::endl;
			cublasScopy(cublas_hd, size, from[i], 1, res[i], 1);
		}
	}
	void add(real **added, int rows, int size, real multi, real **res)
	{
		for (int i = 0; i < rows; ++i)
		{
			cublasSaxpy(cublas_hd, size, &multi, added[i], 1, res[i], 1);
		}
	}
	void copy_and_add(real **from, int rows, int size, real multi, real **added, real **res)
	{
		copy(from, rows, size, res);
		add(added, rows, size, multi, res);
	}




	class Runge_Kutta
	{
	private:
		const size_t N_sample;
		const size_t dim;
		const size_t N_t;
		const real tau;
		const real dt;
		//time-independent parameters could be easily added in subclass
		//lambdas are for time-dependent parameters
		real **lambda;
		real **lambda_half;

	public:
		//coordinates and momentum stored in device
		real **coord_dev;
		real **momtm_dev;

		//intermediate variables needed during time evolution
		real **deri_coord_dev;
		real **deri_momtm_dev;
		real **temp_coord_dev;
		real **temp_momtm_dev;
		real **temp_deri_coord_dev;
		real **temp_deri_momtm_dev;


	public:
		Runge_Kutta(size_t N_sample_, size_t dim_, size_t N_t_, real tau_)
			:N_sample(N_sample_), dim(dim_), N_t(N_t_), tau(tau_), dt(tau / ((real)N_t_))
		{
			coord_dev = alloc_mem(dim, N_sample);
			momtm_dev = alloc_mem(dim, N_sample);
			deri_coord_dev = alloc_mem(dim, N_sample);
			deri_momtm_dev = alloc_mem(dim, N_sample);
			temp_coord_dev = alloc_mem(dim, N_sample);
			temp_momtm_dev = alloc_mem(dim, N_sample);
			temp_deri_coord_dev = alloc_mem(dim, N_sample);
			temp_deri_momtm_dev = alloc_mem(dim, N_sample);

			std::cout << "finished constructor" << std::endl;
		};

		//run RK method
		virtual void run()
		{
			//copy values to coord_device and momtm_device as initial condition


			//real *E_i = ;
			//E_initial(real **coord, real **momtm, real* E_i);

			for (int i_t = 0; i_t < N_t; ++i_t)
			{
				one_step_RK4(i_t);
				std::cout << "step " << i_t << std::endl;
			}

			//do stats and output
			//real *E_f = ;
			//E_final(coord, momtm, E_f);
		};

		//Runge-Kutta method of fourth order
		virtual void one_step_RK4(int i_t)
		{
			derivatives(coord_dev, momtm_dev, i_t, false, temp_deri_coord_dev, temp_deri_momtm_dev);

			copy(temp_deri_coord_dev, dim, N_sample, deri_coord_dev);
			copy(temp_deri_momtm_dev, dim, N_sample, deri_momtm_dev);

			copy_and_add(coord_dev, dim, N_sample, dt / 2.0, temp_deri_coord_dev, temp_coord_dev);
			copy_and_add(momtm_dev, dim, N_sample, dt / 2.0, temp_deri_momtm_dev, temp_momtm_dev);
			//            temp_coord_dev = coord_dev;
			//            temp_coord_dev += temp_deri_coord_dev*dt/2.0;
			//            temp_momtm_dev = momtm_dev;
			//            temp_momtm_dev += temp_deri_momtm_dev*dt/2.0;
			derivatives(temp_coord_dev, temp_momtm_dev, i_t, true, temp_deri_coord_dev, temp_deri_momtm_dev);

			add(temp_deri_coord_dev, dim, N_sample, 2.0, deri_coord_dev);
			add(temp_deri_momtm_dev, dim, N_sample, 2.0, deri_momtm_dev);

			copy_and_add(coord_dev, dim, N_sample, dt / 2.0, temp_deri_coord_dev, temp_coord_dev);
			copy_and_add(momtm_dev, dim, N_sample, dt / 2.0, temp_deri_momtm_dev, temp_momtm_dev);
			derivatives(temp_coord_dev, temp_momtm_dev, i_t, true, temp_deri_coord_dev, temp_deri_momtm_dev);

			add(temp_deri_coord_dev, dim, N_sample, 2.0, deri_coord_dev);
			add(temp_deri_momtm_dev, dim, N_sample, 2.0, deri_momtm_dev);

			copy_and_add(coord_dev, dim, N_sample, dt, temp_deri_coord_dev, temp_coord_dev);
			copy_and_add(momtm_dev, dim, N_sample, dt, temp_deri_momtm_dev, temp_momtm_dev);
			derivatives(temp_coord_dev, temp_momtm_dev, i_t, false, temp_deri_coord_dev, temp_deri_momtm_dev);

			add(temp_deri_coord_dev, dim, N_sample, 1.0, deri_coord_dev);
			add(temp_deri_momtm_dev, dim, N_sample, 1.0, deri_momtm_dev);

			add(deri_coord_dev, dim, N_sample, dt / 6.0, coord_dev);
			add(deri_momtm_dev, dim, N_sample, dt / 6.0, momtm_dev);
		};

		//calculate derivatives for an array of coordinates
		virtual void derivatives(real **coord_, real **momtm_, int i_t, bool if_half, real **deri_coord_dev_, real **deri_momtm_dev_)
		{
			copy(momtm_, dim, N_sample, deri_coord_dev_);

			for (int i = 0; i < dim; ++i)
			{
				zeros<<< 2, thrd_per_blk >>> (deri_momtm_dev_[i], N_sample);
			}

			add(coord_, dim, N_sample, -1.0, deri_momtm_dev_);

		};

		//calculate initial and final energy
		virtual void E_initial(real **coord, real **momtm, real* E, int i)
		{

		};

	};

}