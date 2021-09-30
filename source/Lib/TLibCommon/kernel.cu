
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "TLibCommon/CommonDef.h"

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

extern "C" int top()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
} 

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

extern "C" int gemm(int const aW, int const aH, int const bW, int const bH, int const cW, int const cH, PRECISION *h_A, PRECISION *h_B, PRECISION *h_C)
{

	// 定义状态变量
	cublasStatus_t status;

	/*
	** GPU 计算矩阵相乘
	*/

	// 创建并初始化 CUBLAS 库对象
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS 对象实例化出错" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}

	PRECISION *d_A, *d_B, *d_C;
	// 在 显存 中为将要计算的矩阵开辟空间
	cudaMalloc(
		(void**)&d_A,    // 指向开辟的空间的指针
		aW*aH * sizeof(PRECISION)    //　需要开辟空间的字节数
		);
	cudaMalloc(
		(void**)&d_B,
		bW*bH * sizeof(PRECISION)
		);

	// 在 显存 中为将要存放运算结果的矩阵开辟空间
	cudaMalloc(
		(void**)&d_C,
		cW*cH * sizeof(PRECISION)
		);

	// 将矩阵数据传递进 显存 中已经开辟好了的空间
	cublasSetVector(
		aW*aH,    // 要存入显存的元素个数
		sizeof(PRECISION),    // 每个元素大小
		h_A,    // 主机端起始地址
		1,    // 连续元素之间的存储间隔
		d_A,    // GPU 端起始地址
		1    // 连续元素之间的存储间隔
		);
	cublasSetVector(
		bW*bH,
		sizeof(PRECISION),
		h_B,
		1,
		d_B,
		1
		);

	// 同步函数
	cudaThreadSynchronize();

	//Timer myTimer;
	//myTimer.start();
	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
	PRECISION a = 1; PRECISION b = 0;
	// 矩阵相乘。该函数必然将数组解析成列优先数组
	cublasDgemm(
		handle,    // blas 库对象
		CUBLAS_OP_N,    // 矩阵 A 属性参数
		CUBLAS_OP_N,    // 矩阵 B 属性参数
		bW,    // A, C 的行数
		aH,    // B, C 的列数
		bH,    // A 的列数和 B 的行数
		&a,    // 运算式的 α 值
		d_B,    // A 在显存中的地址
		bW,    // lda
		d_A,    // B 在显存中的地址
		aW,    // ldb
		&b,    // 运算式的 β 值
		d_C,    // C 在显存中的地址(结果矩阵)
		cW    // ldc
		);
	// 同步函数
	cudaThreadSynchronize();
	//myTimer.stop();
	//printf("use: %lf\n", myTimer.getElapsedTime());

	// 从 显存 中取出运算结果至 内存中去
	cublasGetVector(
		cW*cH,    //  要取出元素的个数
		sizeof(PRECISION),    // 每个元素大小
		d_C,    // GPU 端起始地址
		1,    // 连续元素之间的存储间隔
		h_C,    // 主机端起始地址
		1    // 连续元素之间的存储间隔
		);


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 释放 CUBLAS 库对象
	cublasDestroy(handle);


	//getchar();

	return 0;

}

