#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

__global__ void AddIntegerArray(int *c, const int *a, const int *b)
{
	int idx = blockIdx.x;
	c[idx] = a[idx] + b[idx];
}

void PrintArray(int *a, int size)
{
	for (int i = 0; i < size; ++i)
		cout << a[i] << ",";
	cout << endl;
}

void CPUAdd(int *c, const int* a, const int* b, int size)
{
	for (int i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}

void GPUAdd(int *c, const int* a, const int* b, int arraySize)
{
	// allocate memory
	int *d_a, *d_b, *d_c; // = nullptr?
	const int numBytes = arraySize * sizeof(int);
	cudaMalloc(&d_a, numBytes);
	cudaMalloc(&d_b, numBytes);
	cudaMalloc(&d_c, numBytes);

	cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, numBytes, cudaMemcpyHostToDevice);

	AddIntegerArray << <arraySize, 1 >> > (d_c, d_a, d_b);

	cudaMemcpy(c, d_c, numBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

typedef void(adder_t)(int*, const int*, const int*, int);
double TimeFunction(adder_t func, int *c, const int *a, const int *b, int arraySize)
{
	auto start = high_resolution_clock::now();
	func(c, a, b, arraySize);
	auto stop = high_resolution_clock::now();
	return (duration_cast<milliseconds>(stop - start)).count();
}

void TestIntegerArrayAdd()
{
	cout << "Adding two arrays" << endl;

	const int arraySize = 10;
	int *a = new int[arraySize];
	int *b = new int[arraySize];
	int *c = new int[arraySize];

	for (int i = 0; i < arraySize; ++i)
	{
		a[i] = i * 10;
		b[i] = i * 15;
	}

	cout << "CPU took " << TimeFunction(CPUAdd,c, a, b, arraySize) << " ms" << endl;
	cout << "GPU took " << TimeFunction(GPUAdd, c, a, b, arraySize) << " ms" << endl;

	PrintArray(a, arraySize);
	PrintArray(b, arraySize);
	PrintArray(c, arraySize);

}