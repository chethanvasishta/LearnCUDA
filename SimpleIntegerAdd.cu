#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

__global__ void add(int *c, const int *a, const int *b)
{
	*c = *b + *a;
}

void SimpleIntegerAdd()
{
	cout << "learn cuda cpp" << endl;
	int a, b, c;
	cout << "Enter a & b" << endl;
	cin >> a >> b;

	// create memory
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	// copy to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add << <1, 1 >> >(d_c, d_a, d_b);

	// copy the output
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	// free the memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout << c;
}