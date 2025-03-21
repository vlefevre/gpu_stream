#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <cstdlib>
#include <type_traits>

#include "cuda_kernels.cuh"

using namespace std;
using namespace chrono;

template <typename T>
string getTypeName() {
	if constexpr (is_same_v<T, int8_t>) return "int8_t";
	else if constexpr (is_same_v<T, int16_t>) return "int16_t";
	else if constexpr (is_same_v<T, int32_t>) return "int32_t";
	else if constexpr (is_same_v<T, int64_t>) return "int64_t";
	else if constexpr (is_same_v<T, float>) return "float";
	else if constexpr (is_same_v<T, double>) return "double";
	else return "unknown";
}


// Init an array on GPU with random values
template <typename T>
void createArray(T **array, size_t size)
{
	T *h_array = new T[size];

	random_device rd;
	mt19937 gen(rd());
	if constexpr (is_integral<T>::value) {
		uniform_int_distribution<T> dist(-100, 100);
		for (size_t i=0; i<size; i++)
			h_array[i] = static_cast<T>(dist(gen));
	} else {
		uniform_real_distribution<T> dist(-100.0, 100.0); // Example range for floating-point types
		for (size_t i=0; i<size; i++)
			h_array[i] = static_cast<T>(dist(gen));
	}

	cudaMalloc(array, sizeof(T)*size);
	cudaMemcpy(*array, h_array, sizeof(T)*size, cudaMemcpyHostToDevice);

	delete[] h_array;
}

// Delete an array on GPU
template <typename T>
void deleteArray(T *array)
{
	cudaFree(array);
}

int getNbArrays(const string& benchmark)
{
	if (benchmark == "initArray") {
		return 1;
	} else if (benchmark == "copyArray") {
		return 2;
	} else if (benchmark == "constScaleArray") {
		return 2;
	} else if (benchmark == "scaleArray") {
		return 3;
	} else if (benchmark == "addArray") {
		return 3;
	} else if (benchmark == "multArray") {
		return 3;
	} else if (benchmark == "axpy") {
		return 3;
	} else if (benchmark == "1write2read") {
		return 4;
	} else if (benchmark == "1write3read") {
		return 5;
	} else {
		cerr << "Unknown benchmark.\n";
		return 0;
	}
}

template <typename T>
void initialize(const string& benchmark, size_t size, T **array1, T **array2, T **array3, T **array4)
{
	cout << "Initializing arrays..." << endl;
	createArray(array1, size);
	if (benchmark == "copyArray") {
		createArray(array2, size);
	} else if (benchmark == "scaleArray") {
		createArray(array2, size);
	} else if (benchmark == "addArray") {
		createArray(array2, size);
		createArray(array3, size);
	} else if (benchmark == "multArray") {
		createArray(array2, size);
		createArray(array3, size);
	} else if (benchmark == "axpy") {
		createArray(array2, size);
	} else if (benchmark == "1write2read") {
		createArray(array2, size);
		createArray(array3, size);
	} else if (benchmark == "1write3read") {
		createArray(array2, size);
		createArray(array3, size);
		createArray(array4, size);
	}
}

template <typename T>
void cleanup(const string& benchmark, T *array1, T *array2, T *array3, T *array4)
{
	cout << "Cleaning meory..." << endl;
	deleteArray(array1);
	if (benchmark == "copyArray") {
		deleteArray(array2);
	} else if (benchmark == "scaleArray") {
		deleteArray(array2);
	} else if (benchmark == "addArray") {
		deleteArray(array2);
		deleteArray(array3);
	} else if (benchmark == "multArray") {
		deleteArray(array2);
		deleteArray(array3);
	} else if (benchmark == "axpy") {
		deleteArray(array2);
	} else if (benchmark == "1write2read") {
		deleteArray(array2);
		deleteArray(array3);
	} else if (benchmark == "1write3read") {
		deleteArray(array2);
		deleteArray(array3);
		deleteArray(array4);
	}
}

// Function template to run benchmarks
template <typename T>
void runBenchmark(const string& benchmark, size_t size, int gridSize, int blockSize) {

	int warmups = WARMUPS;
	int iters = ITERS;
	double avg = 0.0;

	T scalar = static_cast<T>(2);
	T *array1, *array2, *array3, *array4;


	double mem = static_cast<double>(sizeof(T)*size)/1024./1024.;
	cout << "Launching " << benchmark << " benchmark evaluation on arrays of " << size << " " << getTypeName<T>() << " elements (" << mem << " MB).\n";
	cout << "GPU kernel configuration: (" << gridSize << "x" << blockSize << ").\n";
	cout << "Warmup rounds: " << warmups << " / Evaluation rounds: " << iters << "." << endl;
	
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	double bandwidth = (prop.memoryClockRate * prop.memoryBusWidth * 2.0) / (8.0 * 1e6);
	cout << "Theoretical GPU bandwidth: " << bandwidth << " GB/s.\n";
	int nb_arrays = getNbArrays(benchmark);
	double total_load = mem * nb_arrays; 
	cout << "Estimated runtime: " << (total_load*1024.) / bandwidth << " µs." << endl;

	initialize(benchmark, size, &array1, &array2, &array3, &array4);

	for (int i=0; i<iters+warmups; i++)
	{
		auto start = high_resolution_clock::now();

		if (benchmark == "initArray") {
			initArray<<<gridSize, blockSize>>>(size, array1, scalar);
		} else if (benchmark == "copyArray") {
			copyArray<<<gridSize, blockSize>>>(size, array1, array2);
		} else if (benchmark == "constScaleArray") {
			constScaleArray<<<gridSize, blockSize>>>(size, array1, scalar);
		} else if (benchmark == "scaleArray") {
			scaleArray<<<gridSize, blockSize>>>(size, array1, array2);
		} else if (benchmark == "addArray") {
			addArray<<<gridSize, blockSize>>>(size, array1, array2, array3);
		} else if (benchmark == "multArray") {
			multArray<<<gridSize, blockSize>>>(size, array1, array2, array3);
		} else if (benchmark == "axpy") {
			axpy<<<gridSize, blockSize>>>(size, array1, array2, scalar);
		} else if (benchmark == "1write2read") {
			custom1write2read<<<gridSize, blockSize>>>(size, array1, array2, array3);
		} else if (benchmark == "1write3read") {
			custom1write3read<<<gridSize, blockSize>>>(size, array1, array2, array3, array4);
		} else {
			cerr << "Unknown benchmark." << endl;
			return;
		}

		cudaDeviceSynchronize();
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		cout << "Benchmark " << benchmark << " completed in " << duration.count() << " µs" << endl;

		if (i >= warmups)
			avg += duration.count();
	}
	cout << "Benchmark " << benchmark << " has an average execution time of " << avg/iters << " µs" << endl;

	cleanup(benchmark, array1, array2, array3, array4);
}

// Function to map string to type
template <typename Func>
void selectType(const string& typeStr, Func func, size_t size, const string& benchmark, int gridSize, int blockSize) {
	if (typeStr == "int8") func(int8_t{}, size, benchmark, gridSize, blockSize);
	else if (typeStr == "int16") func(int16_t{}, size, benchmark, gridSize, blockSize);
	else if (typeStr == "int32") func(int32_t{}, size, benchmark, gridSize, blockSize);
	else if (typeStr == "int64") func(int64_t{}, size, benchmark, gridSize, blockSize);
	else if (typeStr == "float") func(float{}, size, benchmark, gridSize, blockSize);
	else if (typeStr == "double") func(double{}, size, benchmark, gridSize, blockSize);
	else cerr << "Unsupported data type." << endl;
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cerr << "Usage: " << argv[0] << " <size> <datatype> <benchmark> [nbthreads [blocksize]]" << endl;
		return 1;
	}

	size_t size = atoi(argv[1]);
	string typeStr = argv[2];
	string benchmark = argv[3];
	int nb_threads = size; //by default use 1 thread per element
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	int blockSize = prop.maxThreadsPerBlock; //by default use max block size
	if (argc > 4)
	{
		nb_threads = atoi(argv[4]);
		if (argc > 5)
			blockSize = atoi(argv[5]);
	}

	int gridSize = (nb_threads + blockSize-1) / blockSize;

	selectType(typeStr, [](auto type, size_t size, const string& benchmark, int gs, int bs) {
		using T = decltype(type);
		runBenchmark<T>(benchmark, size, gs, bs);
	}, size, benchmark, gridSize, blockSize);

	return 0;
}

