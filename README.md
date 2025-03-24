# GPU Stream benchmark

Inspired by the STREAM benchmark (https://www.cs.virginia.edu/stream/), this application evaluates the performance
of simple GPU memory-bound kernels. It currently runs on NVIDIA GPUs, using the nvcc compiler.

## How to compile

Just run `make`. You need the cuda toolkit to compile.
Number of iterations and warmups can be adjusted with `make it=100 wu=5` for example.

## How to use

`./gpu_stream size datatype benchmark [nbthreads [blocksize]]`

`size` is the number of elements in the 1D arrays.
`datatype` can be either int8, int16, int32, int64, float or double.
`benchmark` is the name of the benchmark to run.
Optionnally, you can specify the number of threads to launch and the number of threads in each GPU threadblock.
By default, the number of threads will be equal to the array size and the blocksize will be the maximum possible value.

## List of benchmarks

- initArray: init a 1D array to a constant value.
- copyArray: copy the content of a 1D array to another.
- constScaleArray: multiply a 1D array by a constant value.
- scaleArray: multiply a 1D array by non-constant values (element-wise).
- addArray: perform element-wise addition of two arrays and store the result into a third.
- multArray: perform element-wise multiplication of two arrays and store the result into a third.
- axpy: axpy kernel (y[i] += alpha * x[i])
- triad: Triad operation from STREAM (z[i] = y[i] + alpha * x[i])
- 1write2read: element-wise operation with 1 array as input/output and 2 arrays as input.
- 1write3read: element-wise operation with 1 array as input/output and 3 arrays as input.

## Examples

`./gpu_stream 1048576 int64 copyArray`

This will run the copyArray benchmark on 8 MB arrays (1024^2 int64_t elements) using maximum threadblock size (1 thread per array entry).

`./gpu_stream 134217728 float scaleArray 67108864 512`

This will run the scaleArray benchmark on 512 MB arrays (512^3 float elements) using threadblocks of 512 threads (1 thread per 2 array entries).
