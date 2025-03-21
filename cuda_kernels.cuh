template <typename T>
__global__ void initArray(size_t N, T *arr, T val)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		arr[tid] = val;
	}	
}

template <typename T>
__global__ void copyArray(size_t N, T *dst, T* src)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		dst[tid] = src[tid];
	}	
}

template <typename T>
__global__ void constScaleArray(size_t N, T *arr, T val)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		arr[tid] = arr[tid]*val;
	}	
}

template <typename T>
__global__ void scaleArray(size_t N, T *arr, T* val)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		arr[tid] = arr[tid]*val[tid];
	}
}

template <typename T>
__global__ void addArray(size_t N, T *x, T *y, T *z)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		z[tid] = x[tid] + y[tid];
	}
}

template <typename T>
__global__ void multArray(size_t N, T *x, T *y, T *z)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		z[tid] = x[tid] * y[tid];
	}
}

template <typename T>
__global__ void axpy(size_t N, T *x, T*y, T alpha)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		y[tid] = y[tid] + x[tid]*alpha;
	}
}

template <typename T>
__global__ void custom1write2read(size_t N, T *w, T *r1, T *r2)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		w[tid] = w[tid]*r1[tid] + r2[tid];
	}
}

template <typename T>
__global__ void custom1write3read(size_t N, T *w, T *r1, T *r2, T *r3)
{
	for (size_t tid=threadIdx.x + blockIdx.x * blockDim.x; tid<N; tid += gridDim.x * blockDim.x)
	{
		w[tid] = w[tid]*r1[tid] + r2[tid] + r3[tid];
	}
}
