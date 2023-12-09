/*
* warmup()
* To get DCU initialization ready
*/
__global__ void warmup() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}
