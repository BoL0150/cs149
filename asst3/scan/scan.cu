// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

__global__ void check_gpu_data(int *deivce_array, int length) {
  for (int i = 0; i < length; i += 1000) {
    printf("%d", deivce_array[i]);
  }
  printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
}
__global__ void upsweep_kernel(int computation_nums, int *result,
                               int two_dplus1, int two_d) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // 任务量不一定是线程块大小的整数倍，所以最后一个线程块会有一些线程超出任务量的范围，不要计算
  if (index < computation_nums) {
    int i = index * two_dplus1;
    result[i + two_dplus1 - 1] += result[i + two_d - 1];
    // printf("i +two_dplus1 - 1 : %d, result[i + two_dplus1 - 1]:%d\n", i,
    //        result[i + two_dplus1 - 1]);
  }
}
__global__ void downsweep_kernel(int computation_nums, int *result,
                                 int two_dplus1, int two_d) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < computation_nums) {
    int i = index * two_dplus1;
    int t = result[i + two_d - 1];
    // downsweep的最后一个数是0
    if (i == 0) {
      result[i + two_dplus1 - 1] = 0;
    }
    result[i + two_d - 1] = result[i + two_dplus1 - 1];
    result[i + two_dplus1 - 1] += t;
  }
}
// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int *input, int64_t N, int *result) {

  int64_t rounded_length = nextPow2(N);
  // upsweep phase
  for (int64_t two_d = 1; two_d <= rounded_length / 2; two_d *= 2) {
    int64_t two_dplus1 = 2 * two_d;
    int64_t computation_nums = rounded_length / two_dplus1;
    const int64_t blocks =
        (computation_nums + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(computation_nums, result,
                                                  two_dplus1, two_d);
    cudaDeviceSynchronize();
  }

  // downsweep phase
  for (int64_t two_d = rounded_length / 2; two_d >= 1; two_d /= 2) {
    int64_t two_dplus1 = 2 * two_d;
    int64_t computation_nums = rounded_length / two_dplus1;
    const int64_t blocks =
        (computation_nums + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(computation_nums, result,
                                                    two_dplus1, two_d);
    // std::cout << "fuckyou2 " << std::endl;
    cudaDeviceSynchronize();
  }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int *inarray, int *end, int *resultarray) {
  int *device_result;
  int *device_input;
  int N = end - inarray;

  // This code rounds the arrays provided to exclusive_scan up
  // to a power of 2, but elements after the end of the original
  // input are left uninitialized and not checked for correctness.
  //
  // Student implementations of exclusive_scan may assume an array's
  // allocated length is a power of 2 for simplicity. This will
  // result in extra work on non-power-of-2 inputs, but it's worth
  // the simplicity of a power of two only solution.

  int rounded_length = nextPow2(end - inarray);

  cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
  cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

  // For convenience, both the input and output vectors on the
  // device are initialized to the input values. This means that
  // students are free to implement an in-place scan on the result
  // vector if desired.  If you do this, you will need to keep this
  // in mind when calling exclusive_scan from find_repeats.
  cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int),
             cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  exclusive_scan(device_input, N, device_result);

  // Wait for completion
  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
             cudaMemcpyDeviceToHost);

  double overallDuration = endTime - startTime;
  return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int *inarray, int *end, int *resultarray) {

  int length = end - inarray;
  thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
  thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

  cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
             cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  thrust::exclusive_scan(d_input, d_input + length, d_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
             cudaMemcpyDeviceToHost);

  thrust::device_free(d_input);
  thrust::device_free(d_output);

  double overallDuration = endTime - startTime;
  return overallDuration;
}
__global__ void mark_repeats(int *input, int *marks, int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length - 1) {
    if (input[idx] == input[idx + 1]) {
      marks[idx] = 1;
    } else {
      marks[idx] = 0;
    }
  } else if (idx == length - 1) {
    marks[idx] = 0;
  }
  // printf("idx %d,marks[idx] %d\n", idx, marks[idx]);
}
__global__ void find_repeats_kernel(int *scan_result, int *marks, int length,
                                    int *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int output_idx = -1;
  if (idx < length - 1) {
    if (marks[idx] == 1) {
      output_idx = scan_result[idx];
      output[output_idx] = idx;
    }
  }
  // printf("idx %d,scan_result[idx] %d\n", idx, scan_result[idx]);
}
// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int *device_input, int length, int *device_output) {

  const int rounded_length = nextPow2(length);
  const int blocks =
      (rounded_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int *marks;
  int *scan_result;
  cudaMalloc(&marks, sizeof(int) * rounded_length);
  cudaMalloc(&scan_result, sizeof(int) * rounded_length);

  mark_repeats<<<blocks, THREADS_PER_BLOCK>>>(device_input, marks,
                                              rounded_length);
  cudaDeviceSynchronize();
  cudaMemcpy(scan_result, marks, sizeof(int) * rounded_length,
             cudaMemcpyDeviceToDevice);
  // 注意，exclusive_scan要求input和result数组都是2的幂，否则就会出错
  // 所以marks和scan_result一定要是2的幂
  exclusive_scan(marks, rounded_length, scan_result);
  find_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(
      scan_result, marks, rounded_length, device_output);
  cudaDeviceSynchronize();
  int count;
  cudaMemcpy(&count, scan_result + length - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaFree(marks);
  cudaFree(scan_result);
  return count;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output,
                       int *output_length) {

  int *device_input;
  int *device_output;
  int rounded_length = nextPow2(length);

  cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
  cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
  cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  double startTime = CycleTimer::currentSeconds();

  int result = find_repeats(device_input, length, device_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  // set output count and results array
  *output_length = result;
  cudaMemcpy(output, device_output, length * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);

  float duration = endTime - startTime;
  return duration;
}

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
