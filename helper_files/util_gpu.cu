#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include <boost/random.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <thrust/sort.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <float.h> 
#endif

/* Using updated (v2) interfaces to cublas and cusparse */
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
//#include "helper_cusolver.h"
#include "cusolverDn.h"

#include "util.h"
#include "util_gpu.cuh"




//============================================================================================
// Kernel utilities
//============================================================================================

const long long int CUDA_NUM_BLOCKS = (long long int)65535;

// use 512 threads per block
const long long int CUDA_NUM_THREADS = (long long int)512;

// number of blocks for threads.
long long int GET_BLOCKS(const long long int N) 
{
  return (N + CUDA_NUM_THREADS - (long long int)1) / CUDA_NUM_THREADS;
}


//============================================================================================
// CUDA ERROR CHECKING
//============================================================================================


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
    if (abort) exit(code);
  }
}


template<typename Dtype>
__global__ void gpu_isBad_kernel(const Dtype* A, const int size, bool* isBad) 
{
  //performs C = A * B
  CUDA_KERNEL_LOOP(i, size) {
    if (::isinf(A[i]) || ::isnan(A[i])){
      isBad[0] = true;
    };
  };
}

template<typename Dtype>
bool gpu_isBad(const Dtype* A, long long int size) 
{

  long long int num_gpu_blocks= GET_BLOCKS(size);

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    int num_loops = 0;
    int num_entries = CUDA_NUM_BLOCKS*512;
    int spot = 0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_isBad_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(A +spot , num_entries, isBad);
      num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS;
      num_loops += 1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_isBad_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A + spot , size - spot, isBad);
  }else{
    gpu_isBad_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A , size, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  return isBad_host;
}

template bool gpu_isBad<float>(const float* A, long long int size) ;

//============================================================================================
// Device information utilities
//============================================================================================

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int getDeviceVersion(void)
  {
    int device;
    struct cudaDeviceProp properties;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
      printf("failed to get device\n");
      return 0;
    }

    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
      printf("failed to get properties\n");
      return 0;
    }

    return properties.major * 100 + properties.minor * 10;
  }

size_t getDeviceMemory(void)
  {
    struct cudaDeviceProp properties;
    int device;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
      return 0;
    }

    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
      return 0;
    }

    return properties.totalGlobalMem;
  }
#if defined(__cplusplus)
}
#endif /* __cplusplus */



//============================================================================================
// Status Check
//============================================================================================


/* For debugging */
void printCusparseStatus(cusparseStatus_t status)
{
  switch(status)
  {
    case CUSPARSE_STATUS_NOT_INITIALIZED:
    printf( "CUSPARSE_STATUS_NOT_INITIALIZED\n" );
    break;
    case CUSPARSE_STATUS_ALLOC_FAILED:
    printf( "CUSPARSE_STATUS_ALLOC_FAILED\n" );
    break;
    case CUSPARSE_STATUS_INVALID_VALUE:
    printf( "CUSPARSE_STATUS_INVALID_VALUE\n" );
    break;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
    printf( "CUSPARSE_STATUS_ARCH_MISMATCH\n" );
    break;
    case CUSPARSE_STATUS_MAPPING_ERROR:
    printf( "CUSPARSE_STATUS_MAPPING_ERROR\n" );
    break;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
    printf( "CUSPARSE_STATUS_EXECUTION_FAILED\n" );
    break;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
    printf( "CUSPARSE_STATUS_INTERNAL_ERROR\n" );
    break;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    printf( "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n" );
    break;
    case CUSPARSE_STATUS_ZERO_PIVOT:
    printf( "CUSPARSE_STATUS_ZERO_PIVOT\n" );
  }
}

/* For debugging */
void printCublasStatus(cublasStatus_t status)
{
  switch(status)
  {
    case CUBLAS_STATUS_NOT_INITIALIZED:
    printf( "CUBLAS_STATUS_NOT_INITIALIZED\n" );
    break;
    case CUBLAS_STATUS_ALLOC_FAILED:
    printf( "CUBLAS_STATUS_ALLOC_FAILED\n" );
    break;
    case CUBLAS_STATUS_INVALID_VALUE:
    printf( "CUBLAS_STATUS_INVALID_VALUE\n" );
    break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
    printf( "CUBLAS_STATUS_ARCH_MISMATCH\n" );
    break;
    case CUBLAS_STATUS_MAPPING_ERROR:
    printf( "CUBLAS_STATUS_MAPPING_ERROR\n" );
    break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
    printf( "CUBLAS_STATUS_EXECUTION_FAILED\n" );
    break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
    printf( "CUBLAS_STATUS_INTERNAL_ERROR\n" );
    break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
    printf( "CUBLAS_STATUS_NOT_SUPPORTED\n" );
    break;
    case CUBLAS_STATUS_LICENSE_ERROR:
    printf( "CUBLAS_STATUS_LICENSE_ERROR\n" );
  }
}

void checkCublasBatchedStatus(cublasStatus_t * batchStatus, int num_blocks)
{
  cublasStatus_t status;
  for (int i = 0; i < num_blocks; i++)
  {
    status = batchStatus[i];
    if (status != CUBLAS_STATUS_SUCCESS){
      switch(status)
      {
        case CUBLAS_STATUS_NOT_INITIALIZED:
        printf( "CUBLAS_STATUS_NOT_INITIALIZED\n" );
        break;
        case CUBLAS_STATUS_ALLOC_FAILED:
        printf( "CUBLAS_STATUS_ALLOC_FAILED\n" );
        break;
        case CUBLAS_STATUS_INVALID_VALUE:
        printf( "CUBLAS_STATUS_INVALID_VALUE\n" );
        break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
        printf( "CUBLAS_STATUS_ARCH_MISMATCH\n" );
        break;
        case CUBLAS_STATUS_MAPPING_ERROR:
        printf( "CUBLAS_STATUS_MAPPING_ERROR\n" );
        break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
        printf( "CUBLAS_STATUS_EXECUTION_FAILED\n" );
        break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
        printf( "CUBLAS_STATUS_INTERNAL_ERROR\n" );
        break;
        case CUBLAS_STATUS_NOT_SUPPORTED:
        printf( "CUBLAS_STATUS_NOT_SUPPORTED\n" );
        break;
        case CUBLAS_STATUS_LICENSE_ERROR:
        printf( "CUBLAS_STATUS_LICENSE_ERROR\n" );
      }
    };
  };
}




//============================================================================================
// Prints and Saves
//============================================================================================

template<typename Dtype>
void print_gpu_mtx_entries(const Dtype* array, int lda, int sda, bool transpose)
{

  const Dtype *host_pointer;
  
  host_pointer=(const Dtype *)malloc(lda * sda * sizeof(Dtype));
  checkErrors(host_pointer);
  checkCudaErrors(cudaMemcpy((void*)host_pointer, array, lda * sda * sizeof(Dtype),cudaMemcpyDeviceToHost));

  std::string line;
  line="[ ";
    if(!transpose){
      for( int j = 0; j < sda; j+= 1 ) {
        for( int i = 0; i < lda; i+= 1 ) {
          if (j==sda-1){
            if (i==lda-1){
              line = (line + ToString<Dtype>(host_pointer[i  + j* lda])).c_str();
            }else{
              line = (line + ToString<Dtype>(host_pointer[i  + j* lda]) + " ; ").c_str();
            };
          }else{
            line = (line + ToString<Dtype>(host_pointer[i  + j* lda]) + " , ").c_str();
          };
        };
      };
    }else{
     for( int i = 0; i < lda; i+= 1 ) {
      for( int j = 0; j < sda; j+= 1 ) {
        if (j==sda-1){
          if (i==lda-1){
            line = (line + ToString<Dtype>(host_pointer[i  + j* lda])).c_str();
          }else{
            line = (line + ToString<Dtype>(host_pointer[i  + j* lda]) + " ; ").c_str();
          };
        }else{
          line = (line + ToString<Dtype>(host_pointer[i  + j* lda]) + " , ").c_str();
        };
      };
    };     
  }

  line = line+ " ];\n";
  LOG(line);

  free((void*)host_pointer);
  
}

template void print_gpu_mtx_entries<bool>(const bool* array, int lda, int sda, bool transpose);
template void print_gpu_mtx_entries<int>(const int* array, int lda, int sda, bool transpose);
template void print_gpu_mtx_entries<float>(const float* array, int lda, int sda, bool transpose);

template<typename Dtype>
void print_gpu_array_entries(const Dtype* array, int count)
{
  const Dtype *host_pointer;
  
  host_pointer=(const Dtype *)malloc(count * sizeof(Dtype));
  checkErrors(host_pointer);
  checkCudaErrors(cudaMemcpy((void*)host_pointer, array, count * sizeof(Dtype),cudaMemcpyDeviceToHost));

  std::string line;
  line="[ ";
    for( int i = 0; i < count; i+= 1 ) {
      if (i==count-1){
        line = (line + ToString<Dtype>(host_pointer[i ])).c_str();
      }else{
        line = (line + ToString<Dtype>(host_pointer[i]) + " , ").c_str();
      };
    };     

    line = line+ " ];\n";
  LOG(line);

  free((void*)host_pointer);
  
}

template void print_gpu_array_entries<int>(const int* array, int count);
template void print_gpu_array_entries<float>(const float* array, int count);

template<typename Dtype>
void save_device_array_to_file(const Dtype* A_dev, int count, std::string title)
{
  Dtype* A_host  = NULL;
  A_host = (Dtype *)malloc(count *  sizeof(Dtype)); 
  checkErrors(A_host);
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, count * sizeof(Dtype), cudaMemcpyDeviceToHost));

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };   
  };
  free(A_host);
}

template void save_device_array_to_file<int>(const int* A_dev, int count, std::string title);
template void save_device_array_to_file<float>(const float* A_dev, int count, std::string title);
template void save_device_array_to_file<double>(const double* A_dev, int count, std::string title);

template<typename Dtype>
void append_device_array_to_file(const Dtype* A_dev, int count, std::string title)
{
  Dtype A_host[count];
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str(), std::ofstream::app);
  //entries<<"; ";
  entries<<"\r\n";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };   
  };
}

template void append_device_array_to_file<int>(const int* A_dev, int count, std::string title);
template void append_device_array_to_file<float>(const float* A_dev, int count, std::string title);
template void append_device_array_to_file<double>(const double* A_dev, int count, std::string title);

template <>
void save_device_arrays_side_by_side_to_file<float>(const float* A_dev, const float* B_dev, int count, std::string title)
{

  float *A_host = NULL;
  float *B_host = NULL;
  A_host=(float *)malloc(count * sizeof(float));
  B_host=(float *)malloc(count * sizeof(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * sizeof(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");

  free(A_host);
  free(B_host);
}

void save_device_arrays_side_by_side_to_file(const int* A_dev, const float* B_dev, int count, std::string title)
{

  int *A_host = NULL;
  float *B_host = NULL;
  A_host=(int *)malloc(count * sizeof(int));
  B_host=(float *)malloc(count * sizeof(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * sizeof(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");

  free(A_host);
  free(B_host);
}

template <>
void save_device_arrays_side_by_side_to_file<float>(const float* A_dev, const float* B_dev, 
  const float* C_dev, int count, std::string title)
{

  float *A_host = NULL;
  float *B_host = NULL;
  float *C_host = NULL;
  A_host=(float *)malloc(count * sizeof(float));
  B_host=(float *)malloc(count * sizeof(float));
  C_host=(float *)malloc(count * sizeof(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * sizeof(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ]<<", "<<C_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");
  free(A_host);
  free(B_host);
  free(C_host);
}


void save_device_arrays_side_by_side_to_file(const int* A_dev, const int* B_dev, 
 const float* C_dev, int count, std::string title)
{

  int *A_host = NULL;
  int *B_host = NULL;
  float *C_host = NULL;
  A_host=(int *)malloc(count * sizeof(int));
  B_host=(int *)malloc(count * sizeof(int));
  C_host=(float *)malloc(count * sizeof(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * sizeof(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ]<<", "<<C_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");
  free(A_host);
  free(B_host);
  free(C_host);
}

template <>
void save_device_arrays_side_by_side_to_file<float>(const float* A_dev, const float* B_dev, 
  const float* C_dev, const float* D_dev, 
  int count, std::string title)
{

  float *A_host = NULL;
  float *B_host = NULL;
  float *C_host = NULL;
  float *D_host = NULL;
  A_host=(float *)malloc(count * sizeof(float));
  B_host=(float *)malloc(count * sizeof(float));
  C_host=(float *)malloc(count * sizeof(float));
  D_host=(float *)malloc(count * sizeof(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkErrors(D_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(D_host, D_dev, count * sizeof(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ]<<", "<<C_host[i ]<<", "<<D_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");
  free(A_host);
  free(B_host);
  free(C_host);
  free(D_host);
}

template<typename Dtype>
void save_device_mtx_to_file(const Dtype* A_dev, int lda, int sda, std::string title, bool transpose)
{

  Dtype* A_host = NULL;
  A_host = (Dtype *)malloc(lda*sda* sizeof(Dtype)); 
  checkErrors(A_host);
  // Dtype A_host[lda*sda];
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, lda*sda*sizeof(Dtype), cudaMemcpyDeviceToHost));

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  if(!transpose){
    for (int j = 0; j < sda; j++){
      //entries<<"***"<<j<<"***";
      for (int i = 0; i < lda; i++){
        entries<<A_host[(long long int)i + (long long int)j * (long long int)lda];
        if(i < lda - 1){
          entries<<", ";
        };
      };
      if(j < sda - 1){
        //entries<<"\r\n";
        entries<<"; ";
      };

    }; 
  }else{
    for (int i = 0; i < lda; i++){
      //entries<<"***"<<i<<"***";
      for (int j = 0; j < sda; j++){
        entries<<A_host[(long long int)i + (long long int)j * (long long int)lda];
        if(j < sda - 1){
          entries<<", ";
        };
      };
      if(i < lda - 1){
        //entries<<"\r\n";
        entries<<"; ";
      };
    };    
  }

  //entries<<"];";
  free(A_host);
}

template void save_device_mtx_to_file<int>(const int* A_dev, int lda, int sda, std::string title, bool transpose);
template void save_device_mtx_to_file<float>(const float* A_dev, int lda, int sda, std::string title, bool transpose);
template void save_device_mtx_to_file<double>(const double* A_dev, int lda, int sda, std::string title, bool transpose);

//============================================================================================
// Add and Scale
//============================================================================================

template <typename Dtype>
__global__ void add_scalar_kernel(const long long int  n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <typename Dtype>
void gpu_add_scalar(const long long int N, const Dtype alpha, Dtype* Y) 
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  // NOLINT_NEXT_LINE(whitespace/operators)
  long long int num_gpu_blocks = GET_BLOCKS(N);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      add_scalar_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(num_entries, alpha, Y + spot);

      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    add_scalar_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(N - spot, alpha, Y+spot);
  }else{
    add_scalar_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(N, alpha, Y);
  }

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if ( isBad_host == true){
    ABORT_IF_NEQ(0, 1, "gpu_reverse_bools given non bool array.") 
  };
}

template void gpu_add_scalar<float>(const long long int N, const float alpha, float* Y);
template void gpu_add_scalar<double>(const long long int N, const double alpha, double* Y);

template <>
void gpu_scale<float>(cublasHandle_t dn_handle, const long long int n, const float alpha, 
  const float *x, float* y) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasScopy(dn_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(dn_handle, n, &alpha, y, 1));
}

template <>
void gpu_scale<double>(cublasHandle_t dn_handle, const long long int n, const double alpha, 
  const double *x, double* y) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
    // This function scales copies a scaled version of the vector x by the scalar α  into the result y.
  CUBLAS_CHECK(cublasDcopy(dn_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(dn_handle, n, &alpha, y, 1));
}
template <>
void gpu_scale<float>(cublasHandle_t dn_handle, const long long int n, const float alpha, float *x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  // This function scales the vector x by the scalar α and overwrites it with the result.
  CUBLAS_CHECK(cublasSscal(dn_handle, n, &alpha, x, 1));
}

template <>
void gpu_scale<double>(cublasHandle_t dn_handle, const long long int n, const double alpha, double *x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  // This function scales the vector x by the scalar α and overwrites it with the result.
  CUBLAS_CHECK(cublasDscal(dn_handle, n, &alpha, x, 1));
}



//============================================================================================
// CURAND INITIALIZATION
//============================================================================================



__global__ void initCurand(curandState *state, const unsigned long seed, const int n)
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //curand_init(seed, idx, 0, &state[idx]);

  CUDA_KERNEL_LOOP(i, n){
    curand_init(seed, i, 0, &state[i]);
  }
}

template < typename Dtype>
__global__ void gpu_get_rand_bools_kernel(curandState *state, Dtype *a, const int n, float probability_of_success)
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //a[idx] = curand_uniform(&state[idx]);
  CUDA_KERNEL_LOOP(i, n){
    a[i] = (Dtype)(curand_uniform(&state[i]) < probability_of_success);
  }
}

// __global__ void testrand2(unsigned long seed, float *a){
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     curandState state;
//     curand_init(seed, idx, 0, &state);
//     a[idx] = curand_uniform(&state);
// }

template < typename Dtype>
void gpu_get_rand_bools(const long long int n,  Dtype* x, float probability_of_success) 
{
  curandState *devState;  
  gpuErrchk(cudaMalloc((void**)&devState, n*sizeof(curandState)));

  const unsigned long seed = cluster_seedgen();

  const long long int num_gpu_blocks= GET_BLOCKS(n);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  initCurand<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, seed, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_get_rand_bools_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, x, n, probability_of_success);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(devState);

}

template void gpu_get_rand_bools<bool>(const long long int n,  bool* x, float probability_of_success) ;
template void gpu_get_rand_bools<float>(const long long int n,  float* x, float probability_of_success) ;

__global__ void gpu_get_rand_groups_kernel(curandState *state, const int n,  int* x, float* probability_of_success, const int num_groups)
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //a[idx] = curand_uniform(&state[idx]);
  CUDA_KERNEL_LOOP(i, n){
    float rand_ = curand_uniform(&state[i]);
    float sum = (float)0.0;
    for(int j = 0; j < num_groups; j++){
      sum += probability_of_success[j];
      if(rand_ < sum || j == num_groups-1){
        x[i] =  j;
        break;
      };
    }
  }
}

void gpu_get_rand_groups(const long long int n,  int* x, float* probability_of_success, const int num_groups) 
{
  curandState *devState;  
  gpuErrchk(cudaMalloc((void**)&devState, n*sizeof(curandState)));

  const unsigned long seed = cluster_seedgen();

  const long long int num_gpu_blocks= GET_BLOCKS(n);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  initCurand<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, seed, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_get_rand_groups_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, n, x, probability_of_success, num_groups);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(devState);

}

template < typename Dtype>
__global__ void gpu_reverse_bools_kernel(const long long int n,  Dtype* x, bool* isBad)
{
  CUDA_KERNEL_LOOP(i, n){
    if(x[i] == (Dtype)0.0){
      x[i] = (Dtype)1.0;
    }else if (x[i] == (Dtype)1.0) {
      x[i] = (Dtype)0.0;
    }else{
      isBad[0] = true;
    }
  }
}

template < typename Dtype>
void gpu_reverse_bools(const long long int n,  Dtype* x) 
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));
  const long long int num_gpu_blocks = GET_BLOCKS(n);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };
  gpu_reverse_bools_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(n, x,isBad);

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if ( isBad_host == true){
    ABORT_IF_NEQ(0, 1, "gpu_reverse_bools given non bool array.") 
  };
}

template void gpu_reverse_bools<float>(const long long int n,  float* x); 


void gpu_rng_uniform(const long long int n, unsigned int* r) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
    // gpu_rng_uniform with two arguments generates integers in the range
    // [0, UINT_MAX].
  curandGenerator_t gen;
    /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, 
    CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
  CURAND_CHECK(curandGenerate(gen, r, n));
  CURAND_CHECK(curandDestroyGenerator(gen));
}

template <>
void gpu_rng_uniform<float>(cublasHandle_t handle, const long long int n, const float a, const float b,
  float* r) 
{
    // gpu_rng_uniform with four arguments generates floats in the range
    // (a, b] (strictly greater than a, less than or equal to b) due to the
    // specification of curandGenerateUniform.  With a = 0, b = 1, just calls
    // curandGenerateUniform; with other limits will shift and scale the outputs
    // appropriately after calling curandGenerateUniform.
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  curandGenerator_t gen;
    /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
  CURAND_CHECK(curandGenerateUniform(gen, r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    gpu_scale(handle, n, range, r);
  }
  if (a != static_cast<float>(0)) {
    gpu_add_scalar(n, a, r);
  }
  CURAND_CHECK(curandDestroyGenerator(gen));
}

template <>
void gpu_rng_uniform<double>(cublasHandle_t handle, const long long int n, const double a, const double b,
  double* r) 
{
    // gpu_rng_uniform with four arguments generates doubles in the range
    // (a, b] (strictly greater than a, less than or equal to b) due to the
    // specification of curandGenerateUniform.  With a = 0, b = 1, just calls
    // curandGenerateUniform; with other limits will shift and scale the outputs
    // appropriately after calling curandGenerateUniform.
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  curandGenerator_t gen;
    /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
  CURAND_CHECK(curandGenerateUniformDouble(gen, r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    gpu_scale(handle, n, range, r);
  }
  if (a != static_cast<double>(0)) {
    gpu_add_scalar(n, a, r);
  }
  CURAND_CHECK(curandDestroyGenerator(gen));
}



template <>
void gpu_rng_gaussian<float>(const long long int n, const float mu, const float sigma, float* r) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1, "Long long long int too big");}
  if(n % (long long int)2 !=  (long long int)0) {ABORT_IF_EQ(0, 1, "Must request even number");}
  curandGenerator_t gen;
    /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
  CURAND_CHECK(curandGenerateNormal(gen, r, n, mu, sigma));
  CURAND_CHECK(curandDestroyGenerator(gen));
}

template <>
void gpu_rng_gaussian<double>(const long long int n, const double mu, const double sigma, double* r) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1, "Long long long int too big");}
  if(n % (long long int)2 !=  (long long int)0) {ABORT_IF_EQ(0, 1, "Must request even number");}
  curandGenerator_t gen;
    /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));
  CURAND_CHECK(curandGenerateNormalDouble(gen, r, n, mu, sigma));
  CURAND_CHECK(curandDestroyGenerator(gen));
}


template < typename Dtype>
void gpu_shuffle_array(cublasHandle_t handle, const long long int n,  Dtype* x)
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  float* order;
  CUDA_CHECK(cudaMalloc((void**)&order, n* sizeof(float)));

  gpu_rng_uniform<float>(handle, n, (float)0.0, (float)1.0, order);


  // print_gpu_array_entries<float>(order, 10 , 1 , n);
  // print_gpu_array_entries<float>(x, 10 , 1 , n);

  thrust::device_ptr<float> order_(order);
  thrust::device_vector<float> order__(order_, order_ + n);
  thrust::device_ptr<Dtype> r_(x);
  thrust::device_vector<Dtype> r__(r_, r_ + n);

  thrust::sort_by_key(order__.begin(), order__.end(), r__.begin());

  // thrust::copy(order__.begin(), order__.end(), order_);
  thrust::copy(r__.begin(), r__.end(), r_);

  // print_gpu_array_entries<Dtype>(order, 10 , 1 , n);
  // print_gpu_array_entries<Dtype>(r, 10 , 1 , n);

  cudaFree(order);

  // order__.clear();
  // r__.clear();
  // thrust::device_vector<Dtype>().swap(order__);
  // thrust::device_vector<Dtype>().swap(r__);
}

template void gpu_shuffle_array<float>(cublasHandle_t handle, const long long int n,  float* x);
template void gpu_shuffle_array<int>(cublasHandle_t handle, const long long int n,  int* x);

template <>
void gpu_sort_index_by_max<float>(cublasHandle_t handle, const long long int n,  float* x, float* indicies)
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
    // print_gpu_array_entries<float>(indicies, 10 , 1 , n);
    // print_gpu_array_entries<float>(x, 10 , 1 , n);

  thrust::device_ptr<float> indicies_(indicies);
  thrust::device_vector<float> indicies__(indicies_, indicies_ + n);
  thrust::device_ptr<float> r_(x);
  thrust::device_vector<float> r__(r_, r_ + n);

  thrust::sort_by_key(r__.begin(), r__.end(), indicies__.begin());

    // thrust::copy(indicies__.begin(), indicies__.end(), indicies_);
    // thrust::copy(r__.begin(), r__.end(), r_);

    // print_gpu_array_entries<float>(indicies, 10 , 1 , n);
    // print_gpu_array_entries<float>(r, 10 , 1 , n);


    // indicies__.clear();
    // r__.clear();
    // thrust::device_vector<float>().swap(indicies__);
    // thrust::device_vector<float>().swap(r__);
}

template <>
void gpu_sort_index_by_max<float>(cublasHandle_t handle, const long long int rows, const long long int  cols,  
  float* x, float* indicies)
{

    // print_gpu_array_entries<float>(indicies, 10 , 1 , n);
    // print_gpu_array_entries<float>(x, 10 , 1 , n);
  if(too_big(cols) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  for(long long int i = (long long int)0; i < rows; i++){
    thrust::device_ptr<float> indicies_(indicies + i * cols);
    thrust::device_vector<float> indicies__(indicies_, indicies_ + cols);
    thrust::device_ptr<float> r_(x + i * cols);
    thrust::device_vector<float> r__(r_, r_ + cols);

    thrust::sort_by_key(r__.begin(), r__.end(), indicies__.begin());

    // thrust::copy(indicies__.begin(), indicies__.end(), indicies_);
    // thrust::copy(r__.begin(), r__.end(), r_);

    // print_gpu_array_entries<float>(indicies, 10 , 1 , n);
    // print_gpu_array_entries<float>(r, 10 , 1 , n);


    // indicies__.clear();
    // r__.clear();
    // thrust::device_vector<float>().swap(indicies__);
    // thrust::device_vector<float>().swap(r__);
  }
}

//============================================================================================
// template wrappers for cuda functions and classic math
//============================================================================================


template<typename Dtype>
__global__ void gpu_set_all_kernel( Dtype* x, const int n, const Dtype alpha) 
{

  CUDA_KERNEL_LOOP(i, n) {
    //origin+x1+rows*y1
    x[i]=alpha;
  };
}

template < typename Dtype> 
void gpu_set_all(  Dtype* x, const long long int n, const Dtype alpha) 
{
  long long int num_gpu_blocks = GET_BLOCKS(n);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS*CUDA_NUM_THREADS);
    long long int spot = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_set_all_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>( x + spot, num_entries, alpha);

      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_set_all_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x + spot, n - spot, alpha);
  }else{
    gpu_set_all_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, n, alpha);
  }

}

template void gpu_set_all<int>(  int* x, const long long int n, const int alpha);
template void gpu_set_all<float>(  float* x, const long long int n, const float alpha);
template void gpu_set_all<double>(  double* x, const long long int n, const double alpha);


__global__ void gpu_set_as_index_kernel( int* x, const int n) 
{

  CUDA_KERNEL_LOOP(i, n) {
    //origin+x1+rows*y1
    x[i]=i;
  };
}


void gpu_set_as_index(  int* x, const long long int n) 
{
  long long int num_gpu_blocks= GET_BLOCKS(n);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_set_as_index_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, n);

}


template<typename Dtype>
__global__ void gpu_set_as_index_kernel( Dtype* x, const int rows, const int cols) 
{

  CUDA_KERNEL_LOOP(i, rows*cols) {
    int row = i % rows;
    int col = i / rows;
    x[i] = (Dtype)row;
  };
}

template <>
void gpu_set_as_index<float>(  float* x, const long long int rows, const long long int cols) 
{
  long long int num_gpu_blocks= GET_BLOCKS(rows*cols);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_set_as_index_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, rows,cols);

}


template<typename Dtype>
__global__ void gpu_set_as_index_host_kernel( Dtype* x, const long long int start, 
  const int num, const long long int rows, const long long int cols) 
{

  CUDA_KERNEL_LOOP(i, num) {
    int row = (int)(((long long int)i + start) % rows);
    int col =  (int)(((long long int)i + start) / rows);
    x[i] = (Dtype)row;
  };
}

template <>
void gpu_set_as_index_host<int>(  int* x_host, const long long int rows, const long long int cols) 
{
  if(1) LOG("called gpu_set_as_index_host") ;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  long long int num_gpu_blocks = GET_BLOCKS(rows*cols);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS*CUDA_NUM_THREADS);
    long long int spot = (long long int)0;

    int * x_dev;
    checkCudaErrors(cudaMalloc((void**)&x_dev, num_entries * sizeof(int)));

    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      checkCudaErrors(cudaMemcpy(x_dev, x_host + spot,  num_entries *  sizeof(int), cudaMemcpyHostToDevice));
      gpu_set_as_index_host_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>( x_dev, spot, num_entries, rows,cols);
      checkCudaErrors(cudaMemcpy(x_host + spot, x_dev,  num_entries *  sizeof(int), cudaMemcpyDeviceToHost));
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    checkCudaErrors(cudaMemcpy(x_dev, x_host + spot,  (rows*cols - spot) *  sizeof(int), cudaMemcpyHostToDevice));
    gpu_set_as_index_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x_dev, spot, rows*cols - spot, rows,cols);
    checkCudaErrors(cudaMemcpy(x_host + spot, x_dev,  (rows*cols - spot) *  sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(x_dev);
  }else{
    int * x_dev;
    checkCudaErrors(cudaMalloc((void**)&x_dev, rows*cols * sizeof(int)));

    checkCudaErrors(cudaMemcpy(x_dev, x_host,  rows*cols *  sizeof(int), cudaMemcpyHostToDevice));
    gpu_set_as_index_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x_dev, 0, rows*cols, rows,cols);
    checkCudaErrors(cudaMemcpy(x_host , x_dev,  rows*cols *  sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(x_dev);
  };

  if(1) LOG("finished call to gpu_set_as_index_host") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(1) LOG("gpu_set_as_index_host run time : "<<readable_time(program_time)<<std::endl);
}

template<>
int gpu_abs_max<int>(const long long int n,  const int* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  //thrust::device_ptr<int> dev_ptr(x);
  int* y = (int*) x;
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(y);
  // copy memory to a new device_vector (which automatically allocates memory)
  thrust::device_vector<int> vec(dev_ptr, dev_ptr + n);
  thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> tuple;
  tuple = thrust::minmax_element(vec.begin(), vec.end());

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3

  int max;

  if(abs(*tuple.first) > abs(*tuple.second)){
    max =  abs(*tuple.first);
  }else{
    max =  abs(*tuple.second);
  };

  // save_device_array_to_file<int>(x, n , "gradient");
  // LOG(INFO) << "max : " <<max ;
  // LOG(INFO) << "Press Enter to continue." ;
  // std::cin.ignore();

  return max;

}

template<>
float gpu_abs_max<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  //thrust::device_ptr<float> dev_ptr(x);
  float* y = (float*) x;
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(y);
  // copy memory to a new device_vector (which automatically allocates memory)
  thrust::device_vector<float> vec(dev_ptr, dev_ptr + n);
  thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> tuple;
  tuple = thrust::minmax_element(vec.begin(), vec.end());

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3

  float max;

  if(abs(*tuple.first) > abs(*tuple.second)){
    max =  abs(*tuple.first);
  }else{
    max =  abs(*tuple.second);
  };

  // save_device_array_to_file<float>(x, n , "gradient");
  // LOG(INFO) << "max : " <<max ;
  // LOG(INFO) << "Press Enter to continue." ;
  // std::cin.ignore();

  return max;

}


template <>
float gpu_min<float>(const long long int n,  const float* x) 
{

  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}


  //thrust::device_ptr<float> dev_ptr(x);
  float* y = (float*) x;
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(y);
  // copy memory to a new device_vector (which automatically allocates memory)
  thrust::device_vector<float> vec(dev_ptr, dev_ptr + n);
  thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple;
  tuple = thrust::minmax_element(vec.begin(), vec.end());

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3

  // save_device_array_to_file<float>(x, n , "gradient");
  // LOG(INFO) << "max : " <<max ;
  // LOG(INFO) << "Press Enter to continue." ;
  // std::cin.ignore();

  return *tuple.first;

}

template <>
float gpu_max<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  //thrust::device_ptr<float> dev_ptr(x);
  float* y = (float*) x;
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(y);
  // copy memory to a new device_vector (which automatically allocates memory)
  thrust::device_vector<float> vec(dev_ptr, dev_ptr + n);
  thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple;
  tuple = thrust::minmax_element(vec.begin(), vec.end());

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3

  return *tuple.second;

}

template <>
float gpu_range<float>(const long long int n,  const float* x) 
{

  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
  //thrust::device_ptr<float> dev_ptr(x);
  float* y = (float*) x;
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(y);
  // copy memory to a new device_vector (which automatically allocates memory)
  thrust::device_vector<float> vec(dev_ptr, dev_ptr + n);
  thrust::pair<thrust::device_vector<float>::iterator,thrust::device_vector<float>::iterator> tuple;
  tuple = thrust::minmax_element(vec.begin(), vec.end());

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3


  // save_device_array_to_file<float>(x, n , "gradient");
  // LOG(INFO) << "max : " <<max ;
  // LOG(INFO) << "Press Enter to continue." ;
  // std::cin.ignore();

  return *tuple.second - *tuple.first;

}

template <>
float gpu_sum<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float sum = thrust::reduce(X, X + n, (float)0., thrust::plus<float>());
  return sum;
}

template <>
float gpu_norm<float>(cublasHandle_t dn_handle, const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1, "Long long long int too big");}
  float s;
  CUBLAS_CHECK(cublasSnrm2(dn_handle, n, x, 1, &s));
  return s;
}

template <>
double gpu_norm<double>(cublasHandle_t dn_handle, const long long int n, const double* x) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1, "Long long long int too big");}
  double s;
  CUBLAS_CHECK(cublasDnrm2(dn_handle, n, x, 1, &s));
  return s;
}

template<typename Dtype> 
Dtype gpu_expected_value(const long long int n,  const Dtype* x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  Dtype* y = (Dtype*) x;
  thrust::device_ptr<Dtype> X = thrust::device_pointer_cast(y);
  Dtype sum = thrust::reduce(X, X + n, (Dtype)0., thrust::plus<Dtype>());
  return (Dtype)(sum/(Dtype)n);
}

template float gpu_expected_value<float>(const long long int n,  const float* x);
template double gpu_expected_value<double>(const long long int n, const double* x);

// square<T> computes the square of a number f(x) -> x*x 
template <typename Dtype> 
struct abss 
{ 
  __host__ __device__ Dtype operator()(const Dtype& x) const {
    if(x >= (Dtype)0.0){
      return x; 
    }else{
      return -x; 
    };
  } 
};

template <>
float gpu_expected_abs_value<float>(const long long int n,  const float* x) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  // setup arguments 
  abss<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s =  thrust::transform_reduce(X, X+n, unary_op, init, binary_op) ; 


  return (float)(s/(float)n);

}

// square<T> computes the square of a number f(x) -> x*x 
template <typename T> 
struct square 
{ 
  __host__ __device__ T operator()(const T& x) const {
   return x * x; 
 } 
};

template <>
float gpu_variance<float>(const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  float expt_ =  gpu_expected_value<float>(n, x);
  // setup arguments 
  square<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s =  thrust::transform_reduce(X, X+n, unary_op, init, binary_op) ; 

  //pow( base, exp )
  return (float)(s/(float)n - pow(expt_ , (float)2.0));
}


template<typename Dtype>
Dtype gpu_expected_dist_two_guassian(cublasHandle_t dn_handle, const long long int n){
  bool Debug = false;

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  long long int m = n;
  if(m % (long long int)2 != (long long int)0){
    m = m + (long long int)1;
  }

  int trials = 500;
  Dtype sum = (Dtype)0.0;
  Dtype* x;
  Dtype* y;
  CUDA_CHECK(cudaMalloc((void**)&x, n * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc((void**)&y, n * sizeof(Dtype)));
  for(int i = 0; i < trials; i++){
    gpu_rng_gaussian<Dtype>(m, (Dtype)0.0, (Dtype)1.0, x);
    gpu_rng_gaussian<Dtype>(m, (Dtype)0.0, (Dtype)1.0, y);

    gpu_axpy<Dtype>(dn_handle, n, (Dtype)(-1.0), x, y);
    if(Debug){
      save_device_arrays_side_by_side_to_file<Dtype>(x, y, n, "x and y-x");
    }

    sum += gpu_norm<Dtype>(dn_handle, n, y);
    if(Debug){
      LOG("sum so far : "<<sum);
      return (Dtype)0.0;
    }
  }
  cudaFree(x);
  cudaFree(y);
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));  
  if(0) LOG("gpu_expected_dist_two_guassian run time : "<<readable_time(program_time)<<std::endl);

  return sum / ((Dtype)trials);
}

template <>
float gpu_sum_of_squares<float>(const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  square<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s = thrust::transform_reduce(X, X+n, unary_op, init, binary_op) ; 
  return s/* /(float)n*/;
}



template <>
void gpu_dot<float>(cublasHandle_t dn_handle, const long long int n, const float* x, const float* y,
  float* out) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasSdot(dn_handle, n, x, 1, y, 1, out));
}

template <>
void gpu_dot<double>(cublasHandle_t dn_handle, const long long int n, const double* x, const double* y,
  double * out) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasDdot(dn_handle, n, x, 1, y, 1, out));
}

template <>
void gpu_asum<float>(cublasHandle_t dn_handle, const long long int n, const float* x, float* y) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasSasum(dn_handle, n, x, 1, y));
}

template <>
void gpu_asum<double>(cublasHandle_t dn_handle, const long long int n, const double* x, double* y) {
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasDasum(dn_handle, n, x, 1, y));
}



template <>
cublasStatus_t transpose<float>(cublasHandle_t handle, 
  long long int m, long long int n, const float *A, float *result)
{
  if(too_big(n) || too_big(m)) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  float one = 1;
  float zero = 0;
  cublasOperation_t trans = CUBLAS_OP_T;

  return cublasSgeam(handle, trans, trans, n, m, &one, A, m, &zero, A, m, result, n);
}

template <>
cublasStatus_t transpose<double>(cublasHandle_t handle,
 long long int m, long long int n, const double *A, double *result)
{
  if(too_big(n) || too_big(m)) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  double one = 1;
  double zero = 0;
  cublasOperation_t trans = CUBLAS_OP_T;
  return cublasDgeam(handle, trans, trans, n, m, &one, A, m, &zero, A, m, result, n);
}

__global__ void transpose_in_place_kernel(const long long int lda, const long long int sda, float *A, 
  const long long int start, const long long int total, 
  bool* isBad)
{
  //WARNING THIS IS WRONG, THERE ARE CYCLES LARGER THAN 2, See CPU version
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //a[idx] = curand_uniform(&state[idx]);
  // CUDA_KERNEL_LOOP(i, total){
  //   long long int j = ((long long int)i + start);


  //   long long int row = j % lda;
  //   long long int col = j / lda;
  //       //j = row + lda * col;
  //       long long int new_j = col + sda * row; // this is the new spot where A[j] belongs
  //       if (new_j < j){
  //         float temp = A[j];
  //         A[j] = A[new_j];
  //         A[new_j] = temp;
  //       }

  //       //does A[new_j] belong in the j^th spot?
  //       long long int row_new = new_j % lda;
  //       long long int col_new = new_j / lda;
  //       //new_i = row_was + lda * col_was;
  //       if (col_new + lda * row_new != j){
  //         isBad[0] = true;
  //       };
  //     }
}

template <>
void transpose_in_place<float>(cublasHandle_t handle, const long long int lda, const long long int sda, float *A)
{
  bool Debug = false;
  /*
    long long int num_gpu_blocks= GET_BLOCKS(lda * sda);

    ABORT_IF_NEQ(lda, sda, "Transpose in place for r!= c is more sofisticated than what is currently written"); 

    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));
    

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      long long int num_loops = 0;
      long long int num_entries = CUDA_NUM_BLOCKS*CUDA_NUM_THREADS;
      // LOG(INFO) << "num_entries: "<<num_entries;
      long long int spot = 0;
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
        transpose_in_place_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(lda, sda, A, spot, num_entries);

        num_gpu_blocks= num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
        num_loops+=(long long int)1;
        spot = num_loops * num_entries;
        if(Debug) LOG("starting index :"<<spot);
        if(Debug) LOG("total entries : "<<spot);
        if(Debug) LOG("next start : "<<spot + num_entries);



      };
      // spot is the number of entries done so far
      // total - (done) = left to go
      transpose_in_place_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(lda, sda, A, spot, lda * sda - spot);
    }else{
      transpose_in_place_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(lda, sda, A, 0, lda * sda);
    };

    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if ( isBad_host == true){
      ABORT_IF_NEQ(0, 1, "Tranpose_in_place broke") 
    };
  */
  float* result;
  CUDA_CHECK(cudaMalloc((void**)&result, lda * sda * sizeof(float)));

  transpose<float>(handle,lda, sda, A, result);

  CUDA_CHECK(cudaMemcpy(A, result, lda * sda * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaFree(result);

}

__global__ void swap_ordering_kernel(const long long int rows, const long long int cols, const float *A, 
 const bool row_major_ordering, float * result,
 const long long int start, const int total)
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //a[idx] = curand_uniform(&state[idx]);
  CUDA_KERNEL_LOOP(i, total){

    long long int j = ((long long int)i + start);

    if(!row_major_ordering){
        //starts in colum major ordering
      long long int row = j % rows;
      long long int col = j / rows;
        //i = row + rows * col;
      long long int new_i = cols * row + col;
      result[new_i] = A[j];

    }else{
        //starts in row major ordering
      long long int row = j / cols;
      long long int col = j % cols;
        //i = cols * row + col;
      long long int new_i = row + rows * col;
      result[new_i] = A[j];

    }
  }
}

template <>
void swap_ordering<float>(const long long int rows, const long long int cols, float *A, const bool row_major_ordering)
{
  bool Debug = false;
  long long int num_gpu_blocks= GET_BLOCKS(rows * cols);
  float* result;
  CUDA_CHECK(cudaMalloc((void**)&result, rows * cols * sizeof(float)));

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = 0;
    long long int num_entries = CUDA_NUM_BLOCKS*CUDA_NUM_THREADS;
    // LOG(INFO) << "num_entries: "<<num_entries;
    long long int spot = 0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      swap_ordering_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, spot, num_entries);

      num_gpu_blocks= num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops+=(long long int)1;
      spot = num_loops * num_entries;
      if(Debug) LOG("starting index :"<<spot);
      if(Debug) LOG("total entries : "<<spot);
      if(Debug) LOG("next start : "<<spot + num_entries);

    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    swap_ordering_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, spot, rows * cols - spot - 1);
  }else{
    swap_ordering_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, 0, rows * cols);
  };

  CUDA_CHECK(cudaMemcpy(A, result, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaFree(result);
  
}

template <>
void gpu_gemv<float>(cublasHandle_t dn_handle, const bool TransA, const int M,
 const int N, const float alpha, const float* A, const float* x, const int inc_x,
 const float beta, float* y, const int inc_y) 
{
  cublasOperation_t cuTransA = (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;

  CUBLAS_CHECK(cublasSgemv(dn_handle, cuTransA, N, M, &alpha, A, N, x, inc_x, &beta, y, inc_y));


  /*
    cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t cuTransA, 
                               int N, int M, const float *alpha, const float *A, 
                               int lda, const float *x, int incx, const float *beta, 
                               float *y, int incy)

    performs y = α op ( A ) x + β y

    where A is a N × M matrix stored in column-major format, 
    x and y are vectors, and α and β are scalars. Also, for matrix A

     op ( A ) = A  if cuTransA == CUBLAS_OP_N 
              = A^T  if cuTransA == CUBLAS_OP_T

    N - number of rows of matrix A.
    M - number of columns of matrix A.
    lda - leading dimension of two-dimensional array used to store matrix A. lda must be at least max(1,N).
    incx - stride between consecutive elements of x.
    x - vector at least (1+(M-1)*abs(incx)) elements if cuTransA==CUBLAS_OP_N and at least (1+(N-1)*abs(incx)) elements otherwise.

  */
}


template <>
void gpu_gemv<double>(cublasHandle_t dn_handle, const bool TransA, const int M,
  const int N, const double alpha, const double* A, const double* x, const int inc_x,
  const double beta, double* y, const int inc_y) 
{
  cublasOperation_t cuTransA =
  (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(dn_handle, cuTransA, N, M, &alpha,
    A, N, x, inc_x, &beta, y, inc_y));
}

template<typename Dtype>
__global__ void gpu_hadamard_kernel(const int n, const Dtype* A, Dtype* B) 
{
  //performs B = A * B
  CUDA_KERNEL_LOOP(i, n) {
  //origin+x1+rows*y1
    B[i]=A[i]*B[i];
    if(abs(B[i]) == (Dtype)0.0) B[i] = (Dtype)0.0;

  };
}

template<>
void gpu_hadamard<float>(const long long int n, const float* A, float* B ) 
{
  //performs B = A * B
    long long int num_gpu_blocks = GET_BLOCKS(n);
  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //   ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };
  gpu_hadamard_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(n, A, B);

}

template <>
void gpu_axpy<float>(cublasHandle_t dn_handle, const long long int N, const float alpha, const float* X,
 float* Y) 
{
  if(too_big(N)) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasSaxpy(dn_handle, N, &alpha, X, 1, Y, 1));
  /*
     cublasSaxpy multiplies the vector x by the scalar α and adds it
     to the vector y overwriting the latest vector with the result. 
     Hence, the performed operation is y [ j ] = α × x [ k ] + y [ j ] 
     for i = 1 , … , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy . 
     Notice that the last two equations reflect 1-based indexing used 
     for compatibility with Fortran.
  */
}

template <>
void gpu_axpy<double>(cublasHandle_t dn_handle, const long long int N, const double alpha, const double* X,
double* Y) 
{
  if(too_big(N)) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasDaxpy(dn_handle, N, &alpha, X, 1, Y, 1));
  /*
   cublasSaxpy multiplies the vector x by the scalar α and adds it
   to the vector y overwriting the latest vector with the result. 
   Hence, the performed operation is y [ j ] = α × x [ k ] + y [ j ] 
   for i = 1 , … , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy . 
   Notice that the last two equations reflect 1-based indexing used 
   for compatibility with Fortran.
  */
 }


template <typename Dtype>
void gpu_axpby(cublasHandle_t dn_handle, const long long int N, const Dtype alpha, const Dtype* X,
const Dtype beta, Dtype* Y) 
{
  gpu_scale<Dtype>(dn_handle, N, beta, Y);
  gpu_axpy<Dtype>(dn_handle, N, alpha, X, Y);
  /*
     cublasSaxpy multiplies the vector x by the scalar α and adds it
     to the vector beta * y overwriting the latest vector with the result. 
     Hence, the performed operation is y [ j ] = α × x [ k ] + beta * y [ j ] 
     for i = 1 , … , n , k = 1 + ( i - 1 ) *  incx and j = 1 + ( i - 1 ) *  incy . 
     Notice that the last two equations reflect 1-based indexing used 
     for compatibility with Fortran.
  */
 }

template void gpu_axpby<float>(cublasHandle_t dn_handle,const long long int N, const float alpha, const float* X,
  const float beta, float* Y);
template void gpu_axpby<double>(cublasHandle_t dn_handle,const long long int N, const double alpha, const double* X,
  const double beta, double* Y);

template<>
float gpu_sum_of_squares_of_diff<float>(cublasHandle_t dn_handle, const long long int n, const float* x, float* y)
{
  gpu_axpby<float>(dn_handle, n, (float)(1.0), x, (float)(-1.0), y);

  square<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float* r = (float*) y;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(r);
  float s = thrust::transform_reduce(X, X+n, unary_op, init, binary_op) ; 
  return s/* /(float)n*/;
}

template <>
cublasStatus_t cublasXgemm<float>(cublasHandle_t handle,
 cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
 float *alpha, const float *A, int lda,
 float *B, int ldb, float *beta,
 float *C, int ldc)
{
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasXgemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
 int m, int n, int k,
 double *alpha, const double *A, int lda,
 double *B, int ldb, double *beta,
 double *C, int ldc)
{
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
void gpu_gemm<float>(cublasHandle_t dn_handle, const bool TransA,
 const bool TransB, const int M, const int N, const int K,
 const float alpha, const float* A, const float* B, const float beta,
 float* C) 
{
  // Note that cublas follows fortran order.
  // column-major ordering marches down a complete column
  // before moving to the next column
  // thus the leading dimension is the number of ROWS 
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
  (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
  (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(dn_handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
  //N is number of rows of matrix op(B) and C.
  //M number of columns of matrix op(A) and C.
  //K is number of columns of op(B) and rows of op(A).
  //
  // op(B) is N by K
  // op(A) is K by M
  // C is N by M

  // performs C=alpha op ( B ) op ( A ) + beta C
}

template <>
void gpu_gemm<double>(cublasHandle_t dn_handle, const bool TransA,
  const bool TransB, const int M, const int N, const int K,
  const double alpha, const double* A, const double* B, const double beta,
  double* C) 
{
  // Note that cublas follows fortran order.
  // column-major ordering marches down a complete column
  // before moving to the next column
  // thus the leading dimension is the number of ROWS 
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
  (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
  (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(dn_handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  //N is number of rows of matrix op(B) and C.
  //M number of columns of matrix op(A) and C.
  //K is number of columns of op(B) and rows of op(A).

  // op(B) is N by K
  // op(A) is K by M
  // C is N by M
  // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C
}


template <>
void gpu_noisey_gemm<float>(cublasHandle_t dn_handle, const bool TransA,
 const bool TransB, const int M, const int N, const int K,
 const float alpha, const float* A, const float range, const float beta,
 float* C) 
{

  float* B;
  checkCudaErrors(cudaMalloc((void**)&B, N * K * sizeof(float)));
  gpu_rng_uniform<float>(dn_handle, N * K, (float)(-1.0) * range, range, B);
  // Note that cublas follows fortran order.
  // column-major ordering marches down a complete column
  // before moving to the next column
  // thus the leading dimension is the number of ROWS 
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
  (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
  (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(dn_handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
  //N is number of rows of matrix op(B) and C.
  //M number of columns of matrix op(A) and C.
  //K is number of columns of op(B) and rows of op(A).
  //
  // op(B) is N by K
  // op(A) is K by M
  // C is N by M

  // performs C=alpha op ( B ) op ( A ) + beta C

  cudaFree(B);
}



template <>
cublasStatus_t cublasXgemmBatched<float>(cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
  float *alpha, const float *Aarray[], int lda,
  const float *Barray[], int ldb, float *beta,
  float *Carray[], int ldc, int batchCount)
{
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
template <>
cublasStatus_t cublasXgemmBatched<double>(cublasHandle_t handle,
  cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
  double *alpha, const double *Aarray[], int lda,
  const double *Barray[], int ldb, double *beta,
  double *Carray[], int ldc,
  int batchCount)
{

  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);

}









template <>
void cublasXSparsegemm<float>(cusparseHandle_t handle, bool TransA, bool TransB, 
  int m, int n, int k, int nnz, float *alpha, const cusparseMatDescr_t descrA, 
  const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
  float *B, int ldb, float *beta, float *C, int ldc)
{
  /*
    This function performs one of the following matrix-matrix operations:

    C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

    A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

    op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE A T if transA == CUSPARSE_OPERATION_TRANSPOSE A H if transA == CUSPARSE_OPERATION_CONJUCUTE_TRANSPOSE
    and

    op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE B T if transB == CUSPARSE_OPERATION_TRANSPOSE B H not supported
  */
    cusparseOperation_t cuTransA =
    (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    cusparseOperation_t cuTransB =
    (TransB == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    CUSPARSE_CHECK( cusparseScsrmm2(handle, cuTransA,cuTransB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));

  //cublasSgemmBatched(handle, cuTransA, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
  }

template <>
void cublasXSparsegemm<double>(cusparseHandle_t handle, bool TransA, bool TransB,
  int m, int n, int k, int nnz, double *alpha, const cusparseMatDescr_t descrA, 
  const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
  double *B, int ldb, double *beta, double *C, int ldc)
{
  cusparseOperation_t cuTransA =
  (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
  (TransB == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseDcsrmm2(handle, cuTransA, cuTransB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc));

}


template <>
void cublasCSCSparseXgemm<float>(cusparseHandle_t handle,int m, int n, int k, int nnz, float *alpha, const float *A, int lda,
 const float *cscValB, const int *cscColPtrB, const int *cscRowIndB, const float *beta, 
 float *C, int ldc)
{
  /*
    This function performs the following matrix-matrix operations:

    C = α ∗ A ∗ B + β ∗ C
    A and C are dense matrices; 
    B is a k×n sparse matrix that is defined in CSC storage format by the three arrays cscValB, cscColPtrB, and cscRowIndB); α  and  β are scalars; and

    Remark: B is base-0.
  */

  CUSPARSE_CHECK(cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc));
}


//============================================================================================
// me-made math
//============================================================================================





  struct test 
  { 
    bool *testing_bools;
    test(bool *t_b) : testing_bools(t_b) {}

    __host__ __device__ 
    bool operator()(const int& x) const {
      return testing_bools[x];
    } 
  };


  int count_testing_data(const int nnz_data,  const int* coo_data, bool *testing_bools) 
  {
    int* y = (int*) coo_data;
    thrust::device_ptr<int> X = thrust::device_pointer_cast(y);
    return thrust::count_if(X, X+nnz_data, test(testing_bools));

  }

  struct in_group 
  { 
    const int *group_indicies;
    const int group;
    in_group(const int *g_i, const int g) : group_indicies(g_i), group(g) {}

    __host__ __device__ 
    bool operator()(const int& x) const {
      return group_indicies[x] == group;
    } 
  };


  void count_each_group_from_coo(const int num_groups, const int* group_indicies, const int nnz_data,  const int* coo_data, int* group_sizes) 
  {
    int* y = (int*) coo_data;
    thrust::device_ptr<int> X = thrust::device_pointer_cast(y);
    for(int group = 0; group < num_groups; group++){
      group_sizes[group] = thrust::count_if(X, X+nnz_data, in_group(group_indicies, group));
    }


  }


  struct is_true 
  { 
    __host__ __device__ 
    bool operator()(const bool& x) const {
      return x;
    } 
  };


  int count_nnz_bools(const int n,  const bool *testing_bools) 
  {
    bool* y = (bool*) testing_bools;
    thrust::device_ptr<bool> X = thrust::device_pointer_cast(y);
    return thrust::count_if(X, X+n, is_true());

  }

  struct is_true_ 
  { 
    const int group;
    is_true_(const int g) : group(g) {}

    __host__ __device__ 
    bool operator()(const int& x) const {
      return x == group;
    } 
  };


  void count_each_group(const int n,  const int* group_indicies, int* group_sizes, const int num_groups ) 
  {
    int* y = (int*) group_indicies;
    thrust::device_ptr<int> X = thrust::device_pointer_cast(y);
    for(int group = 0; group < num_groups; group++){
      group_sizes[group] = thrust::count_if(X, X+n, is_true_(group));
    }

  }




  __global__ void gpu_split_data_kernel(const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev, 
    const int ratings_rows, const bool *testing_bools,
    int* csr_format_ratingsMtx_userID_dev_training,
    int* coo_format_ratingsMtx_itemID_dev_training,
    float* coo_format_ratingsMtx_rating_dev_training,
    const int ratings_rows_training,
    int* csr_format_ratingsMtx_userID_dev_testing,
    int* coo_format_ratingsMtx_itemID_dev_testing,
    float* coo_format_ratingsMtx_rating_dev_testing,
    const int ratings_rows_testing) 
  {

    CUDA_KERNEL_LOOP(row, ratings_rows){

      int r_ = 0;
      int csr_start = 0;
      int num_user = 0;
      while(r_ < row){
        if(testing_bools[r_] == testing_bools[row]){
          csr_start += csr_format_ratingsMtx_userID_dev[r_ + 1] - csr_format_ratingsMtx_userID_dev[r_];
          num_user  += 1;
        }
        r_++;
      }
      int csr_end = csr_start + csr_format_ratingsMtx_userID_dev[row + 1] - csr_format_ratingsMtx_userID_dev[row];
      int* csr_format_ratingsMtx_userID;
      int* coo_format_ratingsMtx_itemID;
      float* coo_format_ratingsMtx_rating;

      if(testing_bools[row]){
        csr_format_ratingsMtx_userID = csr_format_ratingsMtx_userID_dev_testing;
        coo_format_ratingsMtx_itemID = coo_format_ratingsMtx_itemID_dev_testing;
        coo_format_ratingsMtx_rating = coo_format_ratingsMtx_rating_dev_testing;
        if(num_user == ratings_rows_testing - 1) csr_format_ratingsMtx_userID[num_user + 1] = csr_end;
      }else{
        csr_format_ratingsMtx_userID = csr_format_ratingsMtx_userID_dev_training;
        coo_format_ratingsMtx_itemID = coo_format_ratingsMtx_itemID_dev_training;
        coo_format_ratingsMtx_rating = coo_format_ratingsMtx_rating_dev_training;
        if(num_user == ratings_rows_training - 1) csr_format_ratingsMtx_userID[num_user + 1] = csr_end;
      };
      csr_format_ratingsMtx_userID[num_user] = csr_start;

      int i = csr_start;
      for(r_ = csr_format_ratingsMtx_userID_dev[row]; r_ < csr_format_ratingsMtx_userID_dev[row + 1]; r_++){
        coo_format_ratingsMtx_itemID[i] = coo_format_ratingsMtx_itemID_dev[r_];
        coo_format_ratingsMtx_rating[i]  = coo_format_ratingsMtx_rating_dev[r_]; 
        i++;
      }

    }
  }





  void gpu_split_data(const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev, 
    const int ratings_rows, const bool *testing_bools,
    int* csr_format_ratingsMtx_userID_dev_training,
    int* coo_format_ratingsMtx_itemID_dev_training,
    float* coo_format_ratingsMtx_rating_dev_training,
    const int ratings_rows_training,
    int* csr_format_ratingsMtx_userID_dev_testing,
    int* coo_format_ratingsMtx_itemID_dev_testing,
    float* coo_format_ratingsMtx_rating_dev_testing,
    const int ratings_rows_testing) 
  {
    struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
    std::cout<<"gpu_split_data called..."<<std::endl;

    const long long int num_gpu_blocks= GET_BLOCKS(ratings_rows);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
    };

    gpu_split_data_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev, 
      ratings_rows, testing_bools,
      csr_format_ratingsMtx_userID_dev_training,
      coo_format_ratingsMtx_itemID_dev_training,
      coo_format_ratingsMtx_rating_dev_training,
      ratings_rows_training,
      csr_format_ratingsMtx_userID_dev_testing,
      coo_format_ratingsMtx_itemID_dev_testing,
      coo_format_ratingsMtx_rating_dev_testing,
      ratings_rows_testing);
    cudaDeviceSynchronize();
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("getData runtime: %f\n", program_time); 
    std::cout<<"finished call to gpu_split_data in "<<program_time<< "ms"<<std::endl<<std::endl;
  //std::cout<<std::endl;

  }


  __global__ void gpu_split_data_kernel(const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev, 
    const int ratings_rows, const int *group_indicies,
    int** csr_format_ratingsMtx_userID_dev_by_group,
    int** coo_format_ratingsMtx_itemID_dev_by_group,
    float** coo_format_ratingsMtx_rating_dev_by_group,
    const int* ratings_rows_by_group) 
  {

    CUDA_KERNEL_LOOP(row, ratings_rows){

      int r_ = 0;
      int csr_start = 0;
      int num_user = 0;
      int group = group_indicies[row];
      while(r_ < row){
        if(group_indicies[r_] == group){
          csr_start += csr_format_ratingsMtx_userID_dev[r_ + 1] - csr_format_ratingsMtx_userID_dev[r_];
          num_user  += 1;
        }
        r_++;
      }
      int csr_end = csr_start + csr_format_ratingsMtx_userID_dev[row + 1] - csr_format_ratingsMtx_userID_dev[row];
      int* csr_format_ratingsMtx_userID;
      int* coo_format_ratingsMtx_itemID;
      float* coo_format_ratingsMtx_rating;
      int ratings_rows_ = ratings_rows_by_group[group];
      csr_format_ratingsMtx_userID = csr_format_ratingsMtx_userID_dev_by_group[group];
      coo_format_ratingsMtx_itemID = coo_format_ratingsMtx_itemID_dev_by_group[group];
      coo_format_ratingsMtx_rating = coo_format_ratingsMtx_rating_dev_by_group[group];
      if(num_user == ratings_rows_ - 1) csr_format_ratingsMtx_userID[num_user + 1] = csr_end;
      csr_format_ratingsMtx_userID[num_user] = csr_start;

      int i = csr_start;
      for(r_ = csr_format_ratingsMtx_userID_dev[row]; r_ < csr_format_ratingsMtx_userID_dev[row + 1]; r_++){
        coo_format_ratingsMtx_itemID[i] = coo_format_ratingsMtx_itemID_dev[r_];
        coo_format_ratingsMtx_rating[i]  = coo_format_ratingsMtx_rating_dev[r_]; 
        i++;
      }

    }
  }





  void gpu_split_data(const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev, 
    const int ratings_rows, const int *group_indicies,
    int** csr_format_ratingsMtx_userID_dev_by_group,
    int** coo_format_ratingsMtx_itemID_dev_by_group,
    float** coo_format_ratingsMtx_rating_dev_by_group,
    const int* ratings_rows_by_group) 
  {
    struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
    LOG("gpu_split_data called...");

    const long long int num_gpu_blocks= GET_BLOCKS(ratings_rows);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
    };

    gpu_split_data_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev, 
      ratings_rows, group_indicies,
      csr_format_ratingsMtx_userID_dev_by_group,
      coo_format_ratingsMtx_itemID_dev_by_group,
      coo_format_ratingsMtx_rating_dev_by_group,
      ratings_rows_by_group);
    cudaDeviceSynchronize();
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("getData runtime: %f\n", program_time); 
    LOG("finished call to gpu_split_data in "<<program_time<< "ms");
  //std::cout<<std::endl;

  }

  __global__ void collect_user_means_kernel(float* user_means_training,float* user_var_training, const long long int ratings_rows_training,
                                          const int* csr_format_ratingsMtx_userID_dev_training,
                                          const float* coo_format_ratingsMtx_rating_dev_training,
                                          float* user_means_testing,float* user_var_testing, const long long int ratings_rows_testing,
                                          const int* csr_format_ratingsMtx_userID_dev_testing,
                                          const float* coo_format_ratingsMtx_rating_dev_testing,
                                          float* user_means_GA,float* user_var_GA, const long long int ratings_rows_GA,
                                          const int* csr_format_ratingsMtx_userID_dev_GA,
                                          const float* coo_format_ratingsMtx_rating_dev_GA,
                                          bool* isBad)
{

    CUDA_KERNEL_LOOP(row, ratings_rows_testing + ratings_rows_training + ratings_rows_GA){
        float m = 0;
        float v = 0;
        int count = 0;
        if(row < ratings_rows_training){
            for(long long int i = csr_format_ratingsMtx_userID_dev_training[row]; i < csr_format_ratingsMtx_userID_dev_training[row + 1]; i++){
                m += coo_format_ratingsMtx_rating_dev_training[i];
                v += pow(coo_format_ratingsMtx_rating_dev_training[i],(float)2.0); 
                count++;
            }
            user_means_training[row] = m / (float)count;
            user_var_training[row] = v / (float)count - pow(user_means_training[row], (float)2.0);
            if(user_var_training[row] <= (float)0.0){
              user_var_training[row] = (float)0.0;
              //isBad[0] = true;
            }
        }else{
            if(row < ratings_rows_training + ratings_rows_GA){
                long long int r = row - ratings_rows_training;
                for(long long int i = csr_format_ratingsMtx_userID_dev_GA[r]; i < csr_format_ratingsMtx_userID_dev_GA[r + 1]; i++){
                    m += coo_format_ratingsMtx_rating_dev_GA[i];
                    v += pow(coo_format_ratingsMtx_rating_dev_GA[i],(float)2.0); 
                    count++;
                }
                user_means_GA[r] = m / (float)count;
                user_var_GA[r] = v / (float)count - pow(user_means_GA[r], (float)2.0);
                if(user_var_GA[r] <= (float)0.0) {
                  user_var_GA[r] = (float)0.0;
                  //isBad[0] = true;
                }
            }else{
                long long int r = row - ratings_rows_training - ratings_rows_GA;
                for(long long int i = csr_format_ratingsMtx_userID_dev_testing[r]; i < csr_format_ratingsMtx_userID_dev_testing[r + 1]; i++){
                    m += coo_format_ratingsMtx_rating_dev_testing[i];
                    v += pow(coo_format_ratingsMtx_rating_dev_testing[i],(float)2.0); 
                    count++;
                }
                user_means_testing[r] = m / (float)count; 
                user_var_testing[r] = v / (float)count - pow(user_means_testing[r], (float)2.0);  
                if(user_var_testing[r] <= (float)0.0) {
                  user_var_testing[r] = (float)0.0; 
                  //isBad[0] = true; 
                }           
            }

        }

        
    }
}


void collect_user_means(float* user_means_training,float* user_var_training, const long long int ratings_rows_training,
                        const int* csr_format_ratingsMtx_userID_dev_training,
                        const float* coo_format_ratingsMtx_rating_dev_training,
                        float* user_means_testing,float* user_var_testing, const long long int ratings_rows_testing,
                        const int* csr_format_ratingsMtx_userID_dev_testing,
                        const float* coo_format_ratingsMtx_rating_dev_testing,
                        float* user_means_GA,float* user_var_GA, const long long int ratings_rows_GA,
                        const int* csr_format_ratingsMtx_userID_dev_GA,
                        const float* coo_format_ratingsMtx_rating_dev_GA)
{
  bool Debug = false;

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  const int num_gpu_blocks= GET_BLOCKS(ratings_rows_testing + ratings_rows_training + ratings_rows_GA);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  collect_user_means_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(user_means_training, user_var_training, ratings_rows_training,
                                                                  csr_format_ratingsMtx_userID_dev_training,
                                                                  coo_format_ratingsMtx_rating_dev_training,
                                                                  user_means_testing, user_var_testing, ratings_rows_testing,
                                                                  csr_format_ratingsMtx_userID_dev_testing,
                                                                  coo_format_ratingsMtx_rating_dev_testing,
                                                                  user_means_GA, user_var_GA, ratings_rows_GA,
                                                                  csr_format_ratingsMtx_userID_dev_GA,
                                                                  coo_format_ratingsMtx_rating_dev_GA,
                                                                  isBad);

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    float ratings_rows_training_min = gpu_min<float>(ratings_rows_training, user_var_training);
    float ratings_rows_testing_min  = gpu_min<float>(ratings_rows_testing,  user_var_testing) ;
    float ratings_rows_GA_min       = gpu_min<float>(ratings_rows_GA,       user_var_GA) ;
    LOG("ratings_rows_training_min :" <<ratings_rows_training_min);
    LOG("ratings_rows_testing_min :"  <<ratings_rows_testing_min);
    LOG("ratings_rows_GA_min :"       <<ratings_rows_GA_min);

    float ratings_rows_training_max = gpu_abs_max<float>(ratings_rows_training, user_var_training);
    float ratings_rows_testing_max  = gpu_abs_max<float>(ratings_rows_testing,  user_var_testing) ;
    float ratings_rows_GA_max       = gpu_abs_max<float>(ratings_rows_GA,       user_var_GA) ;
    LOG("ratings_rows_training_max :" <<ratings_rows_training_max);
    LOG("ratings_rows_testing_max :"  <<ratings_rows_testing_max);
    LOG("ratings_rows_GA_max :"       <<ratings_rows_GA_max);


    save_device_array_to_file<float>(user_var_training, ratings_rows_training, "user_var_training");
    save_device_array_to_file<float>(user_var_testing, ratings_rows_testing, "user_var_testing");
    save_device_array_to_file<float>(user_var_GA, ratings_rows_GA, "user_var_GA");
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}


  __global__ void collect_user_means_kernel(float* user_means,float* user_var, const long long int ratings_rows,
    const int* csr_format_ratingsMtx_userID_dev,
    const float* coo_format_ratingsMtx_rating_dev,
    bool* isBad)
  {

    CUDA_KERNEL_LOOP(row, ratings_rows){
      float m = 0;
      float v = 0;
      int count = 0;
      for(long long int i = csr_format_ratingsMtx_userID_dev[row]; i < csr_format_ratingsMtx_userID_dev[row + 1]; i++){
        m += coo_format_ratingsMtx_rating_dev[i];
        v += pow(coo_format_ratingsMtx_rating_dev[i],(float)2.0); 
        count++;
      }
      user_means[row] = m / (float)count;
      user_var[row] = v / (float)count - pow(user_means[row], (float)2.0);
      if(user_var[row] <= (float)0.0){
        user_var[row] = (float)0.0;
    //isBad[0] = true;
      }
    }
  }


  void collect_user_means(float* user_means,float* user_var, const long long int ratings_rows,
    const int* csr_format_ratingsMtx_userID_dev,
    const float* coo_format_ratingsMtx_rating_dev)
  {
    bool Debug = false;

    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

    const long long int num_gpu_blocks= GET_BLOCKS(ratings_rows);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
    };

    collect_user_means_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(user_means, user_var, ratings_rows,
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_rating_dev,
      isBad);

    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if(isBad_host){
      float ratings_rows_min = gpu_min<float>(ratings_rows, user_var);
      LOG("ratings_rows_min :" <<ratings_rows_min);

      float ratings_rows_max = gpu_abs_max<float>(ratings_rows, user_var);
      LOG("ratings_rows_max :" <<ratings_rows_max);

      save_device_array_to_file<float>(user_var, ratings_rows, "user_var");
      ABORT_IF_NEQ(0, 1, "isBad");
    };

  }






  __global__ void fill_training_mtx_kernel(const long long int ratings_rows_training, 
    const long long int ratings_cols_training, const bool row_major_ordering, 
    const int* csr_format_ratingsMtx_userID_dev_training,
    const int* coo_format_ratingsMtx_itemID_dev_training,
    const float* coo_format_ratingsMtx_rating_dev_training,
    float* full_training_ratings_mtx, long long int start, int num,
    bool* isBad)
  {

    CUDA_KERNEL_LOOP(j, num){
      long long int row = (long long int)j + start;
      for(long long int i = csr_format_ratingsMtx_userID_dev_training[row]; i < csr_format_ratingsMtx_userID_dev_training[row + (long long int)1]; i++){
        long long int col = coo_format_ratingsMtx_itemID_dev_training[i];
        float val = coo_format_ratingsMtx_rating_dev_training[i]; 
        if (::isinf(val) || ::isnan(val)){
          isBad[0] = true;
        };
        if(row_major_ordering){
          full_training_ratings_mtx[ratings_cols_training * row + col] = val;
        }else{
          full_training_ratings_mtx[row + ratings_rows_training * col] = val;
        };
      }
    }
  }


  void gpu_fill_training_mtx(const long long int ratings_rows_training, 
    const long long int ratings_cols_training, 
    const bool row_major_ordering,
    const int* csr_format_ratingsMtx_userID_dev_training,
    const int* coo_format_ratingsMtx_itemID_dev_training,
    const float* coo_format_ratingsMtx_rating_dev_training,
    float* full_training_ratings_mtx)
  {
    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

    long long int num_gpu_blocks= GET_BLOCKS(ratings_rows_training);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      long long int num_loops   = 0;
      long long int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
      long long int spot        = 0;
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      fill_training_mtx_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(ratings_rows_training,ratings_cols_training, row_major_ordering, 
        csr_format_ratingsMtx_userID_dev_training,
        coo_format_ratingsMtx_itemID_dev_training,
        coo_format_ratingsMtx_rating_dev_training,
        full_training_ratings_mtx, spot, num_entries,
        isBad);
        num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS;
        num_loops     += (long long int)1;
        spot           = num_loops * num_entries;
      };
      fill_training_mtx_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training,ratings_cols_training, row_major_ordering, 
        csr_format_ratingsMtx_userID_dev_training,
        coo_format_ratingsMtx_itemID_dev_training,
        coo_format_ratingsMtx_rating_dev_training,
        full_training_ratings_mtx, spot, ratings_rows_training - spot,
        isBad);
    }else{
      fill_training_mtx_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training,ratings_cols_training, row_major_ordering, 
        csr_format_ratingsMtx_userID_dev_training,
        coo_format_ratingsMtx_itemID_dev_training,
        coo_format_ratingsMtx_rating_dev_training,
        full_training_ratings_mtx, (long long int)0, ratings_rows_training,
        isBad);
    };



    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };

  }



__global__ void gpu_supplement_training_mtx_with_content_based_kernel(const long long int ratings_rows_training, 
                                                                      const long long int ratings_cols_training, 
                                                                      const bool row_major_ordering,
                                                                      const int* csr_format_ratingsMtx_userID_dev_training,
                                                                      const int* coo_format_ratingsMtx_itemID_dev_training,
                                                                      const float* coo_format_ratingsMtx_rating_dev_training,
                                                                      float* full_training_ratings_mtx,
                                                                      const int* csr_format_keyWordMtx_itemID_dev,
                                                                      const int* coo_format_keyWordMtx_keyWord_dev,
                                                                      bool* isBad, const long long int start, const long long int how_many)
{

  CUDA_KERNEL_LOOP(b, how_many){
    long long int mtx_index = (long long int)b + start;
    int user = 0;
    int item = 0;
    if(row_major_ordering){
      user = mtx_index / ratings_cols_training;
      item = mtx_index % ratings_cols_training;
    }else{
      user = mtx_index % ratings_rows_training;
      item = mtx_index / ratings_rows_training;
    }

    bool could_estimate = true;
    for(int coo_index = csr_format_ratingsMtx_userID_dev_training[user]; coo_index < csr_format_ratingsMtx_userID_dev_training[user + 1]; coo_index++){
      if(coo_format_ratingsMtx_itemID_dev_training[coo_index] == item) could_estimate = false;
      if(coo_format_ratingsMtx_itemID_dev_training[coo_index]  > item) break;
    }
    if (could_estimate){
      int   count  = 0;
      float rating = (float)0.0;
      for(int j = csr_format_ratingsMtx_userID_dev_training[user]; j < csr_format_ratingsMtx_userID_dev_training[user + 1]; j++){
        int other_item = coo_format_ratingsMtx_itemID_dev_training[j];
        for(int k = csr_format_keyWordMtx_itemID_dev[other_item]; k < csr_format_keyWordMtx_itemID_dev[other_item + 1]; k++){
          int other_keyWord = coo_format_keyWordMtx_keyWord_dev[k];
          int start_l = csr_format_keyWordMtx_itemID_dev[item];
          for(int l = start_l; l < csr_format_keyWordMtx_itemID_dev[item + 1]; l++){
            int keyword = coo_format_keyWordMtx_keyWord_dev[l];
            if(keyword == other_keyWord){
              count += 1;
              //rating += coo_format_ratingsMtx_rating_dev_training[j];
              if(row_major_ordering){
                rating += full_training_ratings_mtx[(long long int)user * ratings_cols_training + (long long int)other_item];
              }else{
                rating += full_training_ratings_mtx[(long long int)user + (long long int)other_item * ratings_rows_training];
              }
              start_l = l+1;
            } else if (keyword > other_keyWord){
              start_l = l;
              break;
            }
          }
        }
      }
      if (count > 0){
        full_training_ratings_mtx[mtx_index] = rating / (float)count;
      }
    }
    if (::isinf(full_training_ratings_mtx[mtx_index]) || ::isnan(full_training_ratings_mtx[mtx_index])){
      isBad[0] = true;
    };
      
  }
}


void gpu_supplement_training_mtx_with_content_based(const long long int ratings_rows_training, 
  const long long int ratings_cols_training, 
  const bool row_major_ordering,
  const int* csr_format_ratingsMtx_userID_dev_training,
  const int* coo_format_ratingsMtx_itemID_dev_training,
  const float* coo_format_ratingsMtx_rating_dev_training,
  float* full_training_ratings_mtx,
  const int* csr_format_keyWordMtx_itemID_dev,
  const int* coo_format_keyWordMtx_keyWord_dev)
{
    bool Debug = false;
    struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);

    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

    long long int num_gpu_blocks= GET_BLOCKS(ratings_rows_training * ratings_cols_training);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      long long int num_loops   = 0;
      long long int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
      long long int spot        = 0;
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
        gpu_supplement_training_mtx_with_content_based_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(ratings_rows_training, 
         ratings_cols_training, 
         row_major_ordering,
         csr_format_ratingsMtx_userID_dev_training,
         coo_format_ratingsMtx_itemID_dev_training,
         coo_format_ratingsMtx_rating_dev_training,
         full_training_ratings_mtx,
         csr_format_keyWordMtx_itemID_dev,
         coo_format_keyWordMtx_keyWord_dev,
         isBad, spot, num_entries);
        num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS;
        num_loops     += (long long int)1;
        spot           = num_loops * num_entries;
      };
      gpu_supplement_training_mtx_with_content_based_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training, 
       ratings_cols_training, 
       row_major_ordering,
       csr_format_ratingsMtx_userID_dev_training,
       coo_format_ratingsMtx_itemID_dev_training,
       coo_format_ratingsMtx_rating_dev_training,
       full_training_ratings_mtx,
       csr_format_keyWordMtx_itemID_dev,
       coo_format_keyWordMtx_keyWord_dev,
       isBad, spot, ratings_rows_training * ratings_cols_training - spot);
    }else{
      gpu_supplement_training_mtx_with_content_based_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training, 
       ratings_cols_training, 
       row_major_ordering,
       csr_format_ratingsMtx_userID_dev_training,
       coo_format_ratingsMtx_itemID_dev_training,
       coo_format_ratingsMtx_rating_dev_training,
       full_training_ratings_mtx,
       csr_format_keyWordMtx_itemID_dev,
       coo_format_keyWordMtx_keyWord_dev,
       isBad, 0, ratings_rows_training * ratings_cols_training);
    };



    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };

    if(Debug) LOG("finished supplmenting ratings mtx with content based information") ;
    cudaDeviceSynchronize();
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  /*if(Debug)*/ LOG("content based run time : "<<readable_time(program_time)<<std::endl);
}



template<typename Dtype>
__global__ void sparse_nearest_row_kernel(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection,  
 Dtype* error, bool* isBad)
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  const int row_skip = csr_rows_B[0];
  CUDA_KERNEL_LOOP(row_B,rows_B) {
    Dtype closest_A_row_dist = (Dtype)10000.0;
    int   closest_A_row      = 0;
    for(int row_A = 0; row_A < rows_A; row_A++){
      Dtype temp = (Dtype)0.0;
      for(int coo_index = csr_rows_B[row_B]; coo_index < csr_rows_B[row_B + 1]; coo_index++){
        int col = coo_cols_B[coo_index];
        temp += pow(dense_mtx_A[row_A + col * rows_A] - coo_entries_B[coo_index],(Dtype)2.0);
      }
      if(temp < closest_A_row_dist || row_A == 0){
        closest_A_row_dist = temp;
        closest_A_row      = row_A;
      }
    }
    selection[row_B] = closest_A_row;
    error[row_B] = closest_A_row_dist;

    if (::isinf(error[row_B]) || ::isnan(error[row_B])){
      isBad[0] = true;
    };
  };
}

template <typename Dtype>
void sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection,  
 Dtype* error) 
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */


    bool Debug = false;
    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

    // if(coo_A == NULL){
    //   Debug = false;
    // }else{
    //   Debug = true;
    // }

    if(Debug) LOG("sparse_nearest_row called");


    const long long int num_gpu_blocks= GET_BLOCKS(rows_B);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
    };


    if(Debug) {
      LOG("rows_A : "<< rows_A);
      LOG("cols : "<< cols);
      LOG("num_sparse_entries : "<< num_sparse_entries);
      LOG("sparse_error called");
      save_device_mtx_to_file<Dtype>(dense_mtx_A, rows_A, cols, "dense_mtx_A");
    //save_device_array_to_file<Dtype>(coo_errors, num_sparse_entries, "coo_errors");
      save_device_array_to_file<Dtype>(coo_entries_B, num_sparse_entries, "coo_entries_B");
      save_device_array_to_file<int>(coo_cols_B, num_sparse_entries, "coo_cols_B");
      save_device_array_to_file<int>(csr_rows_B, rows_B + 1, "csr_rows_B");
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
    };

    sparse_nearest_row_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A, cols, dense_mtx_A, 
                              rows_B, csr_rows_B, coo_cols_B, coo_entries_B, selection, error, isBad);
    if(Debug) {
      save_device_array_to_file<Dtype>(error, rows_B, "row_errors");
      save_device_array_to_file<int>(selection, rows_B, "nearest_row");
      LOG("Press Enter to continue.") ;
      std::cin.ignore();
    };    
    if(Debug && 0)LOG("Here!");
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };
    if(Debug) LOG("sparse_nearest_row call finished");
  }

template void sparse_nearest_row(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const float* coo_entries_B, int* selection,  
 float* error);


  __global__ void center_ratings_kernel(const float* user_means, const float* user_var, const int ratings_rows,
    const int* csr_format_ratingsMtx_userID_dev,
    const float* coo_format_ratingsMtx_rating_dev,
    float* coo_format_ratingsMtx_row_centered_rating_dev,
    const float val_when_var_is_zero, bool* isBad) 
  {

    CUDA_KERNEL_LOOP(row, ratings_rows){
      for(int i = csr_format_ratingsMtx_userID_dev[row]; i < csr_format_ratingsMtx_userID_dev[row + 1]; i++){
        float val = coo_format_ratingsMtx_rating_dev[i] - user_means[row];
        if(user_var[row] > (float)0.0){
          val = val / std::sqrt(user_var[row]);
        }else{
          val = val_when_var_is_zero;
        }

        coo_format_ratingsMtx_row_centered_rating_dev[i] = val;
        if (::isinf(val) || ::isnan(val)){
          isBad[0] = true;
        };
      }
    }
  }


void center_ratings(const float* user_means, const float* user_var, 
  const int ratings_rows, const int num_entries,
  const int* csr_format_ratingsMtx_userID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* coo_format_ratingsMtx_row_centered_rating_dev,
  const float val_when_var_is_zero)
{
  bool Debug = false;

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

// float min_ = gpu_min<float>(num_entries, coo_format_ratingsMtx_rating_dev);
// float max_ = gpu_abs_max<float>(num_entries, coo_format_ratingsMtx_rating_dev);

// max_ - 

  const long long int num_gpu_blocks= GET_BLOCKS(ratings_rows);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  center_ratings_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(user_means, user_var, ratings_rows,
    csr_format_ratingsMtx_userID_dev,
    coo_format_ratingsMtx_rating_dev,
    coo_format_ratingsMtx_row_centered_rating_dev,
    val_when_var_is_zero, isBad);

  if(Debug){
    float _min = gpu_min<float>           (num_entries, coo_format_ratingsMtx_row_centered_rating_dev);
    float _max = gpu_max<float>           (num_entries, coo_format_ratingsMtx_row_centered_rating_dev);
    float expt = gpu_expected_value<float>(num_entries, coo_format_ratingsMtx_row_centered_rating_dev);
    LOG("_min :"  <<_min);
    LOG("_max :"  <<_max);
    LOG("expt :"  <<expt);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}


__global__ void center_rows_kernel(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var, bool* isBad) 
{

  CUDA_KERNEL_LOOP(row, rows){
    float mean = (float)0.0;
    float std_dev = (float)0.0;
    for(int i = 0; i < cols; i++){
      mean += X[row + rows * i];
      std_dev += pow(X[row + rows * i], (float)2.0);
    }
    mean /= (float)cols;
    std_dev /= (float)cols;
    std_dev = std_dev - pow(mean, (float)2.0);
    user_var[row] = std_dev;
    std_dev = sqrt(std_dev);
    user_means[row] = mean;

    if(std_dev == (float)0.0 ){
      for(int i = 0; i < cols; i++){
        X[row + rows * i] = val_when_var_is_zero;
      } 
    }else{
      for(int i = 0; i < cols; i++){
        X[row + rows * i] = (X[row + rows * i] - mean) / std_dev;
        if (::isinf(X[row + rows * i]) || ::isnan(X[row + rows * i])){
          isBad[0] = true;
        };
      } 
    }       
  }
}

void center_rows(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var)
{
  bool Debug = false;

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  // float min_ = gpu_min<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
  // float max_ = gpu_abs_max<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);

  // max_ - 

  const int num_gpu_blocks= GET_BLOCKS(rows);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  center_rows_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols,  X, val_when_var_is_zero, user_means, user_var, isBad);


  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}


__device__ long long int from_below_diag_to_whole_device(long long int below_diag_index, int dimension){

  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) return (long long int)(-1);

  long long int num_so_far = (long long int)0;
  int col = 0;
  long long int num_in_col = (long long int)(dimension - 1);
  while(num_so_far < below_diag_index + (long long int)(1)){  
    if(num_so_far + num_in_col == below_diag_index + (long long int)(1)){
      return (long long int)(col + 1) * dimension - (long long int)(1);
    }
    if(num_so_far + num_in_col > below_diag_index + (long long int)(1)){ 
      break;
    }else{
      num_so_far += num_in_col;
      num_in_col -= (long long int)(1);
      col += 1;
    }
  }
  return (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;
}

__device__ long long int from_below_diag_to_whole_device_faster(long long int below_diag_index, long long int dimension)
{
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)){
    return (long long int)(-1);
  }

  long long int inverted_number = num_below_diag - below_diag_index;
  float n_float = ((float)1.0 + sqrt((float)1.0 + (float)8.0 * (float)inverted_number)) / (float)2.0;
  long long int n = (long long int)round(n_float);
  long long int row;
  long long int col;

  long long int one_less = ((n - (long long int)1) * (n - (long long int)2)) / (long long int)2;
  long long int on_it = (n * (n - (long long int)1)) / (long long int)2;
  long long int one_more = (n * (n + (long long int)1)) / (long long int)2;

  if(one_more < inverted_number){
    return (long long int)(-1);
    // col = dimension - n;
    // row = col + (inverted_number - one_more);
  }else if(one_less < inverted_number && inverted_number < on_it){
    col = dimension - n;
    row = dimension -  (inverted_number - one_less);
  }else if(inverted_number == on_it){
    col = dimension - n;
    row = col + (long long int)1;
  }else if(on_it < inverted_number && inverted_number < one_more){
    col = dimension - n - (long long int)1;
    row = dimension - (inverted_number - on_it);
  }else if(inverted_number == one_more){
    return (long long int)(-1);
    // col = dimension - n;
    // row = dimension - (long long int)1;
  } else {
    return (long long int)(-1);
    // col = dimension - n + (long long int)1;
    // row = dimension - (inverted_number - (n * (n - (long long int)1)) / (long long int)2);
  }
  
  if( row == col){
    return (long long int)(-1); 
  }
  if( col < (long long int)0 || col >= dimension){
    return (long long int)(-1);
  }
  if( row < (long long int)1 || row >= dimension){
    return (long long int)(-1);
  }
  return row + dimension * col;
}

__device__ long long int from_whole_to_below_diag_device(long long int whole_index, long long int dimension){
  bool debug = false;
  long long int row = (whole_index % dimension);
  long long int col = (whole_index / dimension);
  if(row == col) return (long long int)(-1);
  if(row < (long long int)0 || row > dimension - (long long int)1) return (long long int)(-1);
  if(col < (long long int)0 || col > dimension - (long long int)1) return (long long int)(-1);

  long long int temp = row;
  row = max(row, col);
  col = min(temp,col);
  // now row is larger than col

  long long int num_below_diag = (long long int)0;
  long long int count = 0;
  long long int num_in_col = (dimension - (long long int)1);
  while(count < col){
    num_below_diag += num_in_col;
    num_in_col -= (long long int)(1);
    count += (long long int)1;
  }
  return num_below_diag + row - (dimension - num_in_col);
}



__global__ void get_cosine_similarity_kernel(const int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity, bool* isBad) 
{

  CUDA_KERNEL_LOOP(entry, ratings_rows * ratings_rows){
    int user_i = entry % ratings_rows;
    int user_j = entry / ratings_rows;
    if( user_i == user_j){
      cosine_similarity[entry] = (float)1.0;
    }else{
      int   count   = 0;
      float num     = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      for(int i = csr_format_ratingsMtx_userID_dev[user_i]; i < csr_format_ratingsMtx_userID_dev[user_i + 1]; i++){
        for(int j = csr_format_ratingsMtx_userID_dev[user_j]; j < csr_format_ratingsMtx_userID_dev[user_j + 1]; j++){
          int user_i_itemID = coo_format_ratingsMtx_itemID_dev[i];
          int user_j_itemID = coo_format_ratingsMtx_itemID_dev[j];
          if( user_i_itemID == user_j_itemID){
            count   += 1;
            num     += coo_format_ratingsMtx_rating_dev[i] * coo_format_ratingsMtx_rating_dev[j] ;
            denom_i += pow(coo_format_ratingsMtx_rating_dev[i], (float)2.0) ;
            denom_j += pow(coo_format_ratingsMtx_rating_dev[j], (float)2.0) ; 
          }else if(user_i_itemID < user_j_itemID){
            break;
          }
        }
      }
      if(count > 0){
      //float temp = num / sqrt(denom_i * denom_j);
        float temp = count / sqrtf((csr_format_ratingsMtx_userID_dev[user_i + 1] - csr_format_ratingsMtx_userID_dev[user_i]) * (csr_format_ratingsMtx_userID_dev[user_j + 1] - csr_format_ratingsMtx_userID_dev[user_j]));
        cosine_similarity[entry] = temp;
        if (::isinf(temp) || ::isnan(temp)){
          isBad[0] = true;
        };
      }else{
        cosine_similarity[entry] = (float)0.0;
      }
    }
  }
}


void get_cosine_similarity(const long long int ratings_rows, const long long int num_entries,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity)
{
  bool Debug = false;

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

// float min_ = gpu_min<float>(num_entries, coo_format_ratingsMtx_rating_dev);
// float max_ = gpu_abs_max<float>(num_entries, coo_format_ratingsMtx_rating_dev);

// max_ - 

  const long long int num_gpu_blocks= GET_BLOCKS(ratings_rows * ratings_rows);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  get_cosine_similarity_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows,
    csr_format_ratingsMtx_userID_dev,
    coo_format_ratingsMtx_itemID_dev,
    coo_format_ratingsMtx_rating_dev,
    cosine_similarity, isBad);

  if(Debug){

  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}

__global__ void get_cosine_similarity_host_kernel(const long long int start, 
  const int num, 
  const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_dev, bool* isBad ) 
{
  // assume that cosine_similarity_dev is stored in column major ordering (a column stays together in memory).
  CUDA_KERNEL_LOOP(entry, num){
    long long int below_index = (long long int)entry + start;
    long long int whole_index = from_below_diag_to_whole_device_faster(below_index, ratings_rows);
    if(whole_index < (long long int)0 || whole_index >= ratings_rows * ratings_rows){
      isBad[0] = true;
      return;
    }
    int user_i = (int)(whole_index % ratings_rows);
    int user_j = (int)(whole_index / ratings_rows);
    if( user_i == user_j){
      isBad[0] = true;
      return;
      //cosine_similarity_dev[entry] = (float)1.0;
    }else{
      int   count   = 0;
      float num     = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      for(int i = csr_format_ratingsMtx_userID_dev[user_i]; i < csr_format_ratingsMtx_userID_dev[user_i + 1]; i++){
        int user_i_itemID = coo_format_ratingsMtx_itemID_dev[i];
        int user_j_itemID = 0;
        int start_j = csr_format_ratingsMtx_userID_dev[user_j];
        for(int j = start_j; j < csr_format_ratingsMtx_userID_dev[user_j + 1]; j++){
          user_j_itemID = coo_format_ratingsMtx_itemID_dev[j];
          if( user_i_itemID == user_j_itemID){
            count   += 1;
            num     += coo_format_ratingsMtx_rating_dev[i] * coo_format_ratingsMtx_rating_dev[j] ;
            denom_i += pow(coo_format_ratingsMtx_rating_dev[i], (float)2.0) ;
            denom_j += pow(coo_format_ratingsMtx_rating_dev[j], (float)2.0) ; 
            start_j = j + 1;
          }else if(user_i_itemID < user_j_itemID){
            start_j = j;
            break;
          }
        }
      }
      if(count > 0){
          //float temp = num / sqrt(denom_i * denom_j);
        float temp_i = (float)csr_format_ratingsMtx_userID_dev[user_i + 1] - (float)csr_format_ratingsMtx_userID_dev[user_i];
        float temp_j = (float)csr_format_ratingsMtx_userID_dev[user_j + 1] - (float)csr_format_ratingsMtx_userID_dev[user_j];
        float temp = ((float)count) / sqrtf(temp_i * temp_j);
        cosine_similarity_dev[entry] = temp;
        if (::isinf(temp) || ::isnan(temp)){
          isBad[0] = true;
          return;
        };
      }else{
        cosine_similarity_dev[entry] = (float)0.0;
      }
    }
  }
}



void get_cosine_similarity_host_kernel_debug(const long long int start, 
  const int num, 
  const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_dev, bool* isBad ) 
{
  bool debug = true;

  int* csr_format_ratingsMtx_userID_host = (int *)malloc((ratings_rows +1)* sizeof(int));
  checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host, csr_format_ratingsMtx_userID_dev,  (ratings_rows +1) *  sizeof(float), cudaMemcpyDeviceToHost));
  if(debug && 0){
    LOG("max_index =  : "<<csr_format_ratingsMtx_userID_host[ratings_rows]);
  }
  int* coo_format_ratingsMtx_itemID_host = (int *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * sizeof(int));
  float* coo_format_ratingsMtx_rating_host = (float *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * sizeof(float));
  float* cosine_similarity_host = (float *)malloc(num * sizeof(float));
  
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host, coo_format_ratingsMtx_itemID_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host, coo_format_ratingsMtx_rating_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cosine_similarity_host, cosine_similarity_dev,  num *  sizeof(float), cudaMemcpyDeviceToHost));

  for(int entry = 0; entry < num; entry++){
    long long int below_index = (long long int)entry + start;
    long long int whole_index = from_below_diag_to_whole_faster(below_index, ratings_rows);
    int user_i = (int)(whole_index % ratings_rows);
    int user_j = (int)(whole_index / ratings_rows);

    long long int whole_index_slow = from_below_diag_to_whole(below_index, ratings_rows);
    int user_i_slow = (int)(whole_index_slow % ratings_rows);
    int user_j_slow = (int)(whole_index_slow / ratings_rows);
    if(user_i_slow != user_i || user_j_slow != user_j){
      LOG("below_index : "<<below_index);
      LOG("whole_index : "<<whole_index);
      LOG("max whole index =  : "<<ratings_rows * ratings_rows - (long long int)1);
      LOG("user_i : "<<user_i);
      LOG("user_j : "<<user_j);

      LOG("slow whole_index : "<<whole_index_slow);
      LOG("slow user_i : "<<user_i_slow);
      LOG("slow user_j : "<<user_j_slow<<std::endl);
      
      return;
    }    
    if(debug && 0){
      LOG("below_index : "<<below_index);
      LOG("whole_index : "<<whole_index);
      LOG("max whole index =  : "<<ratings_rows * ratings_rows - (long long int)1);
      LOG("user_i : "<<user_i);
      LOG("user_j : "<<user_j);
    }

    if(whole_index < (long long int)0 || whole_index >= ratings_rows * ratings_rows){
      LOG("below_index : "<<below_index);
      LOG("whole_index : "<<whole_index);
      LOG("max whole index =  : "<<ratings_rows * ratings_rows - (long long int)1);
      LOG("user_i : "<<user_i);
      LOG("user_j : "<<user_j);
      LOG("Is BAD!");
      whole_index = from_below_diag_to_whole(below_index, ratings_rows);
      user_i = (int)(whole_index % ratings_rows);
      user_j = (int)(whole_index / ratings_rows);
      LOG("slow whole_index : "<<whole_index);
      LOG("slow user_i : "<<user_i);
      LOG("slow user_j : "<<user_j<<std::endl);
      
      return;
    }
    if( user_i == user_j){
      LOG("Is BAD because user_i == user_j");
      return;
    }else{
      int   count   = 0;
      float num     = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
        int user_i_itemID = coo_format_ratingsMtx_itemID_host[i];
        int user_j_itemID = 0;
        int start_j = csr_format_ratingsMtx_userID_host[user_j];
        //LOG("start_j : " <<start_j);
        for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
          user_j_itemID = coo_format_ratingsMtx_itemID_host[j];
          if(debug && 0){
            LOG("user_i : "<<user_i);
            LOG("coo index i : "<<i);
            LOG("user_i_itemID : "<<user_i_itemID);
            LOG("user_j : "<<user_j);
            LOG("coo index j : "<<j);
            LOG("user_j_itemID : "<<user_j_itemID);
            LOG("largest possible coo index : "<<csr_format_ratingsMtx_userID_host[ratings_rows] - 1);
          }
          if( user_i_itemID == user_j_itemID){
            count   += 1;
            num     += coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j] ;
            denom_i += pow(coo_format_ratingsMtx_rating_host[i], (float)2.0) ;
            denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ; 
            start_j = j + 1;
          }else if(user_i_itemID < user_j_itemID){
            start_j = j;
            break;
          }
        }
      }
      if(count > 0){
          //float temp = num / sqrt(denom_i * denom_j);
        float temp_i = (float)csr_format_ratingsMtx_userID_host[user_i + 1] - (float)csr_format_ratingsMtx_userID_host[user_i];
        float temp_j = (float)csr_format_ratingsMtx_userID_host[user_j + 1] - (float)csr_format_ratingsMtx_userID_host[user_j];
        float temp = ((float)count) / sqrtf(temp_i * temp_j);
        cosine_similarity_host[entry] = temp;
        if (::isinf(temp) || ::isnan(temp)){
          LOG("Is BAD!");
          return;
        };
      }else{
        cosine_similarity_host[entry] = (float)0.0;
      }
    }
  }
 
 free(csr_format_ratingsMtx_userID_host);
 free(coo_format_ratingsMtx_itemID_host);
 free(coo_format_ratingsMtx_rating_host);
 free(cosine_similarity_host);
}




 void get_cosine_similarity_host(const long long int ratings_rows, 
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_host)
 {

  bool Debug = false;
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) LOG("called get_cosine_similarity_host") ;
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;


  //long long int num_gpu_blocks = GET_BLOCKS(ratings_rows * ratings_rows);
  long long int num_gpu_blocks = GET_BLOCKS(num_below_diag);

  if(Debug) LOG("total loops : " <<ceil((num_below_diag) / (CUDA_NUM_BLOCKS*CUDA_NUM_THREADS))) ;


  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    const long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot = (long long int)0;

    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_entries * sizeof(float)));
    if(Debug) LOG("cosine_similarity_dev allocated") ;

    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      if(Debug && 0) {
        LOG("total entries : " <<num_below_diag) ;
        LOG("num entries that fit on GPU: " <<num_entries) ;
        LOG("num_loops : " <<num_loops) ;
        //save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
      }
      //checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  num_entries *  sizeof(float), cudaMemcpyHostToDevice));
      //CUDA_CHECK(cudaDeviceSynchronize());

      get_cosine_similarity_host_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(spot, num_entries, ratings_rows, 
        csr_format_ratingsMtx_userID_dev,
        coo_format_ratingsMtx_itemID_dev,
        coo_format_ratingsMtx_rating_dev,
        cosine_similarity_dev, isBad);

      // get_cosine_similarity_host_kernel_debug(spot, num_entries, ratings_rows, 
      //   csr_format_ratingsMtx_userID_dev,
      //   coo_format_ratingsMtx_itemID_dev,
      //   coo_format_ratingsMtx_rating_dev,
      //   cosine_similarity_dev, isBad);
      //CUDA_CHECK(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  num_entries *  sizeof(float), cudaMemcpyDeviceToHost));
      //CUDA_CHECK(cudaDeviceSynchronize());
      if(Debug && 0) {
        //LOG("Here") ;
        //save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
      }
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
      if(Debug) {
        LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
        LOG("num_loops : " <<num_loops) ;
        LOG("spot : " <<spot) ;
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        if(print) LOG("get_cosine_similarity_host average loop run time : "<<readable_time(program_time / (double)num_loops));
      }
    };
    if(Debug && 0) {
      LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
      LOG("num_loops : " <<num_loops) ;
      LOG("spot : " <<spot) ;
    }
    // spot is the number of entries done so far
    // total - (done) = left to go 
    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  (num_below_diag - spot) *  sizeof(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(spot, num_below_diag - spot, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  (num_below_diag - spot) *  sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(cosine_similarity_dev);
  }else{
    if(Debug) LOG("get_cosine_similarity_host num_gpu_blocks <= CUDA_NUM_BLOCKS") ;
    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_below_diag * sizeof(float)));

    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host,  num_below_diag *  sizeof(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(0, num_below_diag, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host , cosine_similarity_dev,  num_below_diag *  sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(cosine_similarity_dev);
  };


  if(Debug){

  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  if(print) LOG("finished call to get_cosine_similarity_host") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("get_cosine_similarity_host run time : "<<readable_time(program_time)<<std::endl);
}





void get_cosine_similarity_host_experiment(const long long int ratings_rows, 
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_host)
{
  ABORT_IF_NEQ(0, 1, "Function Not Ready.");
  bool Debug = false;
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) LOG("called get_cosine_similarity_host") ;
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;


  //long long int num_gpu_blocks = GET_BLOCKS(ratings_rows * ratings_rows);
  long long int num_gpu_blocks = GET_BLOCKS(num_below_diag);

  if(print) LOG("total loops : " <<ceil((num_below_diag) / (CUDA_NUM_BLOCKS*CUDA_NUM_THREADS))) ;


  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    const long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot = (long long int)0;

    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_entries * sizeof(float)));
    float * cosine_similarity_dev_1;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev_1, num_entries * sizeof(float)));
    float* cosine_similarity_dev_ptrs[2] = {cosine_similarity_dev, cosine_similarity_dev_1};
    if(Debug) LOG("cosine_similarity_dev allocated") ;

    #ifdef _OPENMP
      const int nthreads = 2;
      omp_set_num_threads(nthreads);
      omp_lock_t *locks = (omp_lock_t *)malloc(nthreads * sizeof(omp_lock_t));
      checkErrors(locks);
    #endif
    #pragma omp parallel
    {
      #ifdef _OPENMP
        int th_id = omp_get_thread_num();
      #endif
      for(long long int j = (long long int)th_id; j < nthreads; j += (long long int)nthreads){
        omp_init_lock(locks + j);
        if(th_id == 0){
          omp_unset_lock(locks + j);
        }else{
          omp_set_lock(locks + j);
        }
      }
    }

    #pragma omp parallel shared(num_loops, spot, cosine_similarity_dev)
    {
      #ifdef _OPENMP
        int th_id = omp_get_thread_num();
      #endif
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
        if(th_id == 0){
          omp_set_lock(locks + num_loops % 2);
          if(Debug) {
            LOG("Thread "<<th_id<<", total entries : " <<num_below_diag) ;
            LOG("Thread "<<th_id<<", num entries that fit on GPU: " <<num_entries) ;
            LOG("Thread "<<th_id<<", num_loops : " <<num_loops) ;
            save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
          }
          if(Debug) LOG("Thread "<<th_id<<", Here") ;
          float * this_;
          if (num_loops % 2 == 0){
            this_ = cosine_similarity_dev_ptrs[0];
          }else{
            this_ = cosine_similarity_dev_ptrs[1];
          }
          get_cosine_similarity_host_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(spot, num_entries, ratings_rows, 
              csr_format_ratingsMtx_userID_dev,
              coo_format_ratingsMtx_itemID_dev,
              coo_format_ratingsMtx_rating_dev,
              this_, isBad);

          num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
          num_loops += (long long int)1;
          spot = num_loops * num_entries;
          omp_unset_lock(locks + (num_loops - 1) % 2);
        }
        if(th_id == 1 && num_loops > 0){
          long long int temp = (num_loops - (long long int)1);
          omp_set_lock(locks + temp % 2);
          float * this_;
          if (temp % 2 == 0){
            this_ = cosine_similarity_dev_ptrs[0];
          }else{
            this_ = cosine_similarity_dev_ptrs[1];
          }
          if(Debug) LOG("Thread "<<th_id<<", Here") ;
          checkCudaErrors(cudaMemcpy(cosine_similarity_host + temp * num_entries, this_,  num_entries *  sizeof(float), cudaMemcpyDeviceToHost));
          
          if(Debug) {
            LOG("Thread "<<th_id<<", Here") ;
            save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
          }
          if(Debug) {
            LOG("Thread "<<th_id<<", num_gpu_blocks : " <<num_gpu_blocks) ;
            LOG("Thread "<<th_id<<", num_loops : " <<num_loops) ;
            LOG("Thread "<<th_id<<", spot : " <<spot) ;
            gettimeofday(&program_end, NULL);
            program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
            if(print) LOG("Thread "<<th_id<<", get_cosine_similarity_host run time so far : "<<readable_time(program_time));
          }
          gettimeofday(&program_end, NULL);
          program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
          if(print) LOG("Thread "<<th_id<<", average loop run time : "<<readable_time(program_time / (double)num_loops)<<" after "<<num_loops<<" loops");
          omp_unset_lock(locks + temp % 2);
        }
      } // end while loop
    }
    if(Debug) {
      LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
      LOG("num_loops : " <<num_loops) ;
      LOG("spot : " <<spot) ;
    }
    // spot is the number of entries done so far
    // total - (done) = left to go 
    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  (num_below_diag - spot) *  sizeof(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(spot, num_below_diag - spot, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  (num_below_diag - spot) *  sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(cosine_similarity_dev);
    cudaFree(cosine_similarity_dev_1);
  }else{
    if(Debug) LOG("get_cosine_similarity_host num_gpu_blocks <= CUDA_NUM_BLOCKS") ;
    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_below_diag * sizeof(float)));

    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host,  num_below_diag *  sizeof(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(0, num_below_diag, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host , cosine_similarity_dev,  num_below_diag *  sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(cosine_similarity_dev);
  };


  if(Debug){

  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  if(print) LOG("finished call to get_cosine_similarity_host") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("get_cosine_similarity_host run time : "<<readable_time(program_time)<<std::endl);
}


template<typename Dtype>
__global__ void gpu_permute_kernel(Dtype* A, const int* P, int rows, int cols, bool permute_rows, 
 const int start, const int size) 
{
  //A is rows by cols stored in col major ordering

  if(permute_rows){
    //pvt is an array length rows
    CUDA_KERNEL_LOOP(j, size /*cols*/) {
      int ind=0;
      Dtype temp=0;

      for(int i=0; i<(rows-1); i++){
          // get next index
        ind = P[i];
        while(ind<i)
          ind = P[ind];

          // swap elements in array
        temp = A[i + (j + start) * rows];
        A[i + (j + start) * rows] = A[ind + (j + start) * rows];
        A[ind + (j + start) * rows] = temp;
      };
    };
  } else{
    //pvt is an array length cols
    CUDA_KERNEL_LOOP(i, size /*rows*/) {
      int ind=0;
      Dtype temp=0;

      for(int j=0; j<(cols-1); j++){
              // get next index
        ind = P[j];
        while(ind<j)
          ind = P[ind];

              // swap elements in array
        temp = A[(i + start) + j * rows];
        A[(i + start) + j * rows] = A[(i + start) + ind * rows];
        A[(i + start) + ind * rows] = temp;
      };
    };
  }
}


template<typename Dtype>
void gpu_permute(Dtype* A, const int* P, long long int rows, long long int cols, bool permute_rows) 
{
  /*
  This function applies a permutation mtx P (rows x rows) to the left of mtx a (rows x cols)
  with the ones in P are located at indices indicated by pvt which is an array with length rows
  and entries 0-(rows-1)
  */
  long long int num_gpu_blocks;
  if(permute_rows){
    num_gpu_blocks= GET_BLOCKS(cols);
  }else{
    num_gpu_blocks= GET_BLOCKS(rows);
  };
  
  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //   ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // }; 
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
    long long int spot = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_permute_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, spot, num_entries);
      num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    gpu_permute_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, spot, permute_rows ? (cols - spot) : (rows - spot));
  }else{
    gpu_permute_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, 0, permute_rows ? cols : rows);
  };
}

template void gpu_permute<float>(float* A, const int* P, long long int rows, long long int cols, bool permute_rows);
template void gpu_permute<double>(double* A, const int* P, long long int rows, long long int cols, bool permute_rows);



void gpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const long long int M, const long long int N,  
  bool row_major_ordering, float* x, bool shuffle_rows)
{
    // x in M by N
    // if row_major_ordering 


  int * indicies;
  if(shuffle_rows){
    checkCudaErrors(cudaMalloc((void**)&indicies, M * sizeof(int)));
    gpu_set_as_index(indicies, M);
    gpu_shuffle_array<int>(dn_handle, M, indicies);
    if(row_major_ordering){
      gpu_permute<float>(x, indicies, N, M, !shuffle_rows); 
    }else{
      gpu_permute<float>(x, indicies, M, N, shuffle_rows); 
    };
    cudaFree(indicies);
  }else{
    checkCudaErrors(cudaMalloc((void**)&indicies, N * sizeof(int)));
    gpu_set_as_index(indicies, N);
    gpu_shuffle_array<int>(dn_handle, N, indicies);
    if(row_major_ordering){
      gpu_permute<float>(x, indicies, N, M, shuffle_rows); 
    }else{
      gpu_permute<float>(x, indicies, M, N, !shuffle_rows); 
    };
    cudaFree(indicies);
  }

} 

__global__ void gpu_normalize_mtx_rows_or_cols_kernel(const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool normalize_rows)
{
  if(normalize_rows){
    CUDA_KERNEL_LOOP(i, M ) {
      float norm = (float)0.0;
      if(row_major_ordering){
        for(int j = 0; j < N; j++){
          norm += pow(x[i * N + j], (float)2.0);
        }
        norm = sqrt(norm);
        for(int j = 0; j < N; j++){
          x[i * N + j] = x[i * N + j] / norm;
        }
      }else{
        for(int j = 0; j < N; j++){
          norm += pow(x[i + M * j], (float)2.0);
        }
        norm = sqrt(norm);
        for(int j = 0; j < N; j++){
          x[i + M * j] = x[i + M * j] / norm;
        }
      };
    }
  }else{
    CUDA_KERNEL_LOOP(j, N ) {
      float norm = (float)0.0;
      if(row_major_ordering){
        for(int i = 0; i < M; i++){
          norm += pow(x[i * N + j], (float)2.0);
        }
        norm = sqrt(norm);
        for(int i = 0; i < M; i++){
          x[i * N + j] = x[i * N + j] / norm;
        }
      }else{
        for(int i = 0; i < M; i++){
          norm += pow(x[i + M * j], (float)2.0);
        }
        norm = sqrt(norm);
        for(int i = 0; i < M; i++){
          x[i + M * j] = x[i + M * j] / norm;
        }
      };
    }
  }
}

void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool normalize_rows)
{
  long long int num_gpu_blocks;
  if(normalize_rows){
    num_gpu_blocks= GET_BLOCKS(M);
  }else{
    num_gpu_blocks= GET_BLOCKS(N);
  };
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_normalize_mtx_rows_or_cols_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(M, N, row_major_ordering, x, normalize_rows);
} 

template<typename Dtype>
__global__ void gpu_div_US_in_SVD_kernel(const long long int m, Dtype* U, const Dtype* S, 
  const long long int start, const long long int num, const bool right_divide_by_S,
  bool* isBad) 
{
  //U is m by m in column major ordering
  CUDA_KERNEL_LOOP(j, num ) {
    long long int l = (long long int)j + start;
    long long int k;
    if(right_divide_by_S){
        k = l / m; //get column
      }else{
        k = l % m; //get row
      }

      //U[l] = (int)k;
      //U[l] = S[k];
      U[l] = U[l] / S[k];
      if (::isinf(U[l]) || ::isnan(U[l])){
        isBad[0] = true;
      };
    };

  }
template <>
void gpu_div_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, 
 float* U, const float* S, const bool right_divide_by_S)
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks= GET_BLOCKS(m*num_latent_factors);

  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //     ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };

  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS*CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_div_US_in_SVD_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(m, U, S, spot, num_entries, right_divide_by_S, isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops     += (long long int)1;
      spot           = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_div_US_in_SVD_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(m, U, S, spot, m*num_latent_factors - spot, right_divide_by_S, isBad);
  }else{
    gpu_div_US_in_SVD_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(m, U, S, (long long int)0, m*num_latent_factors, right_divide_by_S, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
} 

template<typename Dtype>
__global__ void gpu_mult_US_in_SVD_kernel(const long long int m, Dtype* U, const Dtype* S, 
  const long long int start, const long long int num, const bool right_multiply_by_S,
  bool* isBad)  
{
  //U is m by m in column major ordering
  CUDA_KERNEL_LOOP(j, num ) {
    long long int l = (long long int)j + start;
    long long int k;
    if(right_multiply_by_S){
      k = l / m; //get column
    }else{
      k = l % m; //get row
    }

    U[l] = U[l] * S[k];
    //U[l] = S[k];
    if (::isinf(U[l]) || ::isnan(U[l])){
      isBad[0] = true;
    };
  };


}

template <>
void gpu_mult_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, 
 float* U, const float* S, const bool right_multiply_by_S)
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks= GET_BLOCKS(m*num_latent_factors);

  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //     ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };

  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_mult_US_in_SVD_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(m, U, S, spot, num_entries, right_multiply_by_S, isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_mult_US_in_SVD_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(m, U, S, spot, m * num_latent_factors - spot, right_multiply_by_S, isBad);
  }else{
    gpu_mult_US_in_SVD_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(m, U, S, (long long int)0, m * num_latent_factors, right_multiply_by_S, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
  cudaFree(isBad);
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
} 

template<typename Dtype>
__global__ void gpu_copy_and_transpose_kernel(const int M, const int N,  const Dtype* x, Dtype* y) 
{

  CUDA_KERNEL_LOOP(j, M * N) {
    int row = j % N;
    int col = j / N;
    y[col + row * M] = x[row + col * M];
  };

}

void gpu_copy_and_transpose(const int M, const int N,  const float* x, float* y) 
{
  // x is M by M in memory
  // y is M by N where N < M 
  const long long int num_gpu_blocks= GET_BLOCKS(M * N);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  gpu_copy_and_transpose_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(M, N, x, y);
}

template <>
void get_num_latent_factors<float>(cublasHandle_t dn_handle, const long long int m, float* S, 
  long long int* num_latent_factors, const float percent) 
{
  bool Debug = false;
  float sum;
  if(Debug) LOG("here!");
  gpu_asum<float>(dn_handle, m, S, &sum);
  if(Debug) LOG("here!");

  float* S_host = NULL;
  S_host = (float *)malloc(sizeof(float)*m); 
  checkErrors(S_host);
  CUDA_CHECK(cudaMemcpy(S_host, S, m * sizeof(float), cudaMemcpyDeviceToHost));

  if(Debug) LOG("here!");

  float sum_so_far;
  num_latent_factors[0] = m-1;
  for(int j = 0; j < m-1; j++){
    sum_so_far += S_host[j];
    if(sum_so_far / sum >= percent) {
      num_latent_factors[0] = j+1;
      if(Debug) LOG("num_latent_factors = "<< j+1);
      break;
    }
  }
  free(S_host);
  if(Debug) LOG("here!");
}

template <>
void preserve_first_m_rows<float>(const long long int old_lda, const long long int new_lda, const long long int sda,
  float* V) 
{
  if(new_lda > old_lda) {
    LOG("WARNING: new_lda > old_lda");
    return;
  }
  for(int j = 0; j < sda; j++){
    checkCudaErrors(cudaMemcpy(V + new_lda * j, V + old_lda * j, 
     new_lda * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  float* V_copy = NULL;
  V_copy = (float *)malloc(sizeof(float) * new_lda * sda); 
  checkErrors(V_copy);
  CUDA_CHECK(cudaMemcpy(V_copy, V, new_lda * sda * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(V);
  CUDA_CHECK(cudaMalloc ( (void**)&V  , sizeof(float) * new_lda * sda ) );
  CUDA_CHECK(cudaMemcpy(V, V_copy, new_lda * sda * sizeof(float), cudaMemcpyHostToDevice));
  free(V_copy);

}

template <>
void gpu_orthogonal_decomp<float>(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle,
  const long long int m, const long long int n, 
  long long int* num_latent_factors, const float percent,
  float* A, float* U, float* V) 
{
  /*
    A is m by n stored in col-maj ordering

    solution     A      =       U       *      S      *       V^T
              m by n          m by m         m by n         n by n

              ..but S only has min(m,n) non zero entries so if m < n

    solution     A      =       U*S      *       V^T
              m by n          m by n           n by n

  */

  bool Debug = false;


  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  float* A_host_copy = NULL;
  A_host_copy = (float *)malloc(sizeof(float)*m*n); 
  checkErrors(A_host_copy);
  CUDA_CHECK(cudaMemcpy(A_host_copy, A, m*n * sizeof(float), cudaMemcpyDeviceToHost));

  long long int lda, sda;
  float *d_U  = NULL;
  float *d_VT = NULL;
  if(n > m){
    //we have to solve the transpose problem
    if(Debug) LOG("Solving the tranpose problem...");
    d_U  = V;
    d_VT = U;
    lda  = n;
    sda  = m;
    transpose_in_place<float>(handle, m,  n, A);
  }else{
    d_U  = U;
    d_VT = V;
    lda  = m;
    sda  = n;
  };




  float *d_S     = NULL;
  int   *devInfo = NULL;
  float *d_work  = NULL;
  float *d_rwork = NULL;
  int   lwork    = 0;

  CUDA_CHECK(cudaMalloc ((void**)&d_S    , sizeof(float)*std::min(lda,sda)));
  CUDA_CHECK(cudaMalloc ((void**)&devInfo, sizeof(int)));
  CUDA_CHECK(cudaMalloc ((void**)&d_rwork, sizeof(float)*(std::min(lda,sda) - 1)));

  // query working space of SVD
  cusolver_status = cusolverDnSgesvd_bufferSize(
              dn_solver_handle,
              lda,
              sda,
              &lwork );
  CUSOLVER_CHECK(cusolver_status);
  //assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  CUDA_CHECK(cudaMalloc((void**)&d_work , sizeof(float)*lwork));

  // compute SVD
  signed char jobu  = 'N'; // min(m,n) columns of U
  signed char jobvt = 'A'; // min(m,n) rows of VT
  if(Debug && 0){
    LOG("cusolverDnSgesvd ("<<dn_solver_handle<<","<<jobu<<","<<jobvt<<", lda = "<<lda<<", sda = "<<sda<<","<<A<<", lda = "<<lda<<","
          <<d_S<<","<<d_U<<", ldu = "<<lda<<","<<d_VT<<", ldvt = "<<sda/*std::min(m,n)*/<<","<<d_work<<", lwork = "<<lwork<<","<<d_rwork<<","<<
      devInfo<<")");
      //typeid(a).name()
    LOG("cusolverDnSgesvd ("<<typeid(dn_solver_handle).name()<<","<<typeid(jobu).name()<<","<<typeid(jobvt).name()<<", lda = "<<typeid(lda).name()<<", sda = "<<typeid(sda).name()<<","<<typeid(A).name()<<", lda = "<<typeid(lda).name()<<","
          <<typeid(d_S).name()<<","<<typeid(d_U).name()<<", ldu = "<<typeid(lda).name()<<","<<typeid(d_VT).name()<<", ldvt = "<<typeid(sda).name()/*std::min(m,n)*/<<","<<typeid(d_work).name()<<", lwork = "<<typeid(lwork).name()<<","<<typeid(d_rwork).name()<<","<<
      typeid(devInfo).name()<<")");
  };
  cusolver_status = cusolverDnSgesvd (
    dn_solver_handle,
    jobu,
    jobvt,
    lda,
    sda,
      A,    // m by n                                 or         n by m  (if solving transposes problem)
      lda,  // lda is not less than max(1,m)
      d_S,  // min(m,n)
      d_U,  // lda by lda
      lda,  // ldu is not less than max(1,m)
      d_VT, // sda by sda
      sda /*std::min(m,n)*/,  // ldvt is not less than max(1,n)
      d_work,
      lwork,
      d_rwork,
      devInfo);
  CUSOLVER_CHECK(cusolver_status);
  CUDA_CHECK(cudaDeviceSynchronize());
  if(Debug && 0){
    CUDA_CHECK(cudaDeviceSynchronize());
    save_device_mtx_to_file<float>(U, m, m, "U");
    save_device_mtx_to_file<float>(V, n, m, "V");
    save_device_array_to_file<float>(d_S, std::min(m,n), "singular_values");
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();
  }
  CUDA_CHECK(cudaMemcpy(A, A_host_copy, m*n * sizeof(float), cudaMemcpyHostToDevice));
  free(A_host_copy);

  if(n > m){
      //the tranpose problem was solved
      //sda = m;
    if(Debug) LOG("need to transpose "<<m<<" by "<<m<<" mtx U");
    transpose_in_place<float>(handle, sda,  sda, U);
    gpu_gemm<float>(handle, false, true, m, n, m, (float)1.0, U, A, (float)0.0, V);
  }else{
      //sda = n;
    if(Debug) LOG("need to transpose "<<n<<" by "<<m<<" mtx V");
    transpose_in_place<float>(handle, sda,  sda, V);
    gpu_gemm<float>(handle, false, false, m, m, n, (float)1.0, V, A, (float)0.0, U);
  };
  if(Debug && 0){
    CUDA_CHECK(cudaDeviceSynchronize());
    save_device_mtx_to_file<float>(U, m, m, "U_2");
    save_device_mtx_to_file<float>(V, n, m, "V_2");\
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();
  }

  gpu_mult_US_in_SVD<float>(m, std::min(m,n), U, d_S, true);
  gpu_div_US_in_SVD<float>(n, std::min(m,n), V, d_S, true);

  float max_S = (float)0.0;
  CUDA_CHECK(cudaMemcpy(&max_S, d_S, sizeof(float), cudaMemcpyDeviceToHost));
  LOG("Largest Singular Value : "<<max_S);

  get_num_latent_factors<float>(handle, std::min(m,n), d_S, num_latent_factors, percent);

  if(Debug){
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG("num_latent_factors : "<<num_latent_factors[0]) ;
    save_device_mtx_to_file<float>(U, m, m, "U_3");
    save_device_mtx_to_file<float>(V, n, m, "V_3");
    save_device_mtx_to_file<float>(A, m, n, "A");
    

    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }    
  save_device_array_to_file<float>(d_S, std::min(m,n), "singular_values");


  if (d_S    ) cudaFree(d_S);
  if (devInfo) cudaFree(devInfo);
  if (d_work ) cudaFree(d_work);
  if (d_rwork) cudaFree(d_rwork);

  if(Debug){
    float * R;
    checkCudaErrors(cudaMalloc((void**)&R, m * n * sizeof(float)));

    /*
        A is m by n stored in col-maj ordering where m<<n
        V is n by m stored in col-maj ordering
        (V^T is m by n)
        U is m by m stored in col-maj ordering
    */
    // M, N, K
    //M number of columns of matrix op(A) and C.
    //N is number of rows of matrix op(B) and C.]
    //K is number of columns of op(B) and rows of op(A).

    // op(B) is N by K
    // op(A) is K by M
    // C is N by M
    // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C

    gpu_gemm<float>(handle, false,
               true, m, m, m /*num_latent_factors[0]*/,
     (float)1.0, U, U, (float)0.0,
     R);

    save_device_mtx_to_file<float>(R, m, m, "UUT", false);

    gpu_gemm<float>(handle, false,
               true, m, m, n /*num_latent_factors[0]*/,
     (float)1.0, V, V, (float)0.0,
     R);

    save_device_mtx_to_file<float>(R, m, m, "VTV", false);



    gpu_gemm<float>(handle, true,
               false, n, m, m /*num_latent_factors[0]*/,
     (float)1.0, V, U, (float)0.0,
     R);

    gpu_axpby<float>(handle, m * n, (float)1.0, A,
      (float)(-1.0), R);

    save_device_mtx_to_file<float>(R, m, n, "svd_error", false);

    //float range_A    = gpu_range<float>(m * n,  A);
        float error      = gpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
        cudaFree(R);
    // LOG("A mtx range of values = "<<range_A) ;
        LOG("SVD max error = "<<error) ;
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
        LOG("Press Enter to continue.") ;
        std::cin.ignore();

  }

  if(Debug) LOG("finished call to gpu_orthogonal_decomp") ;
  cudaDeviceSynchronize();
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  /*if(Debug)*/ LOG("gpu_orthogonal_decomp run time : "<<readable_time(program_time)<<std::endl);
}                    



// solve A*x = b by LU with partial pivoting
template <>
        int linearSolverLU<float>(
          cusolverDnHandle_t handle,
    int n, //columns of A
    const float *Acopy,
    int lda, //rows of A
    const float *b,
    float *x)
        {
    // solves  A*x = b  where b = ones(m,1)
    // A is m by n
    // x is n by 1
          int bufferSize = 0;
          int *info = NULL;
          float *buffer = NULL;
          float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    float start, stop;
    float time_solve;

    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    // start = second();
    // start = second();

    CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
      fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());

    // stop = second();
    // time_solve = stop - start;
    // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
  }

// solve A*x = b by LU with partial pivoting
template <>
  int linearSolverLU<double>(
    cusolverDnHandle_t handle,
    int n, //columns of A
    const double *Acopy,
    int lda, //rows of A
    const double *b,
    double *x)
  {
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    double start, stop;
    double time_solve;

    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    // start = second();
    // start = second();

    CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
      fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    // stop = second();

    // time_solve = stop - start;
    // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
  }



// solve A*x = b by QR
template <>
int linearSolverQR<float>(
  cublasHandle_t cublasHandle,
  cusolverDnHandle_t handle,
  int n, //columns of A
  const float *Acopy,
  int lda, //rows of A
  const float *b,
  float *x)
{
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    float *tau = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    const float one = 1.0;

    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize_geqrf));
    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(
      handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_OP_T,
      n,
      1,
      n,
      A,
      lda,
      NULL,
      x,
      n,
      &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
    
    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(float)*n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    // start = second();
    // start = second();

    // compute QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
      fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    CUSOLVER_CHECK(cusolverDnSormqr(
      handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_OP_T,
      n,
      1,
      n,
      A,
      lda,
      tau,
      x,
      n,
      buffer,
      bufferSize,
      info));

    // x = R \ Q^T*b
    CUBLAS_CHECK(cublasStrsm(
     cublasHandle,
     CUBLAS_SIDE_LEFT,
     CUBLAS_FILL_MODE_UPPER,
     CUBLAS_OP_N,
     CUBLAS_DIAG_NON_UNIT,
     n,
     1,
     &one,
     A,
     lda,
     x,
     n));
    checkCudaErrors(cudaDeviceSynchronize());
    // stop = second();

    // time_solve = stop - start;
    // fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
  }

// solve A*x = b by QR
template <>
int linearSolverQR<double>(
  cublasHandle_t cublasHandle,
  cusolverDnHandle_t handle,
  int n, //columns of A
  const double *Acopy,
  int lda, //rows of A
  const double *b,
  double *x)
{
  int bufferSize = 0;
  int bufferSize_geqrf = 0;
  int bufferSize_ormqr = 0;
  int *info = NULL;
  double *buffer = NULL;
  double *A = NULL;
  double *tau = NULL;
  int h_info = 0;
  double start, stop;
  double time_solve;
  const double one = 1.0;


  CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize_geqrf));
  CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(
    handle,
    CUBLAS_SIDE_LEFT,
    CUBLAS_OP_T,
    n,
    1,
    n,
    A,
    lda,
    NULL,
    x,
    n,
    &bufferSize_ormqr));

  printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
  
  bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

  checkCudaErrors(cudaMalloc(&info, sizeof(int)));
  checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
  checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
  checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(double)*n));

  // prepare a copy of A because getrf will overwrite A with L
  checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

  // start = second();
  // start = second();

  // compute QR factorization
  CUSOLVER_CHECK(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

  checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

  if ( 0 != h_info ){
    fprintf(stderr, "Error: LU factorization failed\n");
  }

  checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

  // compute Q^T*b
  CUSOLVER_CHECK(cusolverDnDormqr(
    handle,
    CUBLAS_SIDE_LEFT,
    CUBLAS_OP_T,
    n,
    1,
    n,
    A,
    lda,
    tau,
    x,
    n,
    buffer,
    bufferSize,
    info));

  // x = R \ Q^T*b
  CUBLAS_CHECK(cublasDtrsm(
   cublasHandle,
   CUBLAS_SIDE_LEFT,
   CUBLAS_FILL_MODE_UPPER,
   CUBLAS_OP_N,
   CUBLAS_DIAG_NON_UNIT,
   n,
   1,
   &one,
   A,
   lda,
   x,
   n));
  checkCudaErrors(cudaDeviceSynchronize());
  // stop = second();

  // time_solve = stop - start;
  // fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

  if (info  ) { checkCudaErrors(cudaFree(info  )); }
  if (buffer) { checkCudaErrors(cudaFree(buffer)); }
  if (A     ) { checkCudaErrors(cudaFree(A)); }
  if (tau   ) { checkCudaErrors(cudaFree(tau)); }

  return 0;
}


int gpu_get_num_entries_in_rows(const int first_row, const int last_row, const int* csr)
{
  int last_entry_index;
  int first_entry_index;

  CUDA_CHECK(cudaMemcpy(&first_entry_index, csr + first_row, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_entry_index, csr + last_row + 1, sizeof(int), cudaMemcpyDeviceToHost));

  return last_entry_index - first_entry_index;


}

int gpu_get_first_coo_index(const int first_row, const int* csr)
{
  int first_entry_index;

  CUDA_CHECK(cudaMemcpy(&first_entry_index, csr + first_row, sizeof(int), cudaMemcpyDeviceToHost));

  return first_entry_index;


}


template<typename Dtype>
__global__ void sparse_error_kernel(const int rows, const int cols, const Dtype* dense_mtx_A, 
 const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, Dtype* coo_errors, const int num_sparse_entries, 
 bool* isBad, Dtype* coo_A, bool Debug)
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

    const int row_skip = csr_rows_B[0];
    CUDA_KERNEL_LOOP(j,num_sparse_entries) {
      int count = 0;
      int row = 0;
    // while(count < j + 1){
    //   count += csr_rows_B[row + 1] - csr_rows_B[row]; //num thing in that row
    //   row +=1;
    // }
      for(row = 0; row < rows; row++){
      count += csr_rows_B[row + 1] - csr_rows_B[row]; //add the number of things in this row to the count
      if(count >= j + 1) break; //you're in the right row
    }
    // count >= j + 1
    int col = coo_cols_B[/*row_skip + */ j];
    coo_errors[j] = coo_entries_B[/*row_skip + */ j] - dense_mtx_A[row + col * rows];

    if(Debug){
      //coo_A[j] = (Dtype)row;
      coo_A[j] = dense_mtx_A[row + col * rows];
    }
    
    if (::isinf(coo_errors[j]) || ::isnan(coo_errors[j])){
      isBad[0] = true;
    };
  };
}

void sparse_error(const int rows, const int cols, const float* dense_mtx_A, 
  const int* csr_rows_B, const int* coo_cols_B,
  const float* coo_entries_B, float* coo_errors, const int num_sparse_entries,
  float* coo_A) 
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */


    bool Debug = false;
    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, sizeof(bool), cudaMemcpyHostToDevice));

    // if(coo_A == NULL){
    //   Debug = false;
    // }else{
    //   Debug = true;
    // }

    if(Debug) LOG("sparse_error called");


    const long long int num_gpu_blocks= GET_BLOCKS(num_sparse_entries);

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
    };


    if(Debug) {
      LOG("rows : "<< rows);
      LOG("cols : "<< cols);
      LOG("num_sparse_entries : "<< num_sparse_entries);
      LOG("sparse_error called");
      save_device_mtx_to_file<float>(dense_mtx_A, rows, cols, "dense_mtx_A");
    //save_device_array_to_file<float>(coo_errors, num_sparse_entries, "coo_errors");
      save_device_array_to_file<float>(coo_entries_B, num_sparse_entries, "coo_entries_B");
      save_device_array_to_file<int>(coo_cols_B, num_sparse_entries, "coo_cols_B");
      save_device_array_to_file<int>(csr_rows_B, rows + 1, "csr_rows_B");
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
    };

    sparse_error_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols, dense_mtx_A, csr_rows_B, coo_cols_B,
      coo_entries_B, coo_errors, num_sparse_entries,
      isBad, coo_A, Debug);
    if(Debug) {
      save_device_array_to_file<float>(coo_errors, num_sparse_entries, "coo_errors_rows");
      LOG("Press Enter to continue.") ;
      std::cin.ignore();
    };    
    if(Debug && 0)LOG("Here!");
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaFree(isBad);
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };
    if(Debug) LOG("sparse_error call finished");
  } 

template <>
  void gpu_spXdense_MMM_check<float>(const cublasHandle_t dn_handle, const bool TransA, const bool TransB, 
    const int m, const int n, const int k, const int first_ind,
    const float alpha,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float *B, const float beta, float *C)
  {

    int * csrRowPtrA_temp;
    CUDA_CHECK(cudaMalloc((void**)&csrRowPtrA_temp, (m+1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(csrRowPtrA_temp, csrRowPtrA, (m+1)  *  sizeof(float), cudaMemcpyDeviceToDevice));
    gpu_add_scalar<int>(m + 1, (int)(-1) * (int)first_ind, csrRowPtrA_temp);

    float* full_A;
    CUDA_CHECK(cudaMalloc((void**)&full_A, (long long int)m * (long long int)k * sizeof(float)));
    gpu_fill_training_mtx((long long int)m, (long long int)k, false, csrRowPtrA_temp, csrColIndA, csrValA, full_A);

    cublasOperation_t cuTransA =
    (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
    (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // CUSPARSE_CHECK( cusparseScsrmm2(handle, cuTransA, cuTransB, 
  //                                 m, n, k, nnz, alpha, descrA, 
  //                                 csrValA, csrRowPtrA_temp, csrColIndA, 
  //                                 B, ldb, beta, C, ldc) );

  //M, N, K
  //M number of columns of matrix op(A) and C.
  //N is number of rows of matrix op(B) and C.]
  //K is number of columns of op(B) and rows of op(A).

  // op(B) is N by K
  // op(A) is K by M
  // C is N by M
  // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C
    gpu_gemm<float>(dn_handle, cuTransB, cuTransA, 
      n, 
      (TransA == true) ? k : m, 
      (TransA == true) ? m : k,
      alpha, B, full_A, beta, C);

    cudaFree(full_A);
    cudaFree(csrRowPtrA_temp);
  }

template <>
  void gpu_spXdense_MMM<float>(const cusparseHandle_t handle, const bool TransA, const bool TransB, 
    const int m, const int n, const int k, const int nnz, const int first_ind,
    const float *alpha, const cusparseMatDescr_t descrA, 
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const float *B, const int ldb, const float *beta, float *C, const int ldc, bool Debug)
  {
    //bool Debug = true;
    if(Debug) LOG("gpu_spXdense_MMM called");
  /*
    This function performs one of the following matrix-matrix operations:

    C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

    A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

    op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONJUCUTE_TRANSPOSE
    and

    op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
    array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.

    n is the number of columns of dense matrix op(B) and C.
  */


    int * csrRowPtrA_temp;
    CUDA_CHECK(cudaMalloc((void**)&csrRowPtrA_temp, (m+1) * sizeof(float)));
    checkCudaErrors(cudaMemcpy(csrRowPtrA_temp, csrRowPtrA, (m+1)  *  sizeof(float), cudaMemcpyDeviceToDevice));
    gpu_add_scalar<int>(m + 1, (int)(-1) * (int)first_ind, csrRowPtrA_temp);

    if(Debug){
      LOG("first_ind: "<<first_ind);
      LOG("first entry of csrRowPtrA_temp: ");
      print_gpu_array_entries<int>(csrRowPtrA_temp, 1);
      LOG("C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C");
      if(TransA){
        LOG("A is "<<m<<" by "<< k<<" in CSR format but it will be transposed");
        if(TransB){
          LOG("B is "<<n<<" by "<< m <<" in dense format but it will be tranposed");
          save_device_mtx_to_file<float>(B, n, m, "B");
        }else{
          LOG("B is "<<m<<" by "<< n);
          save_device_mtx_to_file<float>(B, m, n, "B");
        };
        LOG("C is "<<k<<" by "<< n);
      }else{
        LOG("A is "<<m<<" by "<< k<<" in CSR format");
        if(TransB){
          LOG("B is "<<n<<" by "<< k <<" in dense format but it will be tranposed");
          save_device_mtx_to_file<float>(B, n, m, "B");
        }else{
          LOG("B is "<<k<<" by "<< n);
          save_device_mtx_to_file<float>(B, m, n, "B");
        };
        LOG("C is "<<m<<" by "<< n);
      };
      LOG("Here!");
      cudaDeviceSynchronize();
      save_device_array_to_file<int>(csrRowPtrA, m + 1, "csrRowPtrA");
      save_device_array_to_file<int>(csrColIndA, m + 1, "csrColIndA");
      save_device_array_to_file<float>(csrValA, m + 1, "csrValA");
      
      save_device_mtx_to_file<float>(C, k, n, "C");

      LOG("Here!");
      cudaDeviceSynchronize();
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
    }

    cusparseOperation_t cuTransA =
    (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    cusparseOperation_t cuTransB =
    (TransB == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    CUSPARSE_CHECK( cusparseScsrmm2(handle, cuTransA, cuTransB, 
      m, n, k, nnz, alpha, descrA, 
      csrValA, csrRowPtrA_temp, csrColIndA, 
      B, ldb, beta, C, ldc) );

    cudaFree(csrRowPtrA_temp);

    bool isBad = gpu_isBad<float>(C, (TransA == true) ? k*n : m*n);
    if(isBad){
      ABORT_IF_NEQ(0, 1, "isBad");
    };
    if(Debug) LOG("gpu_spXdense_MMM call finished");
  }






template <>
void gpu_R_error<float>(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
  const int batch_size_testing, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
  const int nnz, const int first_coo_ind, const bool compress, 
  float* testing_entries, float* coo_testing_errors, const float testing_fraction,
  const float *coo_format_ratingsMtx_rating_dev_testing, 
  const int *csr_format_ratingsMtx_userID_dev_testing_batch, 
  const int *coo_format_ratingsMtx_itemID_dev_testing,
  const float *V, float *U_testing, float *R_testing,
  std::string name, float training_rate, float regularization_constant)
{
  ABORT_IF_EQ(0, 1, "This version is currently outdated.");
    bool Debug = false;
    std::string title  =  (name + "_gpu_R_error").c_str();
    LOG((title + " called").c_str());

    struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);


  //============================================================================================
  // Initialize  R_testing * V = U_testing
  //============================================================================================ 
  /*
      R_testing is sparse batch_size_testing by ratings_cols
      V is dense ratings_cols by batch_size_CU (or num_latent_factors)

      Here we want to further sparsify R_testing to measure 
      the prediction on entries we already know

      gpu_spXdense_MMM performs
      C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C


      notes: 

      V^T * V   is the identity mtx if compress==false
      V   * V^T is not the identity mtx 
  */

  gpu_get_rand_bools<float>(nnz,  testing_entries, (float)1.0 - testing_fraction /*probability of 1*/);

      if(Debug && 0){
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, nnz, "ratings_testing_before_hadamard");
      }


      gpu_hadamard<float>(nnz, coo_format_ratingsMtx_rating_dev_testing, testing_entries );

      if(Debug && 0){
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, nnz, "ratings_testing_after_hadamard");
      }


      float alpha = (float)1.0;
      float beta = (float)0.0;
  // float *alpha_dev;
  // float *beta_dev;
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, sizeof(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, sizeof(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, sizeof(float), cudaMemcpyHostToDevice));

      float *U_testing_check;

      if(Debug && 0 ){


        if(compress){
          LOG("num_latent_factors : "<<num_latent_factors);
    //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * num_latent_factors * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_testing<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"testing_entries,csr_format_ratingsMtx_userID_dev_testing_batch, coo_format_ratingsMtx_itemID_dev_testing,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_testing, ldc = "<<batch_size_testing<<" );"  );
        }else{
    //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * batch_size_CU  * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_testing<<" , n = "<<batch_size_CU<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"testing_entries,csr_format_ratingsMtx_userID_dev_testing_batch, coo_format_ratingsMtx_itemID_dev_testing,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_testing, ldc = "<<batch_size_testing<<" );"  );    
        }




    // gpu_spXdense_MMM_check<float>(dn_handle, false, false, batch_size_testing,
    //                              (compress == false) ? batch_size_CU : num_latent_factors, 
    //                              ratings_cols, first_coo_ind, alpha, 
    //                              testing_entries, 
    //                              csr_format_ratingsMtx_userID_dev_testing_batch, 
    //                              coo_format_ratingsMtx_itemID_dev_testing,
    //                              V, beta, U_testing_check);
      }

      gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_testing,
       (compress == false) ? batch_size_CU : num_latent_factors, 
       ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
       testing_entries, 
       csr_format_ratingsMtx_userID_dev_testing_batch, 
       coo_format_ratingsMtx_itemID_dev_testing,
       V, ratings_cols, &beta, U_testing, batch_size_testing, Debug);


  // V^T * V
  // gpu_gemm<float>(dn_handle, false, true, (compress == false) ? batch_size_CU : num_latent_factors,
  //                 (compress == false) ? batch_size_CU : num_latent_factors, ratings_cols, 
  //                 (float)1.0, V, V, (float)0.0, );


  //============================================================================================
  // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
  //============================================================================================ 
  /*
      Here we want to measure the error on the places we know 
      that were predicted in the previous step
  */

      if(Debug && 0){
        if(compress){
          save_device_mtx_to_file<float>(U_testing, batch_size_testing, num_latent_factors, "U_testing");
          save_device_mtx_to_file<float>(V, ratings_cols, num_latent_factors, "V");
      //save_device_arrays_side_by_side_to_file<float>(U_testing, U_testing_check, batch_size_testing * num_latent_factors, "U_testing");
        }else{
          save_device_mtx_to_file<float>(U_testing, batch_size_testing, batch_size_CU, "U_testing");
          save_device_mtx_to_file<float>(V, ratings_cols, batch_size_CU, "V");
      //save_device_arrays_side_by_side_to_file<float>(U_testing, U_testing_check, batch_size_testing * batch_size_CU     , "U_testing");
        }
    //save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_testing, testing_entries, nnz, "R_testing");
    //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v2");
      }

      int i = 0;
      bool not_done = true;
      float error;

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;

      while(not_done && i < 20000){
  //for(int i = 0; i < 5; i++){
    // M, N, K
    // M number of columns of matrix op(A) and C.
    // N is number of rows of matrix op(B) and C.]
    // K is number of columns of op(B) and rows of op(A).

    // op(B) is N by K
    // op(A) is K by M
    // C is N by M
    // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C
        gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_testing, 
          (compress == false) ? batch_size_CU : num_latent_factors,
          (float)1.0, V, U_testing, (float)0.0, R_testing);
        bool isBad = gpu_isBad<float>(R_testing, batch_size_testing * ratings_cols);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };

        float *coo_R;
        if(Debug){
          checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * sizeof(float)));
      //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v3");
        }

        sparse_error(batch_size_testing, ratings_cols, R_testing, 
         csr_format_ratingsMtx_userID_dev_testing_batch, 
         coo_format_ratingsMtx_itemID_dev_testing,
         coo_format_ratingsMtx_rating_dev_testing, 
         coo_testing_errors, nnz, coo_R);
        isBad = gpu_isBad<float>(coo_testing_errors, nnz);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };


        float training_error_temp = gpu_sum_of_squares<float>(nnz, coo_testing_errors);
        if( Debug)  LOG("gpu_R_error iteration : "<<i);
        long long int nnz_ = (long long int)((float)nnz * testing_fraction);


        float temp = training_error_temp / (float)(nnz_);
        float epsilon = 0.00001;
        if(i > 1){
      //if(i % 1000 ==0) LOG("gpu_R_error iteration : "<<i);
      //have we stopped improving?
          if(error - temp < epsilon){
            if (abs(error - temp) > epsilon){
              LOG("gpu_R_error jumped over minimum iteration : "<<i);
          //we jumped over the minimum
              training_rate = training_rate / (float)10.0;
            }else{
          //we've learned enough
              not_done = false;
            }

          }
        }
        error = temp;


        if( Debug){
          LOG("gpu_R_error average error : "<< error); 
      //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

          float *testing_entries_temp; 
          checkCudaErrors(cudaMalloc((void**)&testing_entries_temp, nnz * sizeof(float)));
          checkCudaErrors(cudaMemcpy(testing_entries_temp, testing_entries, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

          temp = gpu_sum_of_squares_of_diff(dn_handle, nnz, 
            coo_format_ratingsMtx_rating_dev_testing, 
            testing_entries_temp);
          LOG("gpu_R_error error normalized by should be max error: "<< training_error_temp / (float)(nnz_)<<std::endl); 
      //LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 


          std::string title = ("ratings_testing_errors_v" + ToString<int>(i)).c_str();
          save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, 
           coo_R, coo_testing_errors, nnz, "ratings_testing_errors_v");

          cudaFree(coo_R);
          cudaFree(testing_entries_temp);
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();

        }



    //============================================================================================
    // (Update  U = U * (1 -alpha * lambda) + alpha * Error * V ) <- random error?????
    //============================================================================================ 
    /*
        m,n,k
        This function performs one of the following matrix-matrix operations:

        C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

        A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

        op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONJUCUTE_TRANSPOSE
        and

        op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
        array of dimensions (ldb, n) if op(B)=B, and (ldb, k) otherwise.

        n is the number of columns of dense matrix op(B) and C.
    */




        beta = (float)1.0 - training_rate * regularization_constant;
        gpu_spXdense_MMM<float>(sp_handle, false, false, 
         batch_size_testing, 
         (compress == false) ? batch_size_CU : num_latent_factors, 
         ratings_cols,
         nnz, first_coo_ind, &training_rate, sp_descr, 
         coo_testing_errors, 
         csr_format_ratingsMtx_userID_dev_testing_batch, 
         coo_format_ratingsMtx_itemID_dev_testing,
         V, ratings_cols, &beta, U_testing, batch_size_testing, Debug);

        i += 1 ;
  } // end while

  if(1) {
    //LOG("gpu_R_error call finished");
    LOG((title + " total iterations : ").c_str()<<i);
    LOG((title + " error : ").c_str()<<error);     
  }

  //if(Debug) cudaFree(U_testing_check);
  cudaDeviceSynchronize();
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG((title + " run time : ").c_str()<<readable_time(program_time)<<std::endl);
}





template <>
void gpu_R_error_training<float>(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
  const int batch_size_training, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
  const int nnz, const int first_coo_ind, const bool compress, float* coo_errors, 
  const float *coo_format_ratingsMtx_rating_dev_training, 
  const int *csr_format_ratingsMtx_userID_dev_training_batch, 
  const int *coo_format_ratingsMtx_itemID_dev_training,
  const float *V, float *U_training, float *R_training, 
  float training_rate, float regularization_constant)
{
  bool Debug = false;
  LOG("gpu_R_error_training called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  //============================================================================================
  // Initialize  R_training * V = U_training
  //============================================================================================ 
  /*
      R_training is sparse batch_size_training by ratings_cols
      V is dense ratings_cols by batch_size_CU (or num_latent_factors)

      Here we want to further sparsify R_training to measure 
      the prediction on entries we already know

      gpu_spXdense_MMM performs
      C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C


      notes: 

      V^T * V   is the identity mtx if compress==false
      V   * V^T is not the identity mtx 
  */

      float alpha = (float)1.0;
      float beta = (float)0.0;
  // float *alpha_dev;
  // float *beta_dev;
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, sizeof(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, sizeof(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, sizeof(float), cudaMemcpyHostToDevice));

        float *U_training_check;
        if(compress){
          CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * num_latent_factors * sizeof(float) ) );
        }else{
          CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * batch_size_CU  * sizeof(float) ) );  
        }

      if(Debug && 0 ){


        if(compress){
          LOG("num_latent_factors : "<<num_latent_factors);
    //CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * num_latent_factors * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_training<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"coo_format_ratingsMtx_rating_dev_training,csr_format_ratingsMtx_userID_dev_training_batch, coo_format_ratingsMtx_itemID_dev_training,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_training, ldc = "<<batch_size_training<<" );"  );
        }else{
    //CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * batch_size_CU  * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_training<<" , n = "<<batch_size_CU<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"coo_format_ratingsMtx_rating_dev_training,csr_format_ratingsMtx_userID_dev_training_batch, coo_format_ratingsMtx_itemID_dev_training,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_training, ldc = "<<batch_size_training<<" );"  );    
        }




    // gpu_spXdense_MMM_check<float>(dn_handle, false, false, batch_size_training,
    //                              (compress == false) ? batch_size_CU : num_latent_factors, 
    //                              ratings_cols, first_coo_ind, alpha, 
    //                              coo_format_ratingsMtx_rating_dev_training, 
    //                              csr_format_ratingsMtx_userID_dev_training_batch, 
    //                              coo_format_ratingsMtx_itemID_dev_training,
    //                              V, beta, U_training_check);
      }

      gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_training,
       (compress == false) ? batch_size_CU : num_latent_factors, 
       ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
       coo_format_ratingsMtx_rating_dev_training, 
       csr_format_ratingsMtx_userID_dev_training_batch, 
       coo_format_ratingsMtx_itemID_dev_training,
       V, ratings_cols, &beta, U_training, batch_size_training, Debug);


  // V^T * V
  // gpu_gemm<float>(dn_handle, false, true, (compress == false) ? batch_size_CU : num_latent_factors,
  //                 (compress == false) ? batch_size_CU : num_latent_factors, ratings_cols, 
  //                 (float)1.0, V, V, (float)0.0, );


  //============================================================================================
  // Compute  Error = R_training -  U_training * V^T  <-- sparse
  //============================================================================================ 
  /*
      Here we want to measure the error on the places we know 
      that were predicted in the previous step
  */

      if(Debug && 0){
        if(compress){
          save_device_mtx_to_file<float>(U_training, batch_size_training, num_latent_factors, "U_training");
          save_device_mtx_to_file<float>(V, ratings_cols, num_latent_factors, "V");
      //save_device_arrays_side_by_side_to_file<float>(U_training, U_training_check, batch_size_training * num_latent_factors, "U_training");
        }else{
          save_device_mtx_to_file<float>(U_training, batch_size_training, batch_size_CU, "U_training");
          save_device_mtx_to_file<float>(V, ratings_cols, batch_size_CU, "V");
      //save_device_arrays_side_by_side_to_file<float>(U_training, U_training_check, batch_size_training * batch_size_CU     , "U_training");
        }
      }

      int i = 0;
      bool not_done = true;
      float error;

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;

      while(not_done && i < 20000){
  //for(int i = 0; i < 5; i++){
    // M, N, K
    // M number of columns of matrix op(A) and C.
    // N is number of rows of matrix op(B) and C.]
    // K is number of columns of op(B) and rows of op(A).

    // op(B) is N by K
    // op(A) is K by M
    // C is N by M
    // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C
        gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_training, 
          (compress == false) ? batch_size_CU : num_latent_factors,
          (float)1.0, V, U_training, (float)0.0, R_training);
        bool isBad = gpu_isBad<float>(R_training, batch_size_training * ratings_cols);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };

        float *coo_R;
        if(Debug){
          checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * sizeof(float)));
          LOG("WARNING : check if Debug = true in sparse_error_kernel")
        }

        sparse_error(batch_size_training, ratings_cols, R_training, 
         csr_format_ratingsMtx_userID_dev_training_batch, 
         coo_format_ratingsMtx_itemID_dev_training,
         coo_format_ratingsMtx_rating_dev_training, 
         coo_errors, nnz, coo_R);
        isBad = gpu_isBad<float>(coo_errors, nnz);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };


        float training_error_temp = gpu_sum_of_squares<float>(nnz, coo_errors);
        if( Debug)  LOG("gpu_R_error iteration : "<<i);


        float temp = training_error_temp / nnz;
        float epsilon = 0.00001;
        if(i > 1){
      //if(i % 1000 ==0) LOG("gpu_R_error iteration : "<<i);
      //have we stopped improving?
          if(error - temp < epsilon){
            if (abs(error - temp) > epsilon){
              LOG("gpu_R_error jumped over minimum iteration : "<<i);
          //we jumped over the minimum
              if(compress){
                checkCudaErrors(cudaMemcpy(U_training, U_training_check, batch_size_training * num_latent_factors * sizeof(float), cudaMemcpyDeviceToDevice));
              }else{
                checkCudaErrors(cudaMemcpy(U_training, U_training_check, batch_size_training * batch_size_CU * sizeof(float), cudaMemcpyDeviceToDevice));
              }
              training_rate = training_rate / (float)10.0;
            }else{
          //we've learned enough
              not_done = false;
            }

          }
        }
        error = temp;


        if( Debug){
          LOG("gpu_R_error average error : "<< error); 
      //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

          cudaFree(coo_R);
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();

        }



    //============================================================================================
    // (Update  U = U * (1 -alpha * lambda) + alpha * Error * V ) <- random error?????
    //============================================================================================ 
    /*
        m,n,k
        This function performs one of the following matrix-matrix operations:

        C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

        A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

        op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONJUCUTE_TRANSPOSE
        and

        op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
        array of dimensions (ldb, n) if op(B)=B, and (ldb, k) otherwise.

        n is the number of columns of dense matrix op(B) and C.
    */


        if(compress){
          checkCudaErrors(cudaMemcpy(U_training_check, U_training, batch_size_training * num_latent_factors * sizeof(float), cudaMemcpyDeviceToDevice));
        }else{
          checkCudaErrors(cudaMemcpy(U_training_check, U_training, batch_size_training * batch_size_CU * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        beta = (float)1.0 - training_rate * regularization_constant;
        gpu_spXdense_MMM<float>(sp_handle, false, false, 
         batch_size_training, 
         (compress == false) ? batch_size_CU : num_latent_factors, 
         ratings_cols,
         nnz, first_coo_ind, &training_rate, sp_descr, 
         coo_errors, 
         csr_format_ratingsMtx_userID_dev_training_batch, 
         coo_format_ratingsMtx_itemID_dev_training,
         V, ratings_cols, &beta, U_training, batch_size_training, Debug);

        i += 1 ;
  } // end while

  if(1) {
    //LOG("gpu_R_error call finished");
    LOG("gpu_R_error_training total iterations : "<<i);
    LOG("gpu_R_error_training error : "<<error);     
  }

  cudaFree(U_training_check);
  cudaDeviceSynchronize();
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("gpu_R_error_training run time : "<<readable_time(program_time)<<std::endl);
}

template <>
void gpu_R_error_testing<float>(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
  const int batch_size_testing, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
  const int nnz, const int first_coo_ind, const bool compress, 
  float* testing_entries, float* coo_testing_errors, const float testing_fraction,
  const float *coo_format_ratingsMtx_rating_dev_testing, 
  const int *csr_format_ratingsMtx_userID_dev_testing_batch, 
  const int *coo_format_ratingsMtx_itemID_dev_testing,
  const float *V, float *U_testing, float *R_testing, 
  float training_rate, float regularization_constant,
  float* testing_error_on_training_entries, float* testing_error_on_testing_entries, 
  long long int* total_iterations)
{
  bool Debug = false;
  LOG("gpu_R_error_testing called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  //============================================================================================
  // Initialize  R_testing * V = U_testing
  //============================================================================================ 
  /*
      R_testing is sparse batch_size_testing by ratings_cols
      V is dense ratings_cols by batch_size_CU (or num_latent_factors)

      Here we want to further sparsify R_testing to measure 
      the prediction on entries we already know

      gpu_spXdense_MMM performs
      C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C


      notes: 

      V^T * V   is the identity mtx if compress==false
      V   * V^T is not the identity mtx 
  */
      float *coo_R;

  gpu_get_rand_bools<float>(nnz,  testing_entries, (float)1.0 - testing_fraction /*probability of 1*/);
      float *testing_entries_cpy; 
      checkCudaErrors(cudaMalloc((void**)&testing_entries_cpy, nnz * sizeof(float)));
      checkCudaErrors(cudaMemcpy(testing_entries_cpy, testing_entries, nnz * sizeof(float), cudaMemcpyDeviceToDevice));


      if(Debug){
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, nnz, "ratings_testing_before_hadamard");
      }

      gpu_hadamard<float>(nnz, coo_format_ratingsMtx_rating_dev_testing, testing_entries );

      if(Debug){
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, nnz, "ratings_testing_after_hadamard");
      }


      float alpha = (float)1.0;
      float beta = (float)0.0;
  // float *alpha_dev;
  // float *beta_dev;
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, sizeof(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, sizeof(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, sizeof(float), cudaMemcpyHostToDevice));

      float *U_testing_check;
        if(compress){
          CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * num_latent_factors * sizeof(float) ) );
        }else{
          CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * batch_size_CU  * sizeof(float) ) );  
        }

      if(Debug && 0 ){


        if(compress){
          LOG("num_latent_factors : "<<num_latent_factors);
    //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * num_latent_factors * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_testing<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"testing_entries,csr_format_ratingsMtx_userID_dev_testing_batch, coo_format_ratingsMtx_itemID_dev_testing,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_testing, ldc = "<<batch_size_testing<<" );"  );
        }else{
    //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * batch_size_CU  * sizeof(float) ) );
          LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
            <<"m = "<<batch_size_testing<<" , n = "<<batch_size_CU<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
            <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
            <<"&alpha, sp_descr,"<<std::endl<<"                                                "
            <<"testing_entries,csr_format_ratingsMtx_userID_dev_testing_batch, coo_format_ratingsMtx_itemID_dev_testing,"<<std::endl<<"                                                "
            <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
            <<"U_testing, ldc = "<<batch_size_testing<<" );"  );    
        }




    // gpu_spXdense_MMM_check<float>(dn_handle, false, false, batch_size_testing,
    //                              (compress == false) ? batch_size_CU : num_latent_factors, 
    //                              ratings_cols, first_coo_ind, alpha, 
    //                              testing_entries, 
    //                              csr_format_ratingsMtx_userID_dev_testing_batch, 
    //                              coo_format_ratingsMtx_itemID_dev_testing,
    //                              V, beta, U_testing_check);
      }



      gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_testing,
       (compress == false) ? batch_size_CU : num_latent_factors, 
       ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
       testing_entries, 
       csr_format_ratingsMtx_userID_dev_testing_batch, 
       coo_format_ratingsMtx_itemID_dev_testing,
       V, ratings_cols, &beta, U_testing, batch_size_testing, Debug);


  // V^T * V
  // gpu_gemm<float>(dn_handle, false, true, (compress == false) ? batch_size_CU : num_latent_factors,
  //                 (compress == false) ? batch_size_CU : num_latent_factors, ratings_cols, 
  //                 (float)1.0, V, V, (float)0.0, );


  //============================================================================================
  // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
  //============================================================================================ 
  /*
      Here we want to measure the error on the places we know 
      that were predicted in the previous step
  */

      if(Debug && 0){
        if(compress){
          save_device_mtx_to_file<float>(U_testing, batch_size_testing, num_latent_factors, "U_testing");
          save_device_mtx_to_file<float>(V, ratings_cols, num_latent_factors, "V");
          //save_device_arrays_side_by_side_to_file<float>(U_testing, U_testing_check, batch_size_testing * num_latent_factors, "U_testing");
        }else{
          save_device_mtx_to_file<float>(U_testing, batch_size_testing, batch_size_CU, "U_testing");
          save_device_mtx_to_file<float>(V, ratings_cols, batch_size_CU, "V");
          //save_device_arrays_side_by_side_to_file<float>(U_testing, U_testing_check, batch_size_testing * batch_size_CU     , "U_testing");
        }
        //save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_testing, testing_entries, nnz, "R_testing");
      }
      if(Debug){
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v2");
      }

      int i = 0;
      bool not_done = true;
      float error;

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;

      while(not_done && i < 20000){
      //for(int i = 0; i < 5; i++){
        // M, N, K
        // M number of columns of matrix op(A) and C.
        // N is number of rows of matrix op(B) and C.]
        // K is number of columns of op(B) and rows of op(A).

        // op(B) is N by K
        // op(A) is K by M
        // C is N by M
        // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
        // performs C=alpha op ( B ) op ( A ) + beta C
        gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_testing, 
          (compress == false) ? batch_size_CU : num_latent_factors,
          (float)1.0, V, U_testing, (float)0.0, R_testing);
        bool isBad = gpu_isBad<float>(R_testing, batch_size_testing * ratings_cols);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };


        if(Debug && 0){
          //checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * sizeof(float)));
          //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v3");
        }

        sparse_error(batch_size_testing, ratings_cols, R_testing, 
         csr_format_ratingsMtx_userID_dev_testing_batch, 
         coo_format_ratingsMtx_itemID_dev_testing,
         testing_entries, 
         coo_testing_errors, nnz, coo_R);
        isBad = gpu_isBad<float>(coo_testing_errors, nnz);
        if(isBad){
          ABORT_IF_NEQ(0, 1, "isBad");
        };

        gpu_hadamard<float>(nnz, testing_entries_cpy, coo_testing_errors );

        float training_error_temp = gpu_sum_of_squares<float>(nnz, coo_testing_errors);
        if( Debug)  LOG("gpu_R_error iteration : "<<i);
        long long int nnz_ = (long long int)((float)nnz * ((float)1.0 - testing_fraction));


        float temp = training_error_temp / (float)(nnz_);
        float epsilon = 0.00001;
        if(i > 1){
      //if(i % 1000 ==0) LOG("gpu_R_error iteration : "<<i);
      //have we stopped improving?
          if(error - temp < epsilon){
            if (abs(error - temp) > epsilon){
              LOG("gpu_R_error jumped over minimum iteration : "<<i);
          //we jumped over the minimum
              if(compress){
                checkCudaErrors(cudaMemcpy(U_testing, U_testing_check, batch_size_testing * num_latent_factors * sizeof(float), cudaMemcpyDeviceToDevice));
              }else{
                checkCudaErrors(cudaMemcpy(U_testing, U_testing_check, batch_size_testing * batch_size_CU * sizeof(float), cudaMemcpyDeviceToDevice));
              }
              training_rate = training_rate / (float)10.0;
            }else{
          //we've learned enough
              not_done = false;
            }

          }
        }
        error = temp;


        if( Debug && 0){
          LOG("gpu_R_error average error : "<< error); 
          //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

          float *testing_entries_temp; 
          checkCudaErrors(cudaMalloc((void**)&testing_entries_temp, nnz * sizeof(float)));
          checkCudaErrors(cudaMemcpy(testing_entries_temp, testing_entries, nnz * sizeof(float), cudaMemcpyDeviceToDevice));

          temp = gpu_sum_of_squares_of_diff(dn_handle, nnz, 
            coo_format_ratingsMtx_rating_dev_testing, 
            testing_entries_temp);
          LOG("gpu_R_error error normalized by should be max error: "<< training_error_temp / (float)(nnz_)<<std::endl); 
          //LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 


          std::string title = ("ratings_testing_errors_v" + ToString<int>(i)).c_str();
          save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, 
           coo_R, coo_testing_errors, nnz, "ratings_testing_errors_v");

          cudaFree(testing_entries_temp);
          // LOG("Press Enter to continue.") ;
          // std::cin.ignore();

        }



    //============================================================================================
    // (Update  U = U * (1 -alpha * lambda) + alpha * Error * V ) <- random error?????
    //============================================================================================ 
    /*
        m,n,k
        This function performs one of the following matrix-matrix operations:

        C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

        A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

        op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONJUCUTE_TRANSPOSE
        and

        op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
        array of dimensions (ldb, n) if op(B)=B, and (ldb, k) otherwise.

        n is the number of columns of dense matrix op(B) and C.
    */
        if(compress){
          checkCudaErrors(cudaMemcpy(U_testing_check, U_testing, batch_size_testing * num_latent_factors * sizeof(float), cudaMemcpyDeviceToDevice));
        }else{
          checkCudaErrors(cudaMemcpy(U_testing_check, U_testing, batch_size_testing * batch_size_CU * sizeof(float), cudaMemcpyDeviceToDevice));
        }


        beta = (float)1.0 - training_rate * regularization_constant;
        gpu_spXdense_MMM<float>(sp_handle, false, false, 
         batch_size_testing, 
         (compress == false) ? batch_size_CU : num_latent_factors, 
         ratings_cols,
         nnz, first_coo_ind, &training_rate, sp_descr, 
         coo_testing_errors, 
         csr_format_ratingsMtx_userID_dev_testing_batch, 
         coo_format_ratingsMtx_itemID_dev_testing,
         V, ratings_cols, &beta, U_testing, batch_size_testing, Debug);

        i += 1 ;
    } // end while

    if(Debug){
      save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v3");
    }


  sparse_error(batch_size_testing, ratings_cols, R_testing, 
   csr_format_ratingsMtx_userID_dev_testing_batch, 
   coo_format_ratingsMtx_itemID_dev_testing,
   coo_format_ratingsMtx_rating_dev_testing, 
   coo_testing_errors, nnz, coo_R);

  gpu_reverse_bools<float>(nnz,  testing_entries_cpy);
  gpu_hadamard<float>(nnz, testing_entries_cpy, coo_testing_errors );
  gpu_hadamard<float>(nnz, coo_format_ratingsMtx_rating_dev_testing, testing_entries_cpy );
  //save_device_arrays_side_by_side_to_file<float>(coo_testing_errors, testing_entries, nnz, "testing_entry_errors");

  float error_test = gpu_sum_of_squares<float>(nnz, coo_testing_errors);
  float mean_guess_error = gpu_sum_of_squares<float>(nnz, testing_entries_cpy);

  if(1) {
    //LOG("gpu_R_error call finished");
    LOG("gpu_R_error_testing total iterations : "<<i);
    LOG("gpu_R_error_testing MSQER on training entries: "<<error); 

    float nnz_ = (float)nnz * testing_fraction; 
    LOG("gpu_R_error_testing MSQER on testing entries: "<<error_test / nnz_); 

    float expected_dist_two_guassian = gpu_expected_dist_two_guassian<float>(dn_handle, nnz_);
    LOG("expected distance between two "<<nnz_<<" dimenstional guassian vectors : "<<expected_dist_two_guassian);

    LOG("Testing error norm over E[|N(0,1) - N(0,1)|]: "<< std::sqrt(error_test) / expected_dist_two_guassian ); 
    LOG("Testing error norm over norm of testing only entries: "<< std::sqrt(error_test) / std::sqrt(mean_guess_error) );    
  }

  //checkCudaErrors(cudaMemcpy(testing_entries, testing_entries_cpy, nnz * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaFree(testing_entries_cpy);
  if(Debug) {
    cudaFree(coo_R);
    //cudaFree(U_testing_check);
  }
  cudaFree(U_testing_check);
  cudaDeviceSynchronize();
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("gpu_R_error_testing run time : "<<readable_time(program_time)<<std::endl);
  testing_error_on_training_entries[0] += error;
  testing_error_on_testing_entries[0] += error_test;
  total_iterations[0] +=(long long int)i;
}





