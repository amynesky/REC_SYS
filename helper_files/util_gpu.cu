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

// number of blocks for threads.
long long int GET_BLOCKS(const long long int N, long long int num_threads = CUDA_NUM_THREADS) 
{
  return (N + num_threads - (long long int)1) / num_threads;
}

template <typename Dtype>
__device__ Dtype gpu_abs(Dtype val) {
  if(val < (Dtype)0.0){
    return ((Dtype)(-1.0) * val);
  }else{
    return val;
  }
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

  long long int num_gpu_blocks = GET_BLOCKS(size);

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = CUDA_NUM_BLOCKS*CUDA_NUM_THREADS;
    long long int spot = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_isBad_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(A + spot , (int)num_entries, isBad);
      num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_isBad_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A + spot , (int)(size - spot), isBad);
  }else{
    if(too_big(size) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    gpu_isBad_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A , (int)size, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    save_device_array_to_file(A, (int)size, "gpu_isBad_array");
  }
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
void print_gpu_mtx_entries(const Dtype* array, int lda, int sda, std::string title, bool transpose, std::string file_line)
{

  const Dtype *host_pointer;
  
  host_pointer=(const Dtype *)malloc(lda * sda * SIZE_OF(Dtype));
  checkErrors(host_pointer);
  checkCudaErrors(cudaMemcpy((void*)host_pointer, array, lda * sda * SIZE_OF(Dtype),cudaMemcpyDeviceToHost));

  std::string line = (title + " : \r\n").c_str();
    if(transpose){
      for( int j = 0; j < sda; j+= 1 ) {
        for( int i = 0; i < lda; i+= 1 ) {
          long long int index = (long long int)i  + ((long long int)j) * ((long long int)lda);
          if (i==lda-1){
            if (j==sda-1){
              line = (line + ToString<Dtype>(host_pointer[index])).c_str();
            }else{
              line = (line + ToString<Dtype>(host_pointer[index]) + "\r\n").c_str();
            };
          }else{
            line = (line + ToString<Dtype>(host_pointer[index]) + ", ").c_str();
          };
        };
      };
    }else{
     for( int i = 0; i < lda; i+= 1 ) {
      for( int j = 0; j < sda; j+= 1 ) {
        long long int index = (long long int)i  + ((long long int)j) * ((long long int)lda);
        if (j==sda-1){
          if (i==lda-1){
            line = (line + ToString<Dtype>(host_pointer[index])).c_str();
          }else{
            line = (line + ToString<Dtype>(host_pointer[index]) + "\r\n").c_str();
          };
        }else{
          line = (line + ToString<Dtype>(host_pointer[index]) + ", ").c_str();
        };
      };
    };     
  }

  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }

  free((void*)host_pointer);
  
}

template void print_gpu_mtx_entries<bool>(const bool* array, int lda, int sda, std::string title, bool transpose, std::string file_line);
template void print_gpu_mtx_entries<int>(const int* array, int lda, int sda, std::string title, bool transpose, std::string file_line);
template void print_gpu_mtx_entries<float>(const float* array, int lda, int sda, std::string title, bool transpose, std::string file_line);



template<typename Dtype>
void print_gpu_array_entries(const Dtype* array, int count, std::string file_line)
{
  const Dtype *host_pointer;
  
  host_pointer=(const Dtype *)malloc(count * SIZE_OF(Dtype));
  checkErrors(host_pointer);
  checkCudaErrors(cudaMemcpy((void*)host_pointer, array, count * SIZE_OF(Dtype),cudaMemcpyDeviceToHost));

  std::string line;
  line="[ ";
  for( int i = 0; i < count; i+= 1 ) {
    if (i==count-1){
      line = (line + ToString<Dtype>(host_pointer[i ])).c_str();
    }else{
      line = (line + ToString<Dtype>(host_pointer[i]) + ", ").c_str();
    };
  };     

  line = line+ " ];\n";
  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }
  free((void*)host_pointer); 
}

template void print_gpu_array_entries<int>(const int* array, int count, std::string file_line);
template void print_gpu_array_entries<float>(const float* array, int count, std::string file_line);

template<typename Dtype>
void save_device_array_to_file(const Dtype* A_dev, int count, std::string title, std::string file_line)
{
  Dtype* A_host  = NULL;
  A_host = (Dtype *)malloc(count *  SIZE_OF(Dtype)); 
  checkErrors(A_host);
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, count * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));

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
    entries.flush();
  };
  entries.close();
  if(file_line != ""){
    LOG2(file_line, "save_device_array_to_file "<< title << " "<< count <<" dimensional array");
  } 
  free(A_host);
}

template void save_device_array_to_file<int>(const int* A_dev, int count, std::string title, std::string file_line);
template void save_device_array_to_file<float>(const float* A_dev, int count, std::string title, std::string file_line);
template void save_device_array_to_file<double>(const double* A_dev, int count, std::string title, std::string file_line);

template<typename Dtype>
void append_device_array_to_file(const Dtype* A_dev, int count, std::string title)
{
  Dtype A_host[count];
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, count*SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
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
    entries.flush();   
  }
  entries.close();
}

template void append_device_array_to_file<int>(const int* A_dev, int count, std::string title);
template void append_device_array_to_file<float>(const float* A_dev, int count, std::string title);
template void append_device_array_to_file<double>(const double* A_dev, int count, std::string title);

template <>
void save_device_arrays_side_by_side_to_file<float>(const float* A_dev, const float* B_dev, int count, std::string title)
{

  float *A_host = NULL;
  float *B_host = NULL;
  A_host=(float *)malloc(count * SIZE_OF(float));
  B_host=(float *)malloc(count * SIZE_OF(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries/*<<"["<<i<<"] : "*/<<A_host[i ]<<", "<<B_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };
    entries.flush();
  };
  //LOG("file saved");
  entries.close();
  free(A_host);
  free(B_host);
}

void save_device_arrays_side_by_side_to_file(const int* A_dev, const float* B_dev, int count, std::string title)
{

  int *A_host = NULL;
  float *B_host = NULL;
  A_host=(int *)malloc(count * SIZE_OF(int));
  B_host=(float *)malloc(count * SIZE_OF(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries/*<<"["<<i<<"] : "*/<<A_host[i ]<<", "<<B_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };
    entries.flush();
  };
  //LOG("file saved");
  entries.close();
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
  A_host=(float *)malloc(count * SIZE_OF(float));
  B_host=(float *)malloc(count * SIZE_OF(float));
  C_host=(float *)malloc(count * SIZE_OF(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries/*<<"["<<i<<"] : "*/<<A_host[i ]<<", "<<B_host[i ]<<", "<<C_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };
    entries.flush();
  };
  entries.close();
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
  A_host=(int *)malloc(count * SIZE_OF(int));
  B_host=(int *)malloc(count * SIZE_OF(int));
  C_host=(float *)malloc(count * SIZE_OF(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));


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
    entries.flush();
  };
  entries.close();
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
  A_host=(float *)malloc(count * SIZE_OF(float));
  B_host=(float *)malloc(count * SIZE_OF(float));
  C_host=(float *)malloc(count * SIZE_OF(float));
  D_host=(float *)malloc(count * SIZE_OF(float));
  checkErrors(A_host);
  checkErrors(B_host);
  checkErrors(C_host);
  checkErrors(D_host);
  checkCudaErrors(cudaMemcpy(A_host, A_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C_host, C_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(D_host, D_dev, count * SIZE_OF(float), cudaMemcpyDeviceToHost));


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
    entries.flush();
  };
  entries.close();
  //LOG("file saved");
  free(A_host);
  free(B_host);
  free(C_host);
  free(D_host);
}

template<typename Dtype>
void save_device_mtx_to_file(const Dtype* A_dev, int lda, int sda, std::string title, bool transpose, std::string file_line)
{
  if(file_line != ""){
    LOG("Running save_device_mtx_to_file.");
  } 
  long long int total = ((long long int)lda) * ((long long int)sda);
  Dtype* A_host = NULL;
  A_host = (Dtype *)malloc(total * SIZE_OF(Dtype)); 
  checkErrors(A_host);
  // Dtype A_host[lda*sda];
  CUDA_CHECK(cudaMemcpy(A_host, A_dev, total * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  if(!transpose){
    for (int i = 0; i < lda; i++){
      //entries<<"***"<<i<<"***";
      for (int j = 0; j < sda; j++){
        long long int index = ((long long int)i) + ((long long int)j) * ((long long int)lda);
        entries<<A_host[index];
        if(j < sda - 1){
          //entries<<"\r\n";
          entries<<", ";
        }
      };
      if(i < lda - 1){
        //entries<<"; ";
        entries<<"\r\n";
      };
    }; 
    if(file_line != ""){
      LOG2(file_line, "save_device_mtx_to_file "<< title<< " "<<lda<<" by "<<sda<< "finished");
    }  
  }else{
    for (int j = 0; j < sda; j++){
      //entries<<"***"<<j<<"***";
      for (int i = 0; i < lda; i++){
        long long int index = ((long long int)i) + ((long long int)j) * ((long long int)lda);
        entries<<A_host[index];
        if(i < lda - 1){
          entries<<", ";
        };
      };
      if(j < sda - 1){
        entries<<"\r\n";
        //entries<<"; ";
      };
      entries.flush();
    }; 
    if(file_line != ""){
      LOG2(file_line, "save_device_mtx_to_file "<< title<<  " "<<sda<<" by "<<lda<< " after transposition"<< "finished");
    }    
  }
  entries.close();
  //entries<<"];";
  free(A_host);
}

template void save_device_mtx_to_file<int>(const int* A_dev, int lda, int sda, std::string title, bool transpose, std::string file_line);
template void save_device_mtx_to_file<float>(const float* A_dev, int lda, int sda, std::string title, bool transpose, std::string file_line);
template void save_device_mtx_to_file<double>(const double* A_dev, int lda, int sda, std::string title, bool transpose, std::string file_line);

//============================================================================================
// Add and Scale
//============================================================================================

template <typename Dtype>
__global__ void add_scalar_kernel(int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <typename Dtype>
void gpu_add_scalar(const long long int N, const Dtype alpha, Dtype* Y) 
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // NOLINT_NEXT_LINE(whitespace/operators)
  long long int num_gpu_blocks = GET_BLOCKS(N);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      add_scalar_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(num_entries, alpha, Y + spot);

      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    add_scalar_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>((int)(N - spot), alpha, Y+spot);
  }else{
    if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    add_scalar_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>((int)N, alpha, Y);
  }

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasScopy(dn_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(dn_handle, n, &alpha, y, 1));
}

template <>
void gpu_scale<double>(cublasHandle_t dn_handle, const long long int n, const double alpha, 
  const double *x, double* y) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    // This function scales copies a scaled version of the vector x by the scalar α  into the result y.
  CUBLAS_CHECK(cublasDcopy(dn_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(dn_handle, n, &alpha, y, 1));
}
template <>
void gpu_scale<float>(cublasHandle_t dn_handle, const long long int n, const float alpha, float *x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  // This function scales the vector x by the scalar α and overwrites it with the result.
  CUBLAS_CHECK(cublasSscal(dn_handle, n, &alpha, x, 1));
}

template <>
void gpu_scale<double>(cublasHandle_t dn_handle, const long long int n, const double alpha, double *x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
__global__ void gpu_get_rand_bools_kernel(curandState *state, Dtype *a, const int n, float probability_of_success, bool* isBad)
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //a[idx] = curand_uniform(&state[idx]);
  CUDA_KERNEL_LOOP(i, n){
    a[i] = (Dtype)(curand_uniform(&state[i]) < probability_of_success);
    // if (::isinf(a[i]) || ::isnan(a[i])){
    //   isBad[0] = true;
    // }
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  curandState *devState;  
  gpuErrchk(cudaMalloc((void**)&devState, n * SIZE_OF(curandState)));

  const unsigned long seed = cluster_seedgen();

  const long long int num_gpu_blocks= GET_BLOCKS(n);

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  initCurand<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, seed, (int)n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_get_rand_bools_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, x, (int)n, probability_of_success, isBad);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(devState));

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if ( isBad_host == true){
    save_device_array_to_file(x, (int)n, "gpu_reverse_bools_array");
    ABORT_IF_NEQ(0, 1, "gpu_reverse_bools given non bool array.") 
  };

}

template  void gpu_get_rand_bools<bool>(const long long int n,  bool* x, float probability_of_success) ;
template void gpu_get_rand_bools<float>(const long long int n,  float* x, float probability_of_success) ;

__global__ void gpu_mark_ACU_users_kernel(const int ratings_rows_ACU, const int ratings_rows, const int* x_dev, int* y, bool *isBad) 
{
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //a[idx] = curand_uniform(&state[idx]);
  CUDA_KERNEL_LOOP(i, ratings_rows_ACU){
    int temp = x_dev[ratings_rows - 1 - i];
    if(temp >= ratings_rows){
      isBad[0] = true;
    }
    y[x_dev[ratings_rows - 1 - i]] = 2;
  }
}

void gpu_mark_ACU_users(const int ratings_rows_ACU, const int ratings_rows, const int* x_host, int* y ) 
{
  LOG("gpu_mark_ACU_users called.")
  bool Debug = false;
  if(too_big(ratings_rows_ACU) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  ABORT_IF_LESS(ratings_rows, ratings_rows_ACU, "oops!");

  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows_ACU);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  int* x_dev = NULL;
  checkCudaErrors(cudaMalloc((void**)&x_dev, ratings_rows * SIZE_OF(int)));
  CUDA_CHECK(cudaMemcpy(x_dev, x_host, ratings_rows * SIZE_OF(int), cudaMemcpyHostToDevice));

  gpu_mark_ACU_users_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_ACU, ratings_rows, x_dev, y, isBad);
  if(Debug){
    save_device_array_to_file(y, ratings_rows, "y");
    save_device_array_to_file(x_dev, ratings_rows, "x_dev");
  }

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if ( isBad_host == true){
    save_device_array_to_file(y, ratings_rows, "y");
    save_device_array_to_file(x_dev, ratings_rows, "x_dev");
    ABORT_IF_NEQ(0, 1, "failure");
  };

  checkCudaErrors(cudaFree(x_dev));
  LOG("gpu_mark_ACU_users finished.")
}

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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  curandState *devState;  
  gpuErrchk(cudaMalloc((void**)&devState, n*SIZE_OF(curandState)));

  const unsigned long seed = cluster_seedgen();

  const long long int num_gpu_blocks = GET_BLOCKS(n);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  initCurand<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, seed, (int)n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpu_get_rand_groups_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(devState, (int)n, x, probability_of_success, num_groups);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(devState));

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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));
  const long long int num_gpu_blocks = GET_BLOCKS(n);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };
  gpu_reverse_bools_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(n, x,isBad);

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if ( isBad_host == true){
    save_device_array_to_file(x, (int)n, "gpu_reverse_bools_array");
    ABORT_IF_NEQ(0, 1, "gpu_reverse_bools given non bool array.") 
  };
}

template void gpu_reverse_bools<float>(const long long int n,  float* x); 


void gpu_rng_uniform(const long long int n, unsigned int* r) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
void gpu_axpy<float>(cublasHandle_t dn_handle, const long long int N, const float alpha, const float* X,
 float* Y) 
{
  if(too_big(N)) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
  if(too_big(N)) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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

template <>
void gpu_rng_gaussian<float>(const long long int n, const float mu, const float sigma, float* r, bool add, cublasHandle_t dn_handle) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));

  if((bool)(n % (long long int)2 !=  (long long int)0)) {
    LOG("n : "<< n);
    float* r_temp;
    checkCudaErrors(cudaMalloc((void**)&r_temp,  (n + (long long int)1) * SIZE_OF(float)));
    CURAND_CHECK(curandGenerateNormal(gen, r_temp, (n + (long long int)1), mu, sigma));
    CURAND_CHECK(curandDestroyGenerator(gen)); 
    if(add){
      gpu_axpy(dn_handle, n, (float)1.0, r_temp, r);
    }else{
      checkCudaErrors(cudaMemcpy(r,  r_temp, n * SIZE_OF(float), cudaMemcpyDeviceToDevice));
    } 
    checkCudaErrors(cudaFree(r_temp));
  }else{   
    if(add){
      float* r_temp;
      checkCudaErrors(cudaMalloc((void**)&r_temp,  n * SIZE_OF(float)));
      CURAND_CHECK(curandGenerateNormal(gen, r_temp, n, mu, sigma));
      CURAND_CHECK(curandDestroyGenerator(gen)); 
      gpu_axpy(dn_handle, n, (float)1.0, r_temp, r);
      checkCudaErrors(cudaFree(r_temp));
    }else{
      CURAND_CHECK(curandGenerateNormal(gen, r, n, mu, sigma));
      CURAND_CHECK(curandDestroyGenerator(gen)); 
    } 
  }

}

template <>
void gpu_rng_gaussian<double>(const long long int n, const double mu, const double sigma, double* r, bool add, cublasHandle_t dn_handle) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
  if(add) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cluster_seedgen()));

  if((bool)(n % (long long int)2 !=  (long long int)0)) {
    double* r_temp;
    checkCudaErrors(cudaMalloc((void**)&r_temp,  (n + (long long int)1) * SIZE_OF(double)));
    CURAND_CHECK(curandGenerateNormalDouble(gen, r_temp, (n + (long long int)1), mu, sigma));
    CURAND_CHECK(curandDestroyGenerator(gen));  
    checkCudaErrors(cudaMemcpy(r,  r_temp, n * SIZE_OF(double), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(r_temp));
  }else{
    CURAND_CHECK(curandGenerateNormalDouble(gen, r, n, mu, sigma));
    CURAND_CHECK(curandDestroyGenerator(gen));    
  }

}


template < typename Dtype>
void gpu_shuffle_array(cublasHandle_t handle, const long long int n,  Dtype* x)
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  float* order;
  CUDA_CHECK(cudaMalloc((void**)&order, n* SIZE_OF(float)));

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

  checkCudaErrors(cudaFree(order));

  // order__.clear();
  // r__.clear();
  // thrust::device_vector<Dtype>().swap(order__);
  // thrust::device_vector<Dtype>().swap(r__);
}

template void gpu_shuffle_array<float>(cublasHandle_t handle, const long long int n,  float* x);
template void gpu_shuffle_array<int>(cublasHandle_t handle, const long long int n,  int* x);

template<typename Dtype, typename Itype>
__device__ int device_partition (Dtype* x, int low_index, int high_index, Itype* indicies)
{
    // pivot (Element to be placed at right position)
    Dtype pivot = x[high_index];  
    Dtype temp = 0.0;
    Itype temp_ = 0;
    int i = (low_index - 1);  // Index of smaller element

    for (int j = low_index; j < high_index; j++)
    {
      // If current element is smaller than the pivot
      if (x[j] < pivot)
      {
        i++;    // increment index of smaller element
        temp = x[i];
        x[i] = x[j];
        x[j] = temp;
        temp_ = indicies[i];
        indicies[i] = indicies[j];
        indicies[j] = temp_;
      }
    }
    temp = x[i + 1];
    x[i + 1] = x[high_index];
    x[high_index] = temp;
    temp_ = indicies[i + 1];
    indicies[i + 1] = indicies[high_index];
    indicies[high_index] = temp_;
    return (i + 1);
}

/* low  --> Starting index,  high  --> Ending index */
template<typename Dtype, typename Itype>
__device__ void device_quickSort_by_key(Dtype* x, int low_index, int high_index, Itype* indicies, bool* isBad)
{
  bool debug = true;
  //ABORT_IF_NEQ(0, 1, "function not ready");
  if (low_index < high_index)
  {
    /* pi is partitioning index, arr[pi] is now
       at right place */
    int pi = device_partition<Dtype,Itype>(x, low_index, high_index, indicies);
    if(debug){
      if(low_index < 0 || low_index > 1316 ){
        isBad[0] = true;
      }
      if(high_index < 0 || high_index > 1316 ){
        isBad[0] = true;
      }
      if(pi < 0 || pi > 1316 ){
        isBad[0] = true;
      }
    }
    device_quickSort_by_key<Dtype,Itype>(x, low_index, pi - 1, indicies,isBad);  // Before pi
    device_quickSort_by_key<Dtype,Itype>(x, pi + 1, high_index, indicies,isBad); // After pi
  }
}

template<typename Dtype>
__global__ void gpu_sort_csr_colums_kernel(long long int start, int num, 
                                const int *csr_format_ratingsMtx_userID_dev,
                                int* coo_format_ratingsMtx_itemID_dev,
                                Dtype* coo_format_ratingsMtx_rating_dev, bool* isBad) 
{
  CUDA_KERNEL_LOOP(i, num) {
    long long int j = (long long int)i + start;
    int first_place = (csr_format_ratingsMtx_userID_dev[j]);
    int last_place = (csr_format_ratingsMtx_userID_dev[j + (long long int)1] - 1);
    device_quickSort_by_key<int,Dtype>(coo_format_ratingsMtx_itemID_dev, first_place, last_place, coo_format_ratingsMtx_rating_dev, isBad);
  }
}

template <>
void gpu_sort_csr_colums<float>(const long long int ratings_rows, 
                                const int *csr_format_ratingsMtx_userID_dev,
                                int* coo_format_ratingsMtx_itemID_dev,
                                float* coo_format_ratingsMtx_rating_dev, 
                                long long int num_entries_,
                                std::string preprocessing_path)
{
  if(1) LOG("called gpu_sort_csr_colums");
  bool debug = true;
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  int *csr_format_ratingsMtx_userID_host = NULL;
  if(debug){
    csr_format_ratingsMtx_userID_host  = (int *)malloc((ratings_rows + (long long int)1) * SIZE_OF(int));
    checkErrors(csr_format_ratingsMtx_userID_host);
    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host,  csr_format_ratingsMtx_userID_dev,  (ratings_rows + (long long int)1) *  SIZE_OF(int), cudaMemcpyDeviceToHost));
  }

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  long long int CUDA_NUM_BLOCKS_TEMP = CUDA_NUM_BLOCKS;
  long long int CUDA_NUM_THREADS_TEMP = CUDA_NUM_THREADS;
  if(0){
    if(debug) LOG(" changing CUDA_NUM_BLOCKS_TEMP, and CUDA_NUM_THREADS_TEMP values") ;
    CUDA_NUM_BLOCKS_TEMP = (long long int)1;
    CUDA_NUM_THREADS_TEMP = (long long int)1;
  }

  long long int num_gpu_blocks = GET_BLOCKS(ratings_rows, CUDA_NUM_THREADS_TEMP);

  if(debug){
    LOG("CUDA_NUM_BLOCKS_TEMP : "<<CUDA_NUM_BLOCKS_TEMP);
    LOG("CUDA_NUM_THREADS_TEMP : "<<CUDA_NUM_THREADS_TEMP);
  }

  if (num_gpu_blocks > CUDA_NUM_BLOCKS_TEMP){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS_TEMP * CUDA_NUM_THREADS_TEMP);
    long long int spot = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

    while (num_gpu_blocks > CUDA_NUM_BLOCKS_TEMP){
      if(debug){
        LOG("num_gpu_blocks : "<<num_gpu_blocks);
        LOG("num_loops : "<<num_loops);
        LOG("spot : "<<spot);
        LOG("num_entries : "<<num_entries);

        int first_place = (csr_format_ratingsMtx_userID_host[spot]);
        int last_place = (csr_format_ratingsMtx_userID_host[spot + num_entries]); 
        LOG("first_place : "<<first_place);
        LOG("last_place : "<<last_place - 1);
        LOG("num_entries : "<<last_place - first_place);
        save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev + spot, (int)num_entries + 1, preprocessing_path + "csr_format_ratingsMtx_userID_dev");
        save_device_array_to_file<int>(coo_format_ratingsMtx_itemID_dev + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_dev");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_dev");
        //checkCudaErrors(cudaDeviceSynchronize());
      }
      gpu_sort_csr_colums_kernel<float><<<CUDA_NUM_BLOCKS_TEMP, CUDA_NUM_THREADS_TEMP>>>(spot, (int)num_entries,
                                                                                        csr_format_ratingsMtx_userID_dev,
                                                                                        coo_format_ratingsMtx_itemID_dev,
                                                                                        coo_format_ratingsMtx_rating_dev, isBad);
      
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS_TEMP;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
      CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaFree(isBad));
      if ( isBad_host == true){
        ABORT_IF_NEQ(0, 1, "uh oh!") 
      };
      if(debug){
        //checkCudaErrors(cudaDeviceSynchronize());
        int first_place = (csr_format_ratingsMtx_userID_host[spot]);
        int last_place = (csr_format_ratingsMtx_userID_host[spot + num_entries]);
        save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev + spot, (int)num_entries + 1, preprocessing_path + "csr_format_ratingsMtx_userID_dev");
        save_device_array_to_file<int>(coo_format_ratingsMtx_itemID_dev + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_dev");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_dev");
        //checkCudaErrors(cudaDeviceSynchronize()); 
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("gpu_sort_csr_colums run time loop "<<num_loops<<" : "<<readable_time(program_time)<<std::endl);              
      }
    }
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_sort_csr_colums_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>(spot, (int)(ratings_rows - spot),
                                                                            csr_format_ratingsMtx_userID_dev,
                                                                            coo_format_ratingsMtx_itemID_dev,
                                                                            coo_format_ratingsMtx_rating_dev, isBad);
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if ( isBad_host == true){
      ABORT_IF_NEQ(0, 1, "uh oh!") 
    };
  }else{
    if(too_big(ratings_rows) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    if(debug){
      LOG("num_gpu_blocks : "<<num_gpu_blocks);
    }
    gpu_sort_csr_colums_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>((long long int)0, (int)ratings_rows,
                                                                            csr_format_ratingsMtx_userID_dev,
                                                                            coo_format_ratingsMtx_itemID_dev,
                                                                            coo_format_ratingsMtx_rating_dev, isBad);
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if ( isBad_host == true){
      ABORT_IF_NEQ(0, 1, "uh oh!") 
    };
  }  
  //if(debug) checkCudaErrors(cudaDeviceSynchronize());
  if(1) LOG("finished call to gpu_sort_csr_colums") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(1) LOG("gpu_sort_csr_colums run time : "<<readable_time(program_time)<<std::endl);
}

void gpu_sort_csr_colums_test()
{

  int num_rows = 1;
  int num_entries = 1317;

  int csr_format_ratingsMtx_userID_host[num_rows + 1] = {0, num_entries};
  int coo_format_ratingsMtx_itemID_host[num_entries] = {0, 8, 18, 35, 55, 59, 86, 94, 103, 109, 110, 125, 149, 152, 157, 159, 160, 162, 164, 167, 172, 179, 184, 207, 229, 230, 234, 248, 257, 265, 279, 281, 287, 315, 316, 343, 348, 352, 355, 376, 379, 392, 404, 433, 441, 456, 470, 473, 474, 479, 493, 499, 526, 545, 550, 552, 554, 557, 585, 586, 589, 591, 598, 607, 609, 672, 677, 732, 735, 740, 744, 749, 761, 777, 779, 783, 785, 787, 809, 831, 840, 857, 860, 884, 898, 902, 903, 907, 909, 911, 912, 919, 921, 922, 939, 952, 967, 968, 990, 1006, 1009, 1010, 1012, 1013, 1015, 1017, 1018, 1026, 1027, 1029, 1030, 1031, 1034, 1072, 1083, 1085, 1087, 1100, 1125, 1126, 1134, 1147, 1174, 1177, 1196, 1198, 1202, 1203, 1205, 1206, 1208, 1209, 1211, 1212, 1217, 1219, 1220, 1222, 1223, 1224, 1227, 1232, 1233, 1236, 1240, 1241, 1244, 1246, 1247, 1249, 1251, 1252, 1253, 1254, 1255, 1256, 1259, 1262, 1264, 1269, 1271, 1273, 1274, 1275, 1281, 1282, 1283, 1284, 1287, 1289, 1296, 1300, 1302, 1306, 1319, 1338, 1339, 1344, 1346, 1355, 1358, 1370, 1371, 1372, 1374, 1375, 1379, 1384, 1387, 1393, 1395, 1406, 1407, 1428, 1484, 1494, 1498, 1516, 1526, 1537, 1543, 1550, 1551, 1555, 1561, 1572, 1582, 1586, 1590, 1591, 1605, 1607, 1609, 1616, 1624, 1638, 1644, 1652, 1675, 1680, 1681, 1701, 1703, 1706, 1720, 1730, 1731, 1747, 1783, 1821, 1830, 1857, 1861, 1866, 1880, 1881, 1910, 1916, 1917, 1918, 1920, 1922, 1951, 1952, 1953, 1960, 1964, 1967, 1981, 1999, 2000, 2001, 2002, 2004, 2009, 2010, 2011, 2013, 2015, 2016, 2018, 2027, 2032, 2033, 2037, 2042, 2049, 2050, 2052, 2053, 2057, 2075, 2077, 2087, 2090, 2091, 2094, 2104, 2113, 2114, 2115, 2133, 2136, 2138, 2143, 2152, 2159, 2160, 2161, 2185, 2211, 2244, 2267, 2272, 2299, 2310, 2316, 2323, 2328, 2337, 2353, 2354, 2365, 2372, 2380, 2381, 2401, 2403, 2405, 2406, 2411, 2412, 2413, 2419, 2420, 2421, 2428, 2448, 2449, 2454, 2466, 2469, 2470, 2487, 2501, 2527, 2528, 2548, 2550, 2565, 2570, 2615, 2632, 2639, 2641, 2642, 2653, 2659, 2661, 2693, 2698, 2700, 2705, 2709, 2716, 2719, 2725, 2727, 2734, 2787, 2788, 2790, 2794, 2796, 2797, 2806, 2807, 2809, 2857, 2866, 2870, 2875, 2878, 2879, 2904, 2914, 2915, 2923, 2947, 2948, 2950, 2952, 2984, 2985, 2986, 2990, 2992, 2996, 3017, 3021, 3032, 3038, 3043, 3051, 3061, 3069, 3073, 3074, 3086, 3090, 3104, 3107, 3113, 3133, 3146, 3189, 3195, 3197, 3199, 3242, 3252, 3253, 3256, 3272, 3274, 3299, 3347, 3362, 3363, 3364, 3395, 3396, 3399, 3408, 3420, 3434, 3438, 3439, 3447, 3470, 3480, 3507, 3526, 3549, 3550, 3577, 3592, 3622, 3623, 3634, 3637, 3638, 3653, 3670, 3675, 3680, 3685, 3698, 3702, 3703, 3704, 3705, 3726, 3735, 3739, 3741, 3744, 3762, 3770, 3784, 3792, 3801, 3806, 3825, 3827, 3831, 3835, 3876, 3878, 3916, 3929, 3945, 3947, 3948, 3955, 3958, 3971, 3976, 3980, 3983, 4014, 4021, 4033, 4039, 4080, 4084, 4103, 4123, 4131, 4197, 4209, 4213, 4214, 4222, 4261, 4269, 4274, 4309, 4326, 4342, 4343, 4366, 4368, 4382, 4387, 4395, 4396, 4404, 4436, 4437, 4439, 4443, 4530, 4532, 4541, 4543, 4545, 4551, 4552, 4557, 4579, 4586, 4590, 4620, 4635, 4637, 4642, 4657, 4672, 4677, 4680, 4700, 4717, 4733, 4734, 4811, 4826, 4847, 4854, 4859, 4864, 4875, 4885, 4886, 4901, 4908, 4928, 4962, 4965, 4967, 4968, 4972, 4973, 4978, 4994, 5026, 5037, 5040, 5042, 5049, 5059, 5061, 5071, 5085, 5088, 5092, 5093, 5099, 5104, 5155, 5180, 5181, 5192, 5217, 5218, 5245, 5246, 5253, 5280, 5290, 5293, 5307, 5308, 5312, 5348, 5377, 5393, 5410, 5417, 5418, 5426, 5432, 5437, 5440, 5444, 5451, 5458, 5462, 5480, 5488, 5497, 5501, 5506, 5555, 5567, 5608, 5617, 5629, 5648, 5689, 5704, 5711, 5780, 5781, 5783, 5809, 5832, 5852, 5949, 5961, 5963, 5970, 5973, 5994, 6015, 6053, 6061, 6077, 6098, 6103, 6137, 6139, 6141, 6156, 6173, 6228, 6249, 6263, 6272, 6273, 6282, 6300, 6322, 6349, 6364, 6376, 6382, 6439, 6502, 6529, 6533, 6536, 6540, 6563, 6600, 6663, 6668, 6702, 6720, 6726, 6730, 6733, 6747, 6750, 6765, 6784, 6799, 6856, 6873, 6906, 6933, 6951, 6966, 6986, 6995, 7003, 7012, 7021, 7062, 7089, 7098, 7114, 7115, 7122, 7146, 7190, 7230, 7253, 7256, 7307, 7309, 7312, 7321, 7359, 7360, 7361, 7396, 7418, 7447, 7457, 7563, 7568, 7586, 7697, 7702, 7757, 7765, 7791, 7801, 7819, 7837, 7882, 7886, 7921, 7923, 7924, 7925, 7981, 8015, 8018, 8041, 8124, 8238, 8268, 8359, 8360, 8370, 8386, 8490, 8520, 8530, 8591, 8639, 8643, 8665, 8669, 8672, 8692, 8741, 8750, 8762, 8765, 8805, 8809, 8814, 8816, 8830, 8860, 8873, 8884, 8888, 8893, 8956, 8971, 8975, 8982, 8983, 8984, 8987, 25748, 25749, 25759, 25793, 25797, 25804, 25824, 25889, 25941, 26073, 26121, 26171, 26286, 26337, 26429, 26506, 26584, 26661, 26709, 26766, 26775, 26834, 26864, 26945, 27092, 27104, 27191, 27316, 27433, 27659, 27667, 27727, 27771, 27800, 27838, 30792, 30809, 30893, 31037, 31269, 31426, 31430, 31657, 31749, 31792, 31877, 31949, 32010, 32229, 32360, 32550, 32586, 32934, 33492, 33678, 33793, 33833, 33939, 34047, 34149, 34318, 34658, 35720, 37728, 37948, 40814, 41563, 41565, 41568, 41819, 41879, 42542, 42737, 43674, 43918, 44154, 44902, 44971, 45080, 45446, 45498, 45721, 48393, 48515, 48773, 49081, 49662, 49751, 49768, 49816, 50357, 50797, 50871, 51076, 52107, 52282, 52547, 52580, 52703, 52721, 52999, 53372, 53463, 54000, 54009, 54048, 54825, 54832, 55342, 55468, 56547, 56873, 58558, 58609, 58769, 58880, 59314, 59614, 60068, 60283, 60355, 61239, 61933, 63780, 65467, 65681, 66050, 67297, 68156, 68532, 68589, 68953, 69301, 69608, 69752, 69843, 70750, 71026, 71279, 71932, 71985, 72275, 72303, 72335, 72652, 72924, 72935, 73161, 73358, 73474, 74160, 74316, 74477, 74856, 75976, 75978, 76021, 76694, 76828, 77537, 77775, 77807, 77943, 78024, 78412, 78695, 78859, 79105, 79423, 79635, 79766, 80205, 80423, 80567, 80679, 80718, 80824, 80949, 81392, 81833, 82122, 82303, 82752, 83050, 83670, 83772, 84831, 84988, 86203, 86307, 86398, 86714, 86755, 87030, 87050, 87357, 88098, 88124, 88165, 88328, 89013, 89515, 89548, 89550, 89669, 89766, 89796, 89832, 89871, 90085, 90379, 90534, 90650, 90774, 90776, 91066, 91153, 91418, 91424, 91557, 91559, 91609, 91691, 91708, 91767, 91895, 92084, 92470, 92675, 93036, 93195, 93329, 93392, 93483, 93655, 93784, 94432, 94834, 95587, 95694, 95764, 96293, 96761, 96844, 97767, 97818, 97909, 97947, 97970, 98594, 98765, 98804, 99011, 99053, 99084, 99269, 99272, 99924, 100069, 100495, 100945, 101225, 101286, 101445, 101733, 101824, 101943, 101951, 102011, 102089, 102398, 102424, 102589, 102597, 103102, 103379, 103436, 103518, 103562, 103636, 103662, 103670, 103744, 103812, 104090, 104098, 104639, 104812, 105307, 105480, 105812, 106140, 106396, 106526, 106701, 106867, 107182, 107294, 107381, 107483, 108011, 108047, 108075, 108523, 109031, 109054, 109061, 109105, 109152, 109324, 109572, 109770, 110045, 110115, 110176, 110178, 110228, 110319, 110351, 110534, 110556, 110817, 110896, 111232, 111289, 111311, 112061, 112331, 112394, 112484, 112600, 112930, 112958, 113015, 113219, 113231, 113357, 113605, 113848, 113905, 114279, 114281, 114419, 114576, 115163, 115290, 115378, 115621, 115928, 116856, 116926, 116932, 116988, 117361, 117569, 117581, 117929, 118176, 118197, 118707, 118773, 118853, 118859, 119146, 119311, 119423, 119431, 119795, 120431, 120854, 120933, 121321, 121323, 122287, 123406, 124301, 124536, 124561, 125530, 127629, 128168, 128172, 128444, 128519, 128633, 128861, 129067, 130070, 130348, 130473, 130957, 130983, 131010, 1, 28, 31, 46, 49, 111, 150, 222, 252, 259, 292, 295, 317, 336, 366, 540, 588, 592, 652, 918, 923, 1008, 1035, 1078, 1079, 1088, 1089, 1096, 1135, 1192, 1195, 1197, 1199, 1200, 1207, 1213, 1214, 1216, 1218, 1221, 1239, 1242, 1245, 1248, 1257, 1258, 1260, 1261, 1265, 1277, 1290, 1303, 1320, 1332, 1347, 1349, 1357, 1369, 1373, 1386, 1524, 1583, 1749, 1847, 1919, 1966, 1993, 1996, 2020, 2099, 2117, 2137, 2139, 2142, 2172, 2173, 2192, 2193, 2252, 2287, 2290, 2541, 2627, 2643, 2647, 2663, 2682, 2691, 2715, 2760, 2761, 2803, 2871, 2917, 2943, 2946, 2958, 2967, 2999, 3029, 3036, 3080, 3152, 3264, 3437, 3475, 3478, 3488, 3498, 3888, 3931, 3995, 3996, 4010, 4026, 4104, 4127, 4132, 4225, 4305, 4445, 4466, 4570, 4719, 4753, 4877, 4895, 4910, 4914, 4940, 4979, 4992, 5025, 5038, 5039, 5145, 5170, 5539, 5678, 5796, 5815, 5897, 5951, 5998, 6092, 6241, 6332, 6501, 6538, 6753, 6754, 6773, 6806, 6833, 6887, 7000, 7044, 7045, 7152, 7163, 7246, 7386, 7388, 7437, 7448, 7453, 7481, 7756, 8367, 8481, 8506, 8635, 8689, 8960, 31695};
  float coo_format_ratingsMtx_rating_host[num_entries] = {0.177706, -0.637103, -0.637103, -0.637103, -1.17237, -0.637103, -0.637103, -1.146, -0.637103, -0.137473, -0.326808, -1.11048, -0.155476, -0.440858, 0.674579, -1.94879, 0.674579, -0.637103, -1.94879, -1.94879, -1.94879, 0.674579, -0.637103, -0.478419, 0.674579, -0.637103, -0.637103, 0.674579, -1.16705, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.462651, -0.0602304, -0.861401, -1.94879, -0.637103, 0.674579, -0.637103, -1.94879, 0.0303579, 0.674579, 0.674579, -0.274926, -0.486803, -1.94879, -0.637103, 0.0359569, -1.00097, 0.674579, 0.674579, 0.674579, -0.637103, -0.469645, 0.674579, 0.674579, 0.67458, -0.637103, -0.0297566, 0.674579, -0.637103, 0.674579, -1.94879, -0.528783, -0.637103, -1.28711, -0.293823, -0.637103, -0.125956, -0.644179, -0.637103, -1.19001, -0.670619, -0.637103, 0.674579, 3.29794, 0.0192334, -0.637103, -0.637103, -0.637103, 0.0426224, -0.251982, -0.274276, -0.0199693, -0.437627, -0.177032, -0.637103, -0.557999, -0.160624, 0.674579, -0.278596, -0.381082, -1.07516, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, -0.637103, -0.164661, 0.674579, -0.831809, -0.710015, -1.3044, 0.431595, -0.302686, -0.637103, -0.910546, -0.316508, -0.467197, -1.94879, 1.21624, -1.0206, -0.119223, -0.398339, -0.637103, -0.0200866, -1.94879, -1.94879, -0.193671, -0.247501, 0.674579, -0.117064, -0.637103, -0.4006, 0.0333618, -1.19973, -0.616105, -0.958377, -0.524093, -0.256905, -0.637103, -0.864431, -0.637103, 0.219958, 0.674579, -0.404372, -0.0602766, -0.272447, 0.473058, -0.673156, -0.426598, 0.674579, -0.844806, 0.674579, -1.94879, -0.637103, -0.623995, -0.280731, 0.674579, 0.67458, -0.215643, -0.637103, -0.0329604, 0.674579, 0.674579, -0.82628, -0.242388, -0.0664514, 0.362175, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, -0.637103, -0.0829305, -0.0604934, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.67458, -1.94879, -0.428624, -0.637103, -1.94879, -0.637103, -0.415069, -1.94879, -1.01571, -0.435386, -0.637103, -0.637103, -1.94879, -1.94879, -0.120456, 0.00949191, -0.637103, 0.674579, 0.674579, -0.148218, 0.674579, -0.637103, 0.170818, -0.637103, 0.182678, -1.94879, 0.674579, -0.637103, 0.191381, 0.153274, -0.637103, -0.637103, -0.57337, -0.637103, 0.674579, -0.637103, -1.06701, -0.328279, -0.637103, -0.535967, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, 0.307902, 0.674579, -0.102854, -0.254629, -0.0436821, -1.12417, -1.94879, 0.255385, 0.674579, -0.830422, -0.521523, -1.94879, -0.637103, -0.637103, -0.637103, -0.778003, 0.791021, 0.674579, 0.195782, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.406215, -1.94879, 0.143165, 0.674579, -0.0690882, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.366569, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.0551844, 0.67458, -1.27046, 0.129329, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, 0.0426347, -1.94879, -0.637103, -1.94879, -0.529162, -1.23139, -1.94879, -1.94879, -1.94879, -1.94879, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.754389, -0.637103, -0.637103, -0.701621, 0.374468, 0.674579, 0.674579, -0.637103, 2.04119, -0.0594538, 0.674579, 0.372803, -0.0742494, 0.674579, -0.637103, 0.651898, -0.244879, -0.637103, 0.674579, -0.916603, -0.862378, -0.917509, -1.08289, -0.172581, -0.637103, 0.674579, -0.315244, -0.637103, -0.637103, -0.0161588, -0.637103, -0.637103, 0.674579, 0.674579, -0.301266, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, 0.0358773, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -1.37229, 0.674579, -1.94879, -0.637103, -0.637103, -0.454708, -0.806231, -0.345035, 0.674579, -0.651668, -0.0316486, -0.637103, -0.637103, -0.253871, -0.637103, -1.16859, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -1.23996, -0.637103, 0.674579, 0.148997, -1.94879, 0.277109, 0.674579, -0.103048, -0.637103, -0.637103, -0.637103, 0.202765, -0.272621, -0.0360244, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.67458, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.939542, -1.2072, -0.637103, 0.251636, 0.674579, 0.395388, 0.674579, 0.674579, 0.674579, 2.1697, 0.205222, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.69779, 0.674579, 0.674579, 0.321634, 0.674579, 0.67458, -0.637103, 0.674579, -1.94879, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -1.52034, 0.946077, -1.94879, 0.674579, -0.222471, 0.674579, 0.674579, 0.674579, -1.41221, 0.674579, -0.637103, -1.94879, 0.674579, -0.637103, -0.00356942, 0.674579, 0.674579, -0.637103, -0.272464, 0.223686, -0.637103, -1.41276, -0.637103, 0.674579, -0.637103, -1.94879, 0.674579, -0.637103, -1.94879, -1.29488, -0.637103, 0.674579, 0.674579, -1.94879, 0.674579, -1.94879, 0.20477, -1.12233, 0.195866, -0.383621, 0.674579, 0.674579, 0.178412, -0.637103, 0.674579, -1.6561, -0.786677, -1.94879, -0.637103, 0.338086, -1.94879, -1.94879, -1.94879, -1.94879, -0.637103, -1.94879, -1.50875, 0.00240074, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, -1.94879, -0.189671, 0.34467, 0.674579, -1.31835, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.201652, -1.94879, -0.637103, -0.637103, -0.800606, -1.94879, 0.674579, 0.674579, 0.543248, 0.674579, -0.637103, -0.637103, -1.94879, 0.674579, -1.94879, -0.637103, -1.94879, -0.0160151, -1.94879, 0.674579, -0.182755, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.75268, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.318759, -0.637103, 0.674579, 0.67458, -0.637103, -1.36439, -0.0144304, 0.674579, -0.637103, -0.225503, 1.15468, 0.503029, -0.637103, -0.637103, 2.51199, -1.94879, -0.637103, -1.94879, -1.24546, -0.637103, 0.782777, -0.637103, -0.265776, 0.674579, -0.637103, -0.0997543, -1.94879, -0.637103, 0.345317, -0.637103, -0.637103, -0.637103, 0.448523, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.67458, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.210605, 0.674579, -0.637103, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.137635, -0.637103, 0.674579, -0.637103, 0.674579, 3.29794, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 1.39234, -1.94879, -0.637103, -0.198552, 0.674579, -0.547232, 0.674579, 0.330088, -0.451396, -0.637103, -1.94879, -1.39169, -0.637103, -1.94879, 0.67458, -0.348339, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -1.94879, -1.11002, 0.393678, 0.674579, 0.418379, -0.637103, -0.637103, 1.03817, -0.768724, 0.674579, 2.35354, -0.637103, 0.67458, -0.0504983, -0.637103, 0.0784256, 3.29794, 0.67458, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -1.94879, -0.637103, 0.674579, -0.637103, -0.0754003, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 3.29794, -1.05553, 0.674579, -0.637103, -0.637103, 0.674579, -0.460676, -0.637103, -0.637103, 3.29794, -0.637103, 3.29795, -0.637103, 0.674579, -1.94879, -1.43924, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 3.29794, 3.29794, -0.637103, 0.938161, 0.674579, 0.674579, -1.94879, -1.94879, -0.163584, -1.94879, -0.637103, 0.674579, 0.674579, 0.0920067, 3.29794, 0.674579, 0.674579, -0.637103, 3.29794, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.616435, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, 3.29794, -0.637103, -1.94879, -1.94879, 3.29794, 3.29794, -0.637103, 0.674579, 0.674579, 3.29794, 3.29794, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.17764, -0.637103, -0.637103, 0.674579, -0.637103, 0.074912, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.316765, -0.637103, 0.979924, 0.674579, 1.88818, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.764716, 0.674579, 0.674579, 1.98626, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, -1.94879, 0.674579, 0.674579, 0.674579, 0.674579, -0.271676, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 1.98626, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, -1.94879, 2.54423, -0.637103, 0.674579, -0.637103, 1.02233, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, -1.94879, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.67458, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.67458, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.67458, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 1.98626, 1.98626, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -1.94879, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, 3.29794, 0.674579, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 3.29794, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, 3.29794, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, 0.674579, 0.674579, -0.637103, 3.29794, 1.98626, -0.637103, 0.674579, 0.674579};

  int *csr_format_ratingsMtx_userID_dev = NULL;
  int* coo_format_ratingsMtx_itemID_dev = NULL;
  float* coo_format_ratingsMtx_rating_dev = NULL;
  checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev, (num_rows + 1) * SIZE_OF(int)));
  checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev,  num_entries * SIZE_OF(int)));
  checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev,  num_entries * SIZE_OF(float)));
  checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev,  csr_format_ratingsMtx_userID_host,  (num_rows + 1) * SIZE_OF(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev,  coo_format_ratingsMtx_itemID_host,  num_entries * SIZE_OF(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev,  coo_format_ratingsMtx_rating_host,  num_entries * SIZE_OF(float), cudaMemcpyHostToDevice));


  print_gpu_array_entries(csr_format_ratingsMtx_userID_dev, num_rows + 1);
  print_gpu_array_entries(coo_format_ratingsMtx_itemID_dev, num_entries);
  print_gpu_array_entries(coo_format_ratingsMtx_rating_dev, num_entries);

  gpu_sort_csr_colums<float>(num_rows, csr_format_ratingsMtx_userID_dev, coo_format_ratingsMtx_itemID_dev, coo_format_ratingsMtx_rating_dev);

  print_gpu_array_entries(csr_format_ratingsMtx_userID_dev, num_rows + 1);
  print_gpu_array_entries(coo_format_ratingsMtx_itemID_dev, num_entries);
  print_gpu_array_entries(coo_format_ratingsMtx_rating_dev, num_entries);
  checkCudaErrors(cudaDeviceSynchronize());
}

template <>
void gpu_sort_index_by_max<float>(cublasHandle_t handle, const long long int n,  float* x, float* indicies)
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
  if(too_big(cols) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  for(long long int i = (long long int)0; i < rows; i+=(long long int)1){
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
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      gpu_set_all_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>( x + spot, (int)num_entries, alpha);

      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_set_all_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x + spot, (int)(n - spot), alpha);
  }else{
    if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    gpu_set_all_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, (int)n, alpha);
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  long long int num_gpu_blocks = GET_BLOCKS(n);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_set_as_index_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, (int)n);

}

template <typename Dtype>
__device__ Dtype gpu_linear_helper(const long long int i, const long long int n, Dtype old_first,  Dtype old_last, Dtype new_first,  Dtype new_last) {
  Dtype m = (new_last - new_first) / (old_last - old_first);

  Dtype x = ( (Dtype) i) * ( (old_last - old_first) / (Dtype)(n - (long long int)1) )  +  old_first;

  return m * (x - old_last) + new_last;
}

template <typename Dtype>
__device__ Dtype gpu_specified_log(const long long int i, const long long int n, Dtype first,  Dtype last) {
  Dtype const_ = (Dtype)5.0;

  //return ( (Dtype)(-1.0 * (first - last) / logf(2)) * logf( ((Dtype)i / (Dtype)(n - 1)) + (Dtype)1.0) + first );
  //return ( first  - (expf( ((Dtype)i / (Dtype)(n - (long long int)1)) ) + (Dtype)1.0) * ( (first - last) / (expf(1) + (Dtype)1.0) ) );

  Dtype temp = gpu_linear_helper(i, n, (Dtype)0.0,  (Dtype)1.0, Dtype(-1.0) * const_,  const_);

  return ( first  + ( ( Dtype(-1.0) * expf(temp) + expf(Dtype(-1.0) * const_) ) / (Dtype(-1.0) * expf(const_) + expf(Dtype(-1.0) * const_)) ) * (last - first)   );

}

template<typename Dtype>
__global__ void gpu_set_as_func_of_index_kernel( Dtype* x, const long long int n, Dtype first,  Dtype last) 
{
  CUDA_KERNEL_LOOP(i, n) {
    //origin+x1+rows*y1
    x[i] = gpu_specified_log<Dtype>(i, n, first, last);
  };
}

template<typename Dtype>
void gpu_set_as_func_of_index(Dtype* x, const long long int n, Dtype first,  Dtype last) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  long long int num_gpu_blocks = GET_BLOCKS(n);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_set_as_func_of_index_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, n, first, last);
}

template void gpu_set_as_func_of_index<float>(float* x, const long long int n, float first, float last);


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
  if(too_big(rows*cols) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  long long int num_gpu_blocks = GET_BLOCKS(rows*cols);
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  gpu_set_as_index_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x, (int)rows, (int)cols);

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
    checkCudaErrors(cudaMalloc((void**)&x_dev, num_entries * SIZE_OF(int)));
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      checkCudaErrors(cudaMemcpy(x_dev, x_host + spot,  num_entries *  SIZE_OF(int), cudaMemcpyHostToDevice));
      gpu_set_as_index_host_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>( x_dev, spot, (int)num_entries, rows,cols);
      checkCudaErrors(cudaMemcpy(x_host + spot, x_dev,  num_entries *  SIZE_OF(int), cudaMemcpyDeviceToHost));
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    checkCudaErrors(cudaMemcpy(x_dev, x_host + spot,  (rows*cols - spot) *  SIZE_OF(int), cudaMemcpyHostToDevice));
    gpu_set_as_index_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x_dev, spot, (int)(rows*cols - spot), rows,cols);
    checkCudaErrors(cudaMemcpy(x_host + spot, x_dev,  (rows*cols - spot) *  SIZE_OF(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(x_dev));
  }else{
    if(too_big(rows*cols) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    int * x_dev;
    checkCudaErrors(cudaMalloc((void**)&x_dev, rows*cols * SIZE_OF(int)));

    checkCudaErrors(cudaMemcpy(x_dev, x_host,  rows*cols *  SIZE_OF(int), cudaMemcpyHostToDevice));
    gpu_set_as_index_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>( x_dev, 0, (int)(rows*cols), rows,cols);
    checkCudaErrors(cudaMemcpy(x_host , x_dev,  rows*cols *  SIZE_OF(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(x_dev));
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
  if(too_big(n) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float sum = thrust::reduce(X, X + n, (float)0., thrust::plus<float>());
  return sum;
}

template <>
float gpu_norm<float>(cublasHandle_t dn_handle, const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
  float s;
  CUBLAS_CHECK(cublasSnrm2(dn_handle, n, x, 1, &s));
  return s;
}

template <>
double gpu_norm<double>(cublasHandle_t dn_handle, const long long int n, const double* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
  double s;
  CUBLAS_CHECK(cublasDnrm2(dn_handle, n, x, 1, &s));
  return s;
}

template<typename Dtype> 
Dtype gpu_expected_value(const long long int n,  const Dtype* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  Dtype* y = (Dtype*) x;
  thrust::device_ptr<Dtype> X = thrust::device_pointer_cast(y);
  Dtype sum = thrust::reduce(X, X + n, (Dtype)0.0, thrust::plus<Dtype>());
  return (Dtype)(sum / (Dtype)n);
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
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  // setup arguments 
  abss<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = (float)0.0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s =  thrust::transform_reduce(X, X+n, unary_op, init, binary_op) ; 


  return (float)(s/(float)n);

}

// square<Dtype> computes the square of a number f(x) -> x*x 
template <typename Dtype> 
struct square 
{ 
  __host__ __device__ Dtype operator()(const Dtype& x) const {
   return x * x; 
 } 
};

template <>
float gpu_variance<float>(const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  float expt_ =  gpu_expected_value<float>(n, x);
  // setup arguments 
  square<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = (float)0.0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s =  thrust::transform_reduce(X, X + n, unary_op, init, binary_op) ; 

  //pow( base, exp )
  return (float)(s/(float)n - pow(expt_ , (float)2.0));
}


template<typename Dtype>
Dtype gpu_expected_dist_two_guassian(cublasHandle_t dn_handle, const long long int n)
{
  bool Debug = false;
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
  CUDA_CHECK(cudaMalloc((void**)&x, n * SIZE_OF(Dtype)));
  CUDA_CHECK(cudaMalloc((void**)&y, n * SIZE_OF(Dtype)));
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
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(y));
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));  
  if(0) LOG("gpu_expected_dist_two_guassian run time : "<<readable_time(program_time)<<std::endl);

  return sum / ((Dtype)trials);
}

template <typename Dtype>
__device__ void gpu_incremental_average(const long long int increment_index, Dtype* old_avg, Dtype new_val) {
  if(increment_index == (long long int)1){
    old_avg[0] = new_val;
  }else{
    old_avg[0] += (new_val - old_avg[0]) / ((Dtype)(increment_index));
  }
}


template<typename Dtype>
__global__ void gpu_msq_nonzero_kernel(const long long int n, const Dtype* x, Dtype* y, int* z) 
{
  int count_nonzero = 0;
  CUDA_KERNEL_LOOP(i, (int)n) {
    if(x[i] != (Dtype)0.0){
      count_nonzero++;
      gpu_incremental_average((long long int)(count_nonzero), y, (x[i] * x[i]));
    }
  }
  z[0] = (int)n - count_nonzero;
}

template<typename Dtype>
void gpu_msq_nonzero(const long long int n, const Dtype* x, Dtype* y, bool Debug) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

  Dtype* y_dev;
  int* z_dev;
  checkCudaErrors(cudaMalloc((void**)&y_dev, SIZE_OF(Dtype)));
  checkCudaErrors(cudaMalloc((void**)&z_dev, SIZE_OF(int)));
  y[0] = (Dtype)0.0;
  CUDA_CHECK(cudaMemcpy(y_dev, y, SIZE_OF(Dtype), cudaMemcpyHostToDevice));


  gpu_msq_nonzero_kernel<<<1, 1>>>( n, x, y_dev, z_dev);

  CUDA_CHECK(cudaMemcpy(y, y_dev, SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
  int z = 0;
  CUDA_CHECK(cudaMemcpy(&z, z_dev, SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
  if(Debug){
    LOG(z <<" out of "<< n<<" entries are zero in submitted vector.");
    LOG(((Dtype)z) / ((Dtype)n)  <<" of the entries are zero in submitted vector.");
  }
  cudaFree(y_dev);
  cudaFree(z_dev);
}

template void gpu_msq_nonzero<float>(const long long int n, const float* x, float* y, bool Debug); 


template<typename Dtype>
__global__ void gpu_mean_abs_nonzero_kernel(const long long int n, const Dtype* x, Dtype* y, long long int* z) 
{
  long long int count_nonzero = (long long int)0;
  CUDA_KERNEL_LOOP(i, (int)n) {
    if(x[i] != (Dtype)0.0){
      count_nonzero += (long long int)1;
      if(x[i] > (Dtype)0.0){
        gpu_incremental_average(count_nonzero, y, x[i]);
      }else{
        gpu_incremental_average(count_nonzero, y, ((Dtype)(-1.0)) * (x[i]));
      }
    }
  }
  z[0] = n - count_nonzero;
}

template<typename Dtype>
void gpu_mean_abs_nonzero(const long long int n, const Dtype* x, Dtype* y, bool Debug, std::string vect_name) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  if(Debug && 0){
    LOG("typeid(n) : "<< typeid(n).name());
    LOG("n : "<< n);
    LOG("(int)n : "<< (int)n);
    LOG("static_cast<int>(n) : "<< static_cast<int>(n));
  }
  
  Dtype* y_dev;
  long long int* z_dev;
  checkCudaErrors(cudaMalloc((void**)&y_dev, SIZE_OF(Dtype)));
  checkCudaErrors(cudaMalloc((void**)&z_dev, SIZE_OF(long long int)));
  y[0] = (Dtype)0.0;
  CUDA_CHECK(cudaMemcpy(y_dev, y, SIZE_OF(Dtype), cudaMemcpyHostToDevice));

  gpu_mean_abs_nonzero_kernel<<<1, 1>>>( (int)n, x, y_dev, z_dev);

  CUDA_CHECK(cudaMemcpy(y, y_dev, SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
  long long int z;
  CUDA_CHECK(cudaMemcpy(&z, z_dev, SIZE_OF(long long int), cudaMemcpyDeviceToHost));
  if(Debug){
    LOG(z <<" out of "<< n<<" entries are zero in "<<vect_name<<".");
    LOG( ((Dtype)z) / ((Dtype)n)  <<" of the entries are zero in "<<vect_name<<".");
  }
  cudaFree(y_dev);
  cudaFree(z_dev);
}

template void gpu_mean_abs_nonzero<float>(const long long int n, const float* x, float* y, bool Debug, std::string vect_name); 

template <>
float gpu_sum_of_squares<float>(const long long int n, const float* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  square<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = (float)0.0; 
  // compute norm 
  float* y = (float*) x;
  thrust::device_ptr<float> X = thrust::device_pointer_cast(y);
  float s = thrust::transform_reduce(X, X + (int)n, unary_op, init, binary_op) ; 
  return s/* /(float)n*/;
}


void gpu_sum_of_squares_test() 
{
  long long int n = (long long int)25595;
  float* x = NULL;
  checkCudaErrors(cudaMalloc((void**)&x, n * SIZE_OF(float)));
  gpu_rng_gaussian(n, (float)0.0, (float)1.0, x);
  LOG("~gpu_sum_of_squares_test~");
  LOG("n : "<< n);
  print_gpu_array_entries(x, (int)n);
  LOG("result : "<< gpu_sum_of_squares<float>(n, x));
  cudaFree(x);
}




template <>
void gpu_dot<float>(cublasHandle_t dn_handle, const long long int n, const float* x, const float* y,
  float* out) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasSdot(dn_handle, n, x, 1, y, 1, out));
}

template <>
void gpu_dot<double>(cublasHandle_t dn_handle, const long long int n, const double* x, const double* y,
  double * out) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasDdot(dn_handle, n, x, 1, y, 1, out));
}

template <>
void gpu_asum<float>(cublasHandle_t dn_handle, const long long int n, const float* x, float* y) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasSasum(dn_handle, n, x, 1, y));
}

template <>
void gpu_asum<double>(cublasHandle_t dn_handle, const long long int n, const double* x, double* y) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  CUBLAS_CHECK(cublasDasum(dn_handle, n, x, 1, y));
}



template <>
cublasStatus_t transpose<float>(cublasHandle_t handle, 
  long long int m, long long int n, const float *A, float *result)
{
  if(too_big(n) || too_big(m)) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  float one = 1;
  float zero = 0;
  cublasOperation_t trans = CUBLAS_OP_T;

  return cublasSgeam(handle, trans, trans, n, m, &one, A, m, &zero, A, m, result, n);
}

template <>
cublasStatus_t transpose<double>(cublasHandle_t handle,
 long long int m, long long int n, const double *A, double *result)
{
  if(too_big(n) || too_big(m)) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
    long long int num_gpu_blocks = GET_BLOCKS(lda * sda);

    ABORT_IF_NEQ(lda, sda, "Transpose in place for r!= c is more sofisticated than what is currently written"); 

    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));
    

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      long long int num_loops = 0;
      long long int num_entries = CUDA_NUM_BLOCKS*CUDA_NUM_THREADS;
      // LOG(INFO) << "num_entries: "<<num_entries;
      long long int spot = 0;
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
        transpose_in_place_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(lda, sda, A, spot, num_entries);

        num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
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

    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if ( isBad_host == true){
      ABORT_IF_NEQ(0, 1, "Tranpose_in_place broke") 
    };
  */
  float* result;
  CUDA_CHECK(cudaMalloc((void**)&result, lda * sda * SIZE_OF(float)));

  transpose<float>(handle, lda, sda, A, result);

  CUDA_CHECK(cudaMemcpy(A, result, lda * sda * SIZE_OF(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(result));

}

__global__ void gpu_swap_ordering_kernel(const long long int rows, const long long int cols, const float *A, 
 const bool row_major_ordering, float * result,
 const long long int start, const int total)
{
  //WRONG DO NOT USE
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
void gpu_swap_ordering<float>(const long long int rows, const long long int cols, float *A, const bool row_major_ordering)
{
  /*
    bool Debug = false;
    long long int num_gpu_blocks = GET_BLOCKS(rows * cols);
    float* result;
    CUDA_CHECK(cudaMalloc((void**)&result, rows * cols * SIZE_OF(float)));

    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      long long int num_loops = 0;
      long long int num_entries = CUDA_NUM_BLOCKS*CUDA_NUM_THREADS;
      // LOG(INFO) << "num_entries: "<<num_entries;
      long long int spot = 0;
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
        gpu_swap_ordering_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, spot, num_entries);

        num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
        num_loops+=(long long int)1;
        spot = num_loops * num_entries;
        if(Debug) LOG("starting index :"<<spot);
        if(Debug) LOG("total entries : "<<spot);
        if(Debug) LOG("next start : "<<spot + num_entries);

      };
      // spot is the number of entries done so far
      // total - (done) = left to go 
      gpu_swap_ordering_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, spot, rows * cols - spot - 1);
    }else{
      gpu_swap_ordering_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols, A, row_major_ordering, result, 0, rows * cols);
    };

    CUDA_CHECK(cudaMemcpy(A, result, rows * cols * SIZE_OF(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(result));
  */
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  std::cout<<"gpu_swap_ordering called..."<<std::endl;

  long long int total = rows * cols;
  float* A_copy = NULL;
  A_copy  = (float *)malloc(total * SIZE_OF(float));
  checkErrors(A_copy);
  checkCudaErrors(cudaMemcpy(A_copy,  A,  total  *  SIZE_OF(float), cudaMemcpyDeviceToHost));
  cpu_swap_ordering<float>(rows, cols, A_copy, row_major_ordering);
  checkCudaErrors(cudaMemcpy(A,  A_copy,  total  *  SIZE_OF(float), cudaMemcpyHostToDevice));

  free(A_copy);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("getData runtime: %f\n", program_time); 
  std::cout<<"finished call to gpu_swap_ordering in "<<program_time<< "ms"<<std::endl<<std::endl;
}

template <>
void gpu_gemv<float>(cublasHandle_t dn_handle, const bool TransA, 
  const long long int M, const long long int N, 
  const float alpha, const float* A, const float* x, const long long int inc_x,
 const float beta, float* y, const long long int inc_y) 
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
void gpu_gemv<double>(cublasHandle_t dn_handle, const bool TransA, 
  const long long int M, const long long int N, 
  const double alpha, const double* A, const double* x, const long long int inc_x,
  const double beta, double* y, const long long int inc_y) 
{
  cublasOperation_t cuTransA =
  (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(dn_handle, cuTransA, N, M, &alpha,
    A, N, x, inc_x, &beta, y, inc_y));
}

template<typename Dtype>
__global__ void gpu_hadamard_kernel(const int n, const Dtype* A, Dtype* B, bool* isBad) 
{
  //performs B = A * B
  CUDA_KERNEL_LOOP(i, n) {
  //origin+x1+rows*y1
    B[i]=A[i]*B[i];
    if(abs(B[i]) == (Dtype)0.0) B[i] = (Dtype)0.0;
    if (::isinf(B[i]) || ::isnan(B[i])){
      isBad[0] = true;
    };
  };
}

template<>
void gpu_hadamard<float>(const long long int n, const float* A, float* B ) 
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  //performs B = A * B
  long long int num_gpu_blocks = GET_BLOCKS(n);
  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //   ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };
  gpu_hadamard_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS>>>((int)n, A, B, isBad);
  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
}



template<>
float gpu_sum_of_squares_of_diff<float>(cublasHandle_t dn_handle, const long long int n, const float* x, float* y)
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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


template <typename Dtype>
__global__ void gpu_gemm_debug_kernel(const bool TransA,
 const bool TransB, const long long int M, const long long int N, const long long int K,
 const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
 const Dtype* C, Dtype* c) 
{
  CUDA_KERNEL_LOOP(one, 1){
    long long int row = N - (long long int)1;
    long long int col = M - (long long int)1;

    long long int lda = (TransA == false) ? K : M;
    long long int ldb = (TransB == false) ? N : K;
    c[one] = C[row + col * N] * beta;
    Dtype temp = (Dtype)0.0;
    Dtype a = (float)0.0;
    Dtype b = (float)0.0;
    for(long long int i = (long long int)0; i < K; i += (long long int)1){
      a = (TransA == false) ? A[i + col * lda] : A[col + i * lda];
      b = (TransB == false) ? B[row + i * ldb] : B[i + row * ldb];
      temp += a * b;
    }
    temp *= alpha;
    c[one] += temp;
  }

}

template <>
void gpu_gemm<float>(cublasHandle_t dn_handle, const bool TransA,
 const bool TransB, const long long int M, const long long int N, const long long int K,
 const float alpha, const float* A, const float* B, const float beta,
 float* C) 
{
  if(too_big(K * M) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  if(too_big(N * K) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  if(too_big(M * N) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  bool Debug = true;
  float c;
  float* c_dev;
  if(Debug){
    checkCudaErrors(cudaMalloc((void**)&c_dev,  SIZE_OF(float)));
    //LOG("calling gpu_gemm");
    gpu_gemm_debug_kernel<float><<<1, 1>>>(TransA, TransB, M, N, K, alpha, A, B, beta, C, c_dev);
    checkCudaErrors(cudaMemcpy(&c, c_dev, SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    //LOG("c : "<< c);
  }

  // Note that cublas follows fortran order.
  // column-major ordering marches down a complete column
  // before moving to the next column
  // thus the leading dimension is the number of ROWS 
  long long int lda = (TransA == false) ? K : M;
  long long int ldb = (TransB == false) ? N : K;
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
  if(Debug){
    long long int row = N - (long long int)1;
    long long int col = M - (long long int)1;
    float temp = c;
    float epsilon = (float)0.001;
    checkCudaErrors(cudaMemcpy(&c, C + (row + col * N), SIZE_OF(float), cudaMemcpyDeviceToHost));
    //LOG("C : "<< c);
    cudaFree(c_dev);
    if(c < temp - epsilon || c > temp + epsilon) {
      LOG("hand calculated value of C["<<row + col * N<<"] = "<<temp);
      LOG("cublas calculated value of C["<<row + col * N<<"] = "<<c);
      LOG("gpu_gemm is misbehaving");
      //ABORT_IF_EQ(0, 0,"gpu_gemm is misbehaving");
    }
  }
}

template <>
void gpu_gemm<double>(cublasHandle_t dn_handle, const bool TransA,
  const bool TransB, const long long int M, const long long int N, const long long int K,
  const double alpha, const double* A, const double* B, const double beta,
  double* C) 
{
  // Note that cublas follows fortran order.
  // column-major ordering marches down a complete column
  // before moving to the next column
  // thus the leading dimension is the number of ROWS 
  long long int lda = (TransA == false) ? K : M;
  long long int ldb = (TransB == false) ? N : K;
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
  checkCudaErrors(cudaMalloc((void**)&B, N * K * SIZE_OF(float)));
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

  checkCudaErrors(cudaFree(B));
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
  long long int m, long long int n, long long int k, int nnz, float *alpha, const cusparseMatDescr_t descrA, 
  const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
  float *B, long long int ldb, float *beta, float *C, long long int ldc)
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
  long long int m, long long int n, long long int k, int nnz, double *alpha, const cusparseMatDescr_t descrA, 
  const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
  double *B, long long int ldb, double *beta, double *C, long long int ldc)
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

  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows);

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
  checkCudaErrors(cudaDeviceSynchronize());
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
  const int* ratings_rows_by_group, bool set_only_first_group ) 
{

  CUDA_KERNEL_LOOP(row, ratings_rows){

    int r_ = 0;
    int csr_start = 0;
    int num_user = 0;
    int group = group_indicies[row];
    if(!set_only_first_group || group == 0){
      
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
      if(num_user == ratings_rows_ - 1) 
        csr_format_ratingsMtx_userID[num_user + 1] = csr_end;
      csr_format_ratingsMtx_userID[num_user] = csr_start;

      int i = csr_start;
      for(r_ = csr_format_ratingsMtx_userID_dev[row]; r_ < csr_format_ratingsMtx_userID_dev[row + 1]; r_++){
        coo_format_ratingsMtx_itemID[i] = coo_format_ratingsMtx_itemID_dev[r_];
        coo_format_ratingsMtx_rating[i]  = coo_format_ratingsMtx_rating_dev[r_]; 
        i++;
      }
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
  const int* ratings_rows_by_group, bool set_only_first_group) 
{
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  LOG("gpu_split_data called...");
  if(set_only_first_group){
    LOG("set only first group");
  }

  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows);

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
    ratings_rows_by_group, set_only_first_group);
  checkCudaErrors(cudaDeviceSynchronize());
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
                                          float* user_means_ACU,float* user_var_ACU, const long long int ratings_rows_ACU,
                                          const int* csr_format_ratingsMtx_userID_dev_ACU,
                                          const float* coo_format_ratingsMtx_rating_dev_ACU,
                                          bool* isBad)
{

    CUDA_KERNEL_LOOP(row, ratings_rows_testing + ratings_rows_training + ratings_rows_ACU){
        float m = 0;
        float v = 0;
        int count = 0;
        if(row < ratings_rows_training){
            for(long long int i = csr_format_ratingsMtx_userID_dev_training[row]; i < csr_format_ratingsMtx_userID_dev_training[row + (long long int)1]; i+=(long long int)1){
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
            if(row < ratings_rows_training + ratings_rows_testing){
                long long int r = row - ratings_rows_training;
                for(long long int i = csr_format_ratingsMtx_userID_dev_testing[r]; i < csr_format_ratingsMtx_userID_dev_testing[r + (long long int)1]; i+=(long long int)1){
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
            }else{
                long long int r = row - ratings_rows_training - ratings_rows_testing;
                for(long long int i = csr_format_ratingsMtx_userID_dev_ACU[r]; i < csr_format_ratingsMtx_userID_dev_ACU[r + (long long int)1]; i+=(long long int)1){
                    m += coo_format_ratingsMtx_rating_dev_ACU[i];
                    v += pow(coo_format_ratingsMtx_rating_dev_ACU[i],(float)2.0); 
                    count++;
                }
                user_means_ACU[r] = m / (float)count;
                user_var_ACU[r] = v / (float)count - pow(user_means_ACU[r], (float)2.0);
                if(user_var_ACU[r] <= (float)0.0) {
                  user_var_ACU[r] = (float)0.0;
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
                        float* user_means_ACU,float* user_var_ACU, const long long int ratings_rows_ACU,
                        const int* csr_format_ratingsMtx_userID_dev_ACU,
                        const float* coo_format_ratingsMtx_rating_dev_ACU)
{
  bool Debug = false;

  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  const int num_gpu_blocks = GET_BLOCKS(ratings_rows_testing + ratings_rows_training + ratings_rows_ACU);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  collect_user_means_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(user_means_training, user_var_training, ratings_rows_training,
                                                                  csr_format_ratingsMtx_userID_dev_training,
                                                                  coo_format_ratingsMtx_rating_dev_training,
                                                                  user_means_testing, user_var_testing, ratings_rows_testing,
                                                                  csr_format_ratingsMtx_userID_dev_testing,
                                                                  coo_format_ratingsMtx_rating_dev_testing,
                                                                  user_means_ACU, user_var_ACU, ratings_rows_ACU,
                                                                  csr_format_ratingsMtx_userID_dev_ACU,
                                                                  coo_format_ratingsMtx_rating_dev_ACU,
                                                                  isBad);

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    float ratings_rows_training_min = gpu_min<float>(ratings_rows_training, user_var_training);
    float ratings_rows_testing_min  = gpu_min<float>(ratings_rows_testing,  user_var_testing) ;
    float ratings_rows_ACU_min       = gpu_min<float>(ratings_rows_ACU,       user_var_ACU) ;
    LOG("ratings_rows_training_min :" <<ratings_rows_training_min);
    LOG("ratings_rows_testing_min :"  <<ratings_rows_testing_min);
    LOG("ratings_rows_ACU_min :"       <<ratings_rows_ACU_min);

    float ratings_rows_training_max = gpu_abs_max<float>(ratings_rows_training, user_var_training);
    float ratings_rows_testing_max  = gpu_abs_max<float>(ratings_rows_testing,  user_var_testing) ;
    float ratings_rows_ACU_max       = gpu_abs_max<float>(ratings_rows_ACU,       user_var_ACU) ;
    LOG("ratings_rows_training_max :" <<ratings_rows_training_max);
    LOG("ratings_rows_testing_max :"  <<ratings_rows_testing_max);
    LOG("ratings_rows_ACU_max :"       <<ratings_rows_ACU_max);


    save_device_array_to_file<float>(user_var_training, ratings_rows_training, "user_var_training");
    save_device_array_to_file<float>(user_var_testing, ratings_rows_testing, "user_var_testing");
    save_device_array_to_file<float>(user_var_ACU, ratings_rows_ACU, "user_var_ACU");
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}


__global__ void collect_user_means_kernel(float* user_means,float* user_var, const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  bool* isBad)
{

  CUDA_KERNEL_LOOP(row, ratings_rows){
    float m = (float)0.0;
    float v = (float)0.0;
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  collect_user_means_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(user_means, user_var, ratings_rows,
    csr_format_ratingsMtx_userID_dev,
    coo_format_ratingsMtx_rating_dev,
    isBad);

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  long long int row_skip = (long long int)csr_format_ratingsMtx_userID_dev_training[0];
  CUDA_KERNEL_LOOP(j, num){
    long long int row = (long long int)j + start;
    for(long long int i = csr_format_ratingsMtx_userID_dev_training[row]; i < csr_format_ratingsMtx_userID_dev_training[row + (long long int)1]; i+=(long long int)1){
      long long int col = coo_format_ratingsMtx_itemID_dev_training[i - row_skip];
      float val = coo_format_ratingsMtx_rating_dev_training[i - row_skip]; 
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS(ratings_rows_training);

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
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
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



  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}

__global__ void gpu_fill_training_mtx_if_kernel(const long long int ratings_rows_training, 
  const long long int ratings_cols_training, const bool row_major_ordering, 
  const int* csr_format_ratingsMtx_userID_dev_training,
  const int* coo_format_ratingsMtx_itemID_dev_training,
  const float* coo_format_ratingsMtx_rating_dev_training,
  const float* bools_,
  float* full_training_ratings_mtx, long long int start, int num,
  bool* isBad)
{
  long long int row_skip = (long long int)csr_format_ratingsMtx_userID_dev_training[0];
  CUDA_KERNEL_LOOP(j, num){
    long long int row = (long long int)j + start;
    for(long long int i = csr_format_ratingsMtx_userID_dev_training[row]; i < csr_format_ratingsMtx_userID_dev_training[row + (long long int)1]; i+=(long long int)1){
      if( bools_[i - row_skip] != (float)0.0 ){
        long long int col = coo_format_ratingsMtx_itemID_dev_training[i - row_skip];
        float val = coo_format_ratingsMtx_rating_dev_training[i - row_skip]; 
        if (::isinf(val) || ::isnan(val)){
          isBad[0] = true;
        };
        long long int index_ = (long long int)0;
        if(row_major_ordering){
          index_ = ratings_cols_training * row + col;
        }else{
          index_ = row + ratings_rows_training * col;
        };
        full_training_ratings_mtx[index_] = val;
      }
    }
  }
}


void gpu_fill_training_mtx_if(const long long int ratings_rows_training, 
  const long long int ratings_cols_training, 
  const bool row_major_ordering,
  const int* csr_format_ratingsMtx_userID_dev_training,
  const int* coo_format_ratingsMtx_itemID_dev_training,
  const float* coo_format_ratingsMtx_rating_dev_training,
  const float* bools_,
  float* full_training_ratings_mtx)
{
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS(ratings_rows_training);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = 0;
    long long int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
    long long int spot        = 0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
    gpu_fill_training_mtx_if_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(ratings_rows_training, ratings_cols_training, row_major_ordering, 
      csr_format_ratingsMtx_userID_dev_training,
      coo_format_ratingsMtx_itemID_dev_training,
      coo_format_ratingsMtx_rating_dev_training, bools_,
      full_training_ratings_mtx, spot, num_entries,
      isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops     += (long long int)1;
      spot           = num_loops * num_entries;
    };
    gpu_fill_training_mtx_if_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training, ratings_cols_training, row_major_ordering, 
      csr_format_ratingsMtx_userID_dev_training,
      coo_format_ratingsMtx_itemID_dev_training,
      coo_format_ratingsMtx_rating_dev_training, bools_,
      full_training_ratings_mtx, spot, ratings_rows_training - spot,
      isBad);
  }else{
    gpu_fill_training_mtx_if_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(ratings_rows_training, ratings_cols_training, row_major_ordering, 
      csr_format_ratingsMtx_userID_dev_training,
      coo_format_ratingsMtx_itemID_dev_training,
      coo_format_ratingsMtx_rating_dev_training, bools_,
      full_training_ratings_mtx, (long long int)0, ratings_rows_training,
      isBad);
  };



  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
      // has this user already rated this item?
      if(coo_format_ratingsMtx_itemID_dev_training[coo_index] == item) could_estimate = false;
      if(coo_format_ratingsMtx_itemID_dev_training[coo_index]  > item) break;
    }
    if (could_estimate){
      int   count  = 0;
      float rating = (float)0.0;
      for(int j = csr_format_ratingsMtx_userID_dev_training[user]; j < csr_format_ratingsMtx_userID_dev_training[user + 1]; j++){
        int other_item = coo_format_ratingsMtx_itemID_dev_training[j];
        int start_l = csr_format_keyWordMtx_itemID_dev[item];
        for(int k = csr_format_keyWordMtx_itemID_dev[other_item]; k < csr_format_keyWordMtx_itemID_dev[other_item + 1]; k++){
          int other_keyWord = coo_format_keyWordMtx_keyWord_dev[k];
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
              break;
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
    CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

    long long int num_gpu_blocks = GET_BLOCKS(ratings_rows_training * ratings_cols_training);

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
        num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
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



    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };

    if(Debug) LOG("finished supplmenting ratings mtx with content based information") ;
    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  /*if(Debug)*/ LOG("content based run time : "<<readable_time(program_time)<<std::endl);
}



template<typename Dtype>
__global__ void dense_nearest_row_kernel(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection,  
 Dtype* error, bool* isBad)
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  CUDA_KERNEL_LOOP(row_B,rows_B) {
    Dtype closest_A_row_dist = (Dtype)10000.0;
    int   closest_A_row      = 0;
    for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A++){
      Dtype temp = (Dtype)0.0;
      int count = 0;
      for(long long int col = (long long int)0; col < (long long int)cols; col++){
        count++;
        gpu_incremental_average<Dtype>((long long int)(count), &temp, pow(dense_mtx_A[row_A + col * (long long int)rows_A] - dense_mtx_B[(long long int)row_B + col * (long long int)rows_B],(Dtype)2.0));
        //temp += pow(dense_mtx_A[row_A + col * (long long int)rows_A] - dense_mtx_B[(long long int)row_B + col * (long long int)rows_B],(Dtype)2.0);
      }
      if(temp < closest_A_row_dist || row_A == 0){
        closest_A_row_dist = temp;
        closest_A_row      = (int)row_A;
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
void dense_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype* error) 
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  bool Debug = false;
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // if(coo_A == NULL){
  //   Debug = false;
  // }else{
  //   Debug = true;
  // }

  if(Debug) LOG("dense_nearest_row called");


  const long long int num_gpu_blocks = GET_BLOCKS(rows_B);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };


  if(Debug) {
    LOG("rows_A : "<< rows_A);
    LOG("cols : "<< cols);
    LOG("dense_nearest_row called");
    save_device_mtx_to_file<Dtype>(dense_mtx_A, rows_A, cols, "dense_mtx_A");
    save_device_mtx_to_file<Dtype>(dense_mtx_B, rows_B, cols, "dense_mtx_B");
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();
  };

  dense_nearest_row_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A, cols, dense_mtx_A, 
                            rows_B, dense_mtx_B, selection, error, isBad);
  if(Debug) {
    save_device_array_to_file<Dtype>(error, rows_B, "row_errors");
    save_device_array_to_file<int>(selection, rows_B, "nearest_row");
    LOG("Press Enter to continue.") ;
    std::cin.ignore();
  };    
  if(Debug && 0)LOG("Here!");
  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  if(Debug) LOG("dense_nearest_row call finished");
}

template void dense_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const float* dense_mtx_B, int* selection,  
 float* error);




template<typename Dtype>
__global__ void calculate_KM_error_and_update_kernel(const long long int start, const int num, 
                                                    const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                                    const int rows_B, const Dtype* dense_mtx_B, int* selection,  
                                                    Dtype alpha, Dtype lambda, bool* isBad)
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  CUDA_KERNEL_LOOP(index, num) {
    long long int row_A = ((long long int)index + start) % ((long long int) rows_A);
    long long int col = ((long long int)index + start) / ((long long int) rows_A);
    Dtype temp = (Dtype)0.0;
    int count = 0;
    for(long long int row_B = (long long int)0; row_B < (long long int)rows_B; row_A++){
      if(selection[row_B] == (int)row_A){
        temp += dense_mtx_B[row_B + col * (long long int)rows_B];
        count++;
      }
    }
    dense_mtx_A[row_A + col * (long long int)rows_A] = ((float)1.0 - alpha * lambda) * dense_mtx_A[row_A + col * (long long int)rows_A] + alpha * (temp / ((float)count));

    if (::isinf(dense_mtx_A[row_A + col * (long long int)rows_A]) || ::isnan(dense_mtx_A[row_A + col * (long long int)rows_A])){
      isBad[0] = true;
    };
  };
}

template <typename Dtype>
void calculate_KM_error_and_update(const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                   const int rows_B, const Dtype* dense_mtx_B, int* selection, 
                                   Dtype alpha, Dtype lambda) 
{
    // The rows in dense_mtx_B are closest to selection row in dense_mtx_A
    // Update  A = A * (1 -alpha * lambda) + alpha * (average row in dense_mtx_B in your row group)


    bool Debug = false;
    bool isBad_host = false;
    bool* isBad;
    CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
    CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

    if(Debug) LOG("calculate_KM_error_and_update called");


    long long int num_gpu_blocks = GET_BLOCKS(rows_A * cols);

    if(Debug) {
      LOG("rows_A : "<< rows_A);
      LOG("cols : "<< cols);
      LOG("calculate_KM_error_and_update called");
      save_device_mtx_to_file<Dtype>(dense_mtx_A, rows_A, cols, "dense_mtx_A");
      save_device_mtx_to_file<Dtype>(dense_mtx_B, rows_B, cols, "dense_mtx_B");
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    };
    if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      int num_loops = (long long int)0;
      int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
      int spot = (long long int)0;
      if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
      while (num_gpu_blocks > CUDA_NUM_BLOCKS){
          calculate_KM_error_and_update_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(spot, (int)num_entries, rows_A, cols, dense_mtx_A, 
                                rows_B, dense_mtx_B, selection, alpha, lambda, isBad);
        num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
        num_loops += (long long int)1;
        spot = num_loops * num_entries;
      };
          // spot is the number of entries done so far
          // total - (done) = left to go 
      calculate_KM_error_and_update_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(spot, (int)(rows_A * cols - spot), rows_A, cols, dense_mtx_A, 
                              rows_B, dense_mtx_B, selection, alpha, lambda, isBad);
    }else{
      if(too_big(rows_A * cols) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
      calculate_KM_error_and_update_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>((long long int)0, (int)(rows_A * cols), rows_A, cols, dense_mtx_A, 
                              rows_B, dense_mtx_B, selection, alpha, lambda, isBad);
    };

    if(Debug) {
      save_device_array_to_file<int>(selection, rows_B, "nearest_row");
      LOG("Press Enter to continue.") ;
      std::cin.ignore();
    };    
    if(Debug && 0)LOG("Here!");
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };
    if(Debug) LOG("calculate_KM_error_and_update call finished");
}

template void calculate_KM_error_and_update<float>(const int rows_A, const int cols, float* dense_mtx_A, 
                                                  const int rows_B, const float* dense_mtx_B, int* selection, 
                                                  float alpha, float lambda);


__global__ void center_ratings_kernel(const float* user_means, const float* user_var, const int ratings_rows,
    const int* csr_format_ratingsMtx_userID_dev,
    const float* coo_format_ratingsMtx_rating_dev,
    float* coo_format_ratingsMtx_row_centered_rating_dev,
    const float val_when_var_is_zero, bool* isBad) 
{

  CUDA_KERNEL_LOOP(row, ratings_rows){
    for(int i = csr_format_ratingsMtx_userID_dev[row]; i < csr_format_ratingsMtx_userID_dev[row + 1]; i++){
      float val = coo_format_ratingsMtx_rating_dev[i] - user_means[row];
      if(user_var[row] > (float)0.01){
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // float min_ = gpu_min<float>(num_entries, coo_format_ratingsMtx_rating_dev);
  // float max_ = gpu_abs_max<float>(num_entries, coo_format_ratingsMtx_rating_dev);

  // max_ - 

  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows);

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

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
    for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
      mean += X[(long long int)row + rows * i];
      std_dev += pow(X[(long long int)row + rows * i], (float)2.0);
    }
    mean /= (float)cols;
    std_dev /= (float)cols;
    std_dev = std_dev - pow(mean, (float)2.0);
    user_var[row] = std_dev;
    std_dev = sqrt(std_dev);
    user_means[row] = mean;

    if(std_dev == (float)0.0 ){
      for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
        X[(long long int)row + rows * i] = val_when_var_is_zero;
      } 
    }else{
      for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
        X[(long long int)row + rows * i] = (X[row + rows * i] - mean) / std_dev;
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // float min_ = gpu_min<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
  // float max_ = gpu_abs_max<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);

  // max_ - 

  const int num_gpu_blocks = GET_BLOCKS(rows);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  center_rows_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows, cols,  X, val_when_var_is_zero, user_means, user_var, isBad);


  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

}


__device__ long long int from_below_diag_to_whole_device(long long int below_diag_index, int dimension)
{

  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) 
    return (long long int)(-1);

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
  long long int return_val = (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;
  if( return_val < (long long int)0 || return_val >= (dimension * dimension)){
    return (long long int)(-1);
  }
  return return_val;
}

__device__ long long int from_below_diag_to_whole_device_faster(long long int below_diag_index, long long int dimension)
{
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index >= num_below_diag ){
    return (long long int)(-1);
  }

  long long int inverted_number = num_below_diag - below_diag_index;
  float n_float = ( (float)1.0 + sqrt((float)1.0 + (float)8.0 * (float)inverted_number) ) / (float)2.0;
  long long int n = (long long int)round(n_float);
  long long int row = (long long int)(-1);
  long long int col = (long long int)(0);

  long long int one_less = ((n - (long long int)1) * (n - (long long int)2)) / (long long int)2;
  long long int on_it = (n * (n - (long long int)1)) / (long long int)2;
  long long int one_more = (n * (n + (long long int)1)) / (long long int)2;

  if(one_less < inverted_number && inverted_number < on_it){
    col = dimension - n;
    row = dimension -  (inverted_number - one_less);
  }else if(inverted_number == on_it){
    col = dimension - n;
    row = col + (long long int)1;
  }else if(on_it < inverted_number && inverted_number < one_more){
    col = dimension - n - (long long int)1;
    row = dimension - (inverted_number - on_it);
  }else if(inverted_number == one_more){
    //return (long long int)(-7);
    col = dimension - n - (long long int)1;
    row = col + (long long int)1;
  }else if(inverted_number == one_less){
    //return (long long int)(-8);
    col = dimension - n + (long long int)1;
    row = col + (long long int)1;
  } else {
    return (long long int)(-2);
    // col = dimension - n + (long long int)1;
    // row = dimension - (inverted_number - (n * (n - (long long int)1)) / (long long int)2);
  }
  
  if( row == col){
    return (long long int)(-3); 
  }
  if( col < (long long int)0 || col >= dimension - (long long int)1){
    return (long long int)(-4);
  }
  if( row < (long long int)1 || row >= dimension){
    return (long long int)(-5);
  }
  long long int return_val = row + dimension * col;
  if( return_val < (long long int)0 || return_val >= (dimension * dimension)){
    return (long long int)(-6);
  }
  return return_val;
}

__device__ long long int from_whole_to_below_diag_device(long long int whole_index, long long int dimension)
{
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // float min_ = gpu_min<float>(num_entries, coo_format_ratingsMtx_rating_dev);
  // float max_ = gpu_abs_max<float>(num_entries, coo_format_ratingsMtx_rating_dev);

  // max_ - 
  if(too_big(ratings_rows * ratings_rows) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  const long long int num_gpu_blocks = GET_BLOCKS(ratings_rows * ratings_rows);

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

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  float* cosine_similarity_dev, const bool compare_values, bool* isBad ) 
{
  // assume that cosine_similarity_dev is stored in column major ordering (a column stays together in memory).
  CUDA_KERNEL_LOOP(entry, num){
    long long int below_index = (long long int)entry + start;
    long long int whole_index = from_below_diag_to_whole_device_faster(below_index, ratings_rows);
    if(whole_index < (long long int)0 || whole_index >= ratings_rows * ratings_rows){
      isBad[0] = true;
      if(whole_index < (long long int)0){
        cosine_similarity_dev[entry] = ((float)below_index) + abs(((float)whole_index) / ((float)100.0));
      }else{
        cosine_similarity_dev[entry] = ((float)below_index) + abs(((float)whole_index) / ((float)(ratings_rows * ratings_rows)));
      }
      return;
    }
    int user_i = (int)(whole_index % ratings_rows);
    int user_j = (int)(whole_index / ratings_rows);
    if(user_i == user_j){
      isBad[0] = true;
      cosine_similarity_dev[entry] = (float)(1.23456789);
    }else{
      int   count   = 0;
      float num_    = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      int start_j = csr_format_ratingsMtx_userID_dev[user_j];
      for(int i = csr_format_ratingsMtx_userID_dev[user_i]; i < csr_format_ratingsMtx_userID_dev[user_i + 1]; i++){
        int user_i_itemID = coo_format_ratingsMtx_itemID_dev[i];
        denom_i += pow(coo_format_ratingsMtx_rating_dev[i], (float)2.0) ;
        for(int j = start_j; j < csr_format_ratingsMtx_userID_dev[user_j + 1]; j++){
          int user_j_itemID = coo_format_ratingsMtx_itemID_dev[j];
          denom_j += pow(coo_format_ratingsMtx_rating_dev[j], (float)2.0) ; 
          if( user_i_itemID == user_j_itemID){
            count   += 1;
            num_    += coo_format_ratingsMtx_rating_dev[i] * coo_format_ratingsMtx_rating_dev[j] ;
            start_j = j + 1;
            break;
          }else if(user_i_itemID < user_j_itemID){
            start_j = j;
            denom_j -= pow(coo_format_ratingsMtx_rating_dev[j], (float)2.0) ;
            break;
          }
        }
      }
      if(compare_values){
        // make sure you've looked at all of user_j's ratings
        for(int j = start_j; j < csr_format_ratingsMtx_userID_dev[user_j + 1]; j++){
          denom_j += pow(coo_format_ratingsMtx_rating_dev[j], (float)2.0) ;
        }
      }
      float temp = (float)0.0;
      if(count > 0){
        if(compare_values){ //
          if(sqrtf(denom_i * denom_j) > (float)0.001 && gpu_abs<float>(num_) > (float)0.0001){
            temp = num_ / ( sqrtf(denom_i * denom_j) );// ( num_ / std::sqrt(denom_i) ) / std::sqrt(denom_j) ; 
            if(temp > (float)1.0) temp = (float)1.0;
            if(temp < (float)(-1.0)) temp = (float)(-1.0);
          }
        }else{
          float temp_i = (float)csr_format_ratingsMtx_userID_dev[user_i + 1] - (float)csr_format_ratingsMtx_userID_dev[user_i];
          float temp_j = (float)csr_format_ratingsMtx_userID_dev[user_j + 1] - (float)csr_format_ratingsMtx_userID_dev[user_j];
          temp = ( (float)count) /  sqrtf(temp_i * temp_j) ;          
        }
      }
      cosine_similarity_dev[entry] = temp;
      if (::isinf(temp) || ::isnan(temp)){
        isBad[0] = true;
      };
    }
  }
}


__global__ void get_cosine_similarity_host_kernel(const long long int start, 
  const int num, 
  const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_dev, const bool compare_values, 
  const int* top_N_most_sim_itemIDs_dev,
  const float* top_N_most_sim_item_similarity_dev, 
  const long long int ratings_cols, const int Top_N, 
  bool* isBad ) 
{
  // assume that cosine_similarity_dev is stored in column major ordering (a column stays together in memory).
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
      int   count_i   = 0;
      int   count_j   = 0;
      float num_    = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      int user_i_index = csr_format_ratingsMtx_userID_dev[user_i];
      int user_i_itemID = coo_format_ratingsMtx_itemID_dev[user_i_index];
      int user_j_index = csr_format_ratingsMtx_userID_dev[user_j];
      int user_j_itemID = coo_format_ratingsMtx_itemID_dev[user_j_index];
      for(int item = 0; item < (int)ratings_cols; item++)
      {
        float user_i_rating = (float)0.0;
        while(user_i_index < csr_format_ratingsMtx_userID_dev[user_i + 1] &&
              user_i_itemID < item)
        {
          user_i_index++;
          if(user_i_index < csr_format_ratingsMtx_userID_dev[user_i + 1])
            user_i_itemID = coo_format_ratingsMtx_itemID_dev[user_i_index];
        }
        //we're out of bounds or user_i_itemID >= item
        if(user_i_itemID == item){
          user_i_rating = coo_format_ratingsMtx_rating_dev[user_i_index];
        }else{
          //user_i have not rated the item, has user_i rated similar items?
          //walk through the items user_i HAS rated and calculate a weighted average of ratings for similar items
          int   count_micro   = 0;
          float num_micro    = (float)0.0;
          float denom_micro = (float)0.0;
          int start = 0;
          for(int i = csr_format_ratingsMtx_userID_dev[user_i]; i < csr_format_ratingsMtx_userID_dev[user_i + 1]; i++){
            int user_i_itemID_other = coo_format_ratingsMtx_itemID_dev[i];
            for(int k = start; k < Top_N; k++){
              int _other_similar_item_index = top_N_most_sim_itemIDs_dev[(long long int)k + (long long int)item * (long long int)Top_N];
              if( user_i_itemID_other == _other_similar_item_index){
                count_micro += 1;
                num_micro += coo_format_ratingsMtx_rating_dev[i] * top_N_most_sim_item_similarity_dev[(long long int)k + (long long int)item * (long long int)Top_N];
                denom_micro += top_N_most_sim_item_similarity_dev[(long long int)k + (long long int)item * (long long int)Top_N] ; 
                start = k + 1;
                break;
              }else if(user_i_itemID_other < _other_similar_item_index){
                start = k;
                break;
              }
            }
          }
          if(denom_micro != (float)0.0){
            user_i_rating = num_micro / denom_micro;
          }
        }
        
        float user_j_rating = (float)0.0;
        while(user_j_index < csr_format_ratingsMtx_userID_dev[user_j + 1] &&
              user_j_itemID < item)
        {
          user_j_index++;
          if(user_j_index < csr_format_ratingsMtx_userID_dev[user_j + 1])
            user_j_itemID = coo_format_ratingsMtx_itemID_dev[user_j_index];
        }
        if(user_j_itemID == item){
          user_j_rating = coo_format_ratingsMtx_rating_dev[user_j_index];
        }else{
          //user_j have not rated the item, has user_j rated similar items?
          //walk through the items user_j HAS rated and calculate a weighted average of ratings for similar items
          int   count_micro   = 0;
          float num_micro    = (float)0.0;
          float denom_micro = (float)0.0;
          int start = 0;
          for(int j = csr_format_ratingsMtx_userID_dev[user_j]; j < csr_format_ratingsMtx_userID_dev[user_j + 1]; j++){
            int user_j_itemID_other = coo_format_ratingsMtx_itemID_dev[j];
            for(int k = start; k < Top_N; k++){
              int _other_similar_item_index = top_N_most_sim_itemIDs_dev[(long long int)k + (long long int)item * (long long int)Top_N];
              if( user_j_itemID_other == _other_similar_item_index){
                count_micro   += 1;
                num_micro    += coo_format_ratingsMtx_rating_dev[j] * top_N_most_sim_item_similarity_dev[(long long int)k + (long long int)item * (long long int)Top_N] ;
                denom_micro += top_N_most_sim_item_similarity_dev[(long long int)k + (long long int)item * (long long int)Top_N] ; 
                start = k + 1;
                break;
              }else if(user_j_itemID_other < _other_similar_item_index){
                start = k;
                break;
              }
            }
          }
          if(denom_micro != (float)0.0){
            user_j_rating = num_micro / denom_micro;
          }
        }
        
        if(user_i_rating != (float)0.0){
          count_i   += 1;
          denom_i += pow(user_i_rating, (float)2.0) ;
        }
        if(user_j_rating != (float)0.0){
          denom_j += pow(user_j_rating, (float)2.0) ;
          count_j   += 1;
        }
        if(user_i_rating != (float)0.0 && user_j_rating != (float)0.0){
          count   += 1;
          num_    += user_i_rating * user_j_rating;          
        }
        
      }

      float temp = (float)0.0;
      if(count > 0){
        if(compare_values){ //
          if(sqrtf(denom_i * denom_j) > (float)0.001 && gpu_abs<float>(num_) > (float)0.0001){
            temp = num_ / sqrtf(denom_i * denom_j);
          }
        }else{
          temp = ( (float)count) / sqrtf( (float)(count_i) * (float)(count_j) );          
        }
        if (::isinf(temp) || ::isnan(temp)){
          isBad[0] = true;
          return;
        };
      }
      cosine_similarity_dev[entry] = temp;
    }
  }
}


void get_cosine_similarity_host_kernel_debug(const long long int start, 
  const int num, 
  const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_dev, const bool compare_values, 
  const int* top_N_most_sim_itemIDs_dev,
  const float* top_N_most_sim_item_similarity_dev, 
  const long long int ratings_cols, const int Top_N, 
  bool* isBad ) 
{

  LOG("called get_cosine_similarity_host_kernel_debug") ;
  int* csr_format_ratingsMtx_userID_host = NULL;
  csr_format_ratingsMtx_userID_host = (int *)malloc( (ratings_rows + (long long int)1) * SIZE_OF(int) );
  checkErrors(csr_format_ratingsMtx_userID_host);
  checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host, csr_format_ratingsMtx_userID_dev,  (ratings_rows + (long long int)1) *  SIZE_OF(int), cudaMemcpyDeviceToHost));

  LOG("csr_format_ratingsMtx_userID_host[ratings_rows] : "<< csr_format_ratingsMtx_userID_host[ratings_rows]) ;
  int* coo_format_ratingsMtx_itemID_host = NULL;
  float* coo_format_ratingsMtx_rating_host = NULL;
  float* cosine_similarity_host = NULL;
  coo_format_ratingsMtx_itemID_host = (int *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * SIZE_OF(int));
  coo_format_ratingsMtx_rating_host = (float *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * SIZE_OF(float));
  cosine_similarity_host = (float *)malloc(num * SIZE_OF(float));
  checkErrors(coo_format_ratingsMtx_itemID_host);
  checkErrors(coo_format_ratingsMtx_rating_host);
  checkErrors(cosine_similarity_host);
  
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host, coo_format_ratingsMtx_itemID_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host, coo_format_ratingsMtx_rating_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  SIZE_OF(float), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(cosine_similarity_host, cosine_similarity_dev,  num *  SIZE_OF(float), cudaMemcpyDeviceToHost));

  int* top_N_most_sim_itemIDs_host = NULL;
  float* top_N_most_sim_item_similarity_host = NULL;
  top_N_most_sim_itemIDs_host = (int *)malloc((long long int)Top_N * ratings_cols * SIZE_OF(int));
  top_N_most_sim_item_similarity_host = (float *)malloc((long long int)Top_N * ratings_cols  * SIZE_OF(float));
  checkErrors(top_N_most_sim_itemIDs_host);
  checkErrors(top_N_most_sim_item_similarity_host);
  checkCudaErrors(cudaMemcpy(top_N_most_sim_itemIDs_host, top_N_most_sim_itemIDs_dev, (long long int)Top_N * ratings_cols  *  SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(top_N_most_sim_item_similarity_host, top_N_most_sim_item_similarity_dev,  (long long int)Top_N * ratings_cols *  SIZE_OF(float), cudaMemcpyDeviceToHost));

  int entry = 0;
  //getRandIntsBetween(&entry , 0 , (int)num - 1, 1);
  
  for(entry = 14224; entry < num; entry++)
  {
    //LOG("entry : "<< (long long int)entry + start);
    long long int below_index = (long long int)entry + start;
    long long int whole_index = from_below_diag_to_whole_faster(below_index, ratings_rows);
    if(whole_index < (long long int)0 || whole_index >= ratings_rows * ratings_rows){
      LOG("Is BAD!");
      return;
    }
    int user_i = (int)(whole_index % ratings_rows);
    int user_j = (int)(whole_index / ratings_rows);
    //LOG("user_i : "<< user_i);
    //LOG("user_j : "<< user_j);
    if( user_i == user_j ){
      LOG("Is BAD!");
      return;
      //cosine_similarity_host[entry] = (float)1.0;
    }else{
      int   count   = 0;
      int   count_i   = 0;
      int   count_j   = 0;
      float num_    = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      int user_i_index = csr_format_ratingsMtx_userID_host[user_i];
      int user_i_itemID = coo_format_ratingsMtx_itemID_host[user_i_index];
      int user_j_index = csr_format_ratingsMtx_userID_host[user_j];
      int user_j_itemID = coo_format_ratingsMtx_itemID_host[user_j_index];

      int item = 0;
      //getRandIntsBetween(&item , 0 , (int)ratings_cols - 1, 1);
      for(item = 0; item < (int)ratings_cols; item++)
      {
        //LOG("item : "<< item );
        float user_i_rating = (float)0.0;
        int while_count = 0;
        while(user_i_index < csr_format_ratingsMtx_userID_host[user_i + 1] &&
              user_i_itemID < item)
        {
          user_i_index++;
          if(user_i_index < csr_format_ratingsMtx_userID_host[user_i + 1])
            user_i_itemID = coo_format_ratingsMtx_itemID_host[user_i_index];
        }
        //LOG("user_i_index : "<< user_i_index);
        //LOG("user_i_itemID : "<< user_i_itemID);
        // if(user_i_index < csr_format_ratingsMtx_userID_host[user_i + 1] - 1){
        //   LOG("next user_i_itemID : "<< coo_format_ratingsMtx_itemID_host[user_i_index + 1]);
        // }
        //we're out of bounds or user_i_itemID >= item
        if(user_i_itemID == item){
          user_i_rating = coo_format_ratingsMtx_rating_host[user_i_index];
          //LOG("user_i_rating : "<< user_i_rating);
        }else{
          //user_i have not rated the item, has user_i rated similar items?
          //walk through the items user_i HAS rated and calculate a weighted average of ratings for similar items
          int   count_micro   = 0;
          float num_micro    = (float)0.0;
          float denom_micro = (float)0.0;
          int start = 0;
          for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
            int user_i_itemID_other = coo_format_ratingsMtx_itemID_host[i];
            for(int k = start; k < Top_N; k++){
              int _other_similar_item_index = top_N_most_sim_itemIDs_host[(long long int)k + (long long int)item * (long long int)Top_N];
              if( user_i_itemID_other == _other_similar_item_index){
                //LOG("other similar item  : "<< user_i_itemID_other);
                count_micro += 1;
                num_micro += coo_format_ratingsMtx_rating_host[i] * top_N_most_sim_item_similarity_host[(long long int)k + (long long int)item * (long long int)Top_N];
                denom_micro += top_N_most_sim_item_similarity_host[(long long int)k + (long long int)item * (long long int)Top_N] ; 
                start = k + 1;
                break;
              }else if(user_i_itemID_other < _other_similar_item_index){
                start = k;
                break;
              }
            }
          }
          if(denom_micro != (float)0.0){
            user_i_rating = num_micro / denom_micro;
          }
          //LOG("user_i_rating : "<< user_i_rating);
        }
        
        
        float user_j_rating = (float)0.0;

        while(user_j_index < csr_format_ratingsMtx_userID_host[user_j + 1] &&
              user_j_itemID < item)
        {
          user_j_index++;
          if(user_j_index < csr_format_ratingsMtx_userID_host[user_j + 1])
            user_j_itemID = coo_format_ratingsMtx_itemID_host[user_j_index];
        }
        // LOG("user_j_index : "<< user_j_index);
        // LOG("user_j_itemID : "<< user_j_itemID);
        // if(user_i_index < csr_format_ratingsMtx_userID_host[user_i + 1] - 1){
        //   LOG("next user_j_itemID : "<< coo_format_ratingsMtx_itemID_host[user_j_index + 1]);
        // }
        if(user_j_itemID == item){
          user_j_rating = coo_format_ratingsMtx_rating_host[user_j_index];
          //LOG("user_j_rating : "<< user_j_rating);
        }else{
          //user_j have not rated the item, has user_j rated similar items?
          //walk through the items user_j HAS rated and calculate a weighted average of ratings for similar items
          int   count_micro   = 0;
          float num_micro    = (float)0.0;
          float denom_micro = (float)0.0;
          int start = 0;
          for(int j = csr_format_ratingsMtx_userID_host[user_j]; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
            int user_j_itemID_other = coo_format_ratingsMtx_itemID_host[j];
            for(int k = start; k < Top_N; k++){
              int _other_similar_item_index = top_N_most_sim_itemIDs_host[(long long int)k + (long long int)item * (long long int)Top_N];
              if( user_j_itemID_other == _other_similar_item_index){
                //LOG("other similar item  : "<< user_j_itemID_other);
                count_micro   += 1;
                num_micro    += coo_format_ratingsMtx_rating_host[j] * top_N_most_sim_item_similarity_host[(long long int)k + (long long int)item * (long long int)Top_N] ;
                denom_micro += top_N_most_sim_item_similarity_host[(long long int)k + (long long int)item * (long long int)Top_N] ; 
                start = k + 1;
                break;
              }else if(user_j_itemID_other < _other_similar_item_index){
                start = k;
                break;
              }
            }
          }
          if(denom_micro != (float)0.0){
            user_j_rating = num_micro / denom_micro;
          }
          //LOG("user_j_rating : "<< user_j_rating);
        }
        
        
        if(user_i_rating != (float)0.0){
          count_i   += 1;
          denom_i += pow(user_i_rating, (float)2.0) ;
        }
        if(user_j_rating != (float)0.0){
          denom_j += pow(user_j_rating, (float)2.0) ;
          count_j   += 1;
        }
        if(user_i_rating != (float)0.0 && user_j_rating != (float)0.0){
          count   += 1;
          num_    += user_i_rating * user_j_rating;          
        }
        
      } //end for loop on items

      float temp = (float)0.0;
      if(count > 0){
        if(compare_values){ //
          if(std::sqrt(denom_i * denom_j) > (float)0.001 && std::abs(num_) > (float)0.0001){
            temp = num_ / std::sqrt(denom_i * denom_j);
          }
        }else{
          temp = ( (float)count) / std::sqrt( (float)(count_i) * (float)(count_j) );          
        }
        if (::isinf(temp) || ::isnan(temp)){
          LOG("Is BAD!");
          return;
        };
      }
      LOG("cosine_similarity_host["<<entry<<"] : "<< temp);
      cosine_similarity_host[entry] = temp;
    }
  } // end for loop on cosine mtx entries
  LOG("num : "<< (long long int)num + start);
  free(csr_format_ratingsMtx_userID_host);
  free(coo_format_ratingsMtx_itemID_host);
  free(coo_format_ratingsMtx_rating_host);
  free(cosine_similarity_host);
  free(top_N_most_sim_itemIDs_host);
  free(top_N_most_sim_item_similarity_host);
}



void get_cosine_similarity_host_kernel_debug(const long long int start, 
  const int num, 
  const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_dev,
  const int* coo_format_ratingsMtx_itemID_dev,
  const float* coo_format_ratingsMtx_rating_dev,
  float* cosine_similarity_dev, bool compare_values, bool* isBad ) 
{
  LOG("called get_cosine_similarity_host_kernel_debug") ;
  int* csr_format_ratingsMtx_userID_host = (int *)malloc((ratings_rows +1)* SIZE_OF(int));
  checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host, csr_format_ratingsMtx_userID_dev,  (ratings_rows + (long long int)1) *  SIZE_OF(int), cudaMemcpyDeviceToHost));

  int* coo_format_ratingsMtx_itemID_host = (int *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * SIZE_OF(int));
  float* coo_format_ratingsMtx_rating_host = (float *)malloc(csr_format_ratingsMtx_userID_host[ratings_rows] * SIZE_OF(float));
  float* cosine_similarity_host = (float *)malloc(num * SIZE_OF(float));
  
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host, coo_format_ratingsMtx_itemID_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host, coo_format_ratingsMtx_rating_dev,  csr_format_ratingsMtx_userID_host[ratings_rows] *  SIZE_OF(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(cosine_similarity_host, cosine_similarity_dev,  num *  SIZE_OF(float), cudaMemcpyDeviceToHost));

  for(int entry = 0; entry < num; entry++){
    long long int below_index = (long long int)entry + start;
    long long int whole_index = from_below_diag_to_whole_faster(below_index, ratings_rows);
  
    if(whole_index < (long long int)0 || whole_index >= ratings_rows * ratings_rows){
      LOG("below_index : "<<below_index);
      LOG("whole_index : "<<whole_index);
      LOG("max whole index =  : "<<ratings_rows * ratings_rows - (long long int)1);

      LOG("Is BAD!");
      whole_index = from_below_diag_to_whole(below_index, ratings_rows);
      LOG("slow whole_index : "<<whole_index);
      return;
    }
    int user_i = (int)(whole_index % ratings_rows);
    int user_j = (int)(whole_index / ratings_rows);

    if( user_i == user_j){
      LOG("Is BAD because user_i == user_j");
      LOG("below_index : "<<below_index);
      LOG("whole_index : "<<whole_index);
      LOG("user_i : "<<user_i);
      LOG("user_j : "<<user_j);
      return;
    }else{
      int   count   = 0;
      float num_    = (float)0.0;
      float denom_i = (float)0.0;
      float denom_j = (float)0.0;
      int start_j = csr_format_ratingsMtx_userID_host[user_j];
      for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
        int user_i_itemID = coo_format_ratingsMtx_itemID_host[i];
        denom_i += pow(coo_format_ratingsMtx_rating_host[i], (float)2.0) ;
        for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
          int user_j_itemID = coo_format_ratingsMtx_itemID_host[j];
          denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ; 
          if( user_i_itemID == user_j_itemID){
            count   += 1;
            num_    += coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j] ;
            start_j = j + 1;
            break;
          }else if(user_i_itemID < user_j_itemID){
            start_j = j;
            denom_j -= pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
            break;
          }
        }
      }
      if(compare_values){
        // make sure you've looked at all of user_j's ratings
        for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
          denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
        }
      }
      float temp = (float)0.0;
      if(count > 0){ //
        if(compare_values){ //
          if(std::sqrt(denom_i * denom_j) > (float)0.001 && std::abs(num_) > (float)0.0001){
            temp = num_ / ( std::sqrt(denom_i * denom_j) );// ( num_ / std::sqrt(denom_i) ) / std::sqrt(denom_j) ; 
            if(temp > (float)1.0) temp = (float)1.0;
            if(temp < (float)(-1.0)) temp = (float)(-1.0);
          }
        }else{
          float temp_i = (float)csr_format_ratingsMtx_userID_host[user_i + 1] - (float)csr_format_ratingsMtx_userID_host[user_i];
          float temp_j = (float)csr_format_ratingsMtx_userID_host[user_j + 1] - (float)csr_format_ratingsMtx_userID_host[user_j];
          temp =  ((float)count) /  std::sqrt(temp_i * temp_j) ;          
        }
      }
      cosine_similarity_host[entry] = temp;

      if (::isinf(temp) || ::isnan(temp)  ){
        LOG("Is BAD value is INF or NAN");
        LOG("below_index : "<<below_index);
        LOG("whole_index : "<<whole_index);
        LOG("user_i : "<<user_i);
        LOG("user_j : "<<user_j);
        LOG("cosine_similarity_host[entry] : "<<temp);
        LOG(" ");

        LOG("user_i num ratings : "<<csr_format_ratingsMtx_userID_host[user_i + 1] - csr_format_ratingsMtx_userID_host[user_i]);
        LOG("user_j num ratings : "<<csr_format_ratingsMtx_userID_host[user_j + 1] - csr_format_ratingsMtx_userID_host[user_j]);
        int   count   = 0;
        float num_    = (float)0.0;
        float denom_i = (float)0.0;
        float denom_j = (float)0.0;
        int start_j = csr_format_ratingsMtx_userID_host[user_j];
        for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
          int user_i_itemID = coo_format_ratingsMtx_itemID_host[i];
          LOG("user_i_itemID : "<<user_i_itemID);
          LOG("denom_i += : "<<ToString<float>(pow(coo_format_ratingsMtx_rating_host[i], (float)2.0)));
          denom_i += pow(coo_format_ratingsMtx_rating_host[i], (float)2.0) ;
          for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
            int user_j_itemID = coo_format_ratingsMtx_itemID_host[j];
            LOG("  user_j_itemID : "<<user_j_itemID);
            LOG("  denom_j += : "<<ToString<float>(pow(coo_format_ratingsMtx_rating_host[j], (float)2.0)));
            denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ; 
            if( user_i_itemID == user_j_itemID){
              count   += 1;
              LOG("    num_ += : "<<ToString<float>(coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j]));
              num_    += coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j] ;
              start_j = j + 1;
              break;
            }else if(user_i_itemID < user_j_itemID){
              start_j = j;
              denom_j -= pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
              break;
            }
          }
        }
        if(compare_values){
          // make sure you've looked at all of user_j's ratings
          for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
            denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
          }
        }
        float temp_ = (float)0.0;
        if(count > 0){ 
          LOG("here");
          if(compare_values){ //
            if(std::sqrt(denom_i * denom_j) > (float)0.001 && std::abs(num_) > (float)0.0001){
              LOG("num_ : "<<ToString<float>(num_));
              LOG("std::abs(num_) > (float)0.0001 : "<<(std::abs(num_) > (float)0.0001));
              LOG("std::abs(num_) : "<<ToString<float>(std::abs(num_)) );
              LOG("denom_i : "<<ToString<float>(denom_i));
              LOG("denom_j : "<<ToString<float>(denom_j));
              temp_ = num_ / ( std::sqrt(denom_i * denom_j) );// ( num_ / std::sqrt(denom_i) ) / std::sqrt(denom_j) ; 
              if(temp_ > (float)1.0) temp_ = (float)1.0;
              if(temp_ < (float)(-1.0)) temp_ = (float)(-1.0);
            }
          }else{
            float temp_i = (float)csr_format_ratingsMtx_userID_host[user_i + 1] - (float)csr_format_ratingsMtx_userID_host[user_i];
            float temp_j = (float)csr_format_ratingsMtx_userID_host[user_j + 1] - (float)csr_format_ratingsMtx_userID_host[user_j];
            temp_ =  ((float)count) /  std::sqrt(temp_i * temp_j) ;          
          }
        }
        LOG("temp : "<<temp_);

        return;
      };
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
  float* cosine_similarity_host, 
  bool compare_values,
  const int* top_N_most_sim_itemIDs_dev,
  const float* top_N_most_sim_item_similarity_dev, 
  const long long int ratings_cols, const int Top_N)
 {

  bool Debug = false;
  bool print = true;
  std::string blank = "";


  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  if(print) LOG("called get_cosine_similarity_host") ;

  if(top_N_most_sim_itemIDs_dev){
    if(top_N_most_sim_itemIDs_dev == NULL) top_N_most_sim_itemIDs_dev = NULL;
    if(ratings_cols <= (long long int)0) top_N_most_sim_itemIDs_dev = NULL;
    if(Top_N <= (long long int)0) top_N_most_sim_itemIDs_dev = NULL;
  }
  if(top_N_most_sim_itemIDs_dev){
    Debug = true;
    LOG("Debug bool : " <<Debug) ;
    LOG("top_N_most_sim_itemIDs_dev bool : " <<(top_N_most_sim_itemIDs_dev != NULL)) ;
  }
  if(print) {
    LOG("compare_values bool : " <<compare_values) ;
  }
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;

  long long int CUDA_NUM_BLOCKS_TEMP = CUDA_NUM_BLOCKS;
  long long int CUDA_NUM_THREADS_TEMP = CUDA_NUM_THREADS;
  if(top_N_most_sim_itemIDs_dev != NULL){
    if(Debug) LOG(" changing CUDA_NUM_BLOCKS_TEMP, and CUDA_NUM_THREADS_TEMP values") ;
    CUDA_NUM_BLOCKS_TEMP = (long long int)1000;
    CUDA_NUM_THREADS_TEMP = (long long int)100;
  }

  //long long int num_gpu_blocks = GET_BLOCKS(ratings_rows * ratings_rows);
  long long int num_gpu_blocks = GET_BLOCKS(num_below_diag, CUDA_NUM_THREADS_TEMP);

  if(Debug) {
    LOG("num_below_diag : " <<num_below_diag) ;
    LOG("CUDA_NUM_BLOCKS_TEMP : " <<CUDA_NUM_BLOCKS_TEMP) ;
    LOG("CUDA_NUM_THREADS_TEMP : " <<CUDA_NUM_THREADS_TEMP) ;
    LOG( "total loops : " <<ceil( ((float)(num_below_diag)) / ((float)(CUDA_NUM_BLOCKS_TEMP * CUDA_NUM_THREADS_TEMP)) ) ) ;
  }  

  

  if (num_gpu_blocks > CUDA_NUM_BLOCKS_TEMP){
    long long int num_loops = (long long int)0;
    const long long int num_entries = (long long int)(CUDA_NUM_BLOCKS_TEMP * CUDA_NUM_THREADS_TEMP);
    long long int spot = (long long int)0;

    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_entries * SIZE_OF(float)));
    if(Debug) {
      LOG("cosine_similarity_dev allocated") ;
      gpu_set_all<float>(cosine_similarity_dev, num_entries, (float)0.0);
    }

    while (num_gpu_blocks > CUDA_NUM_BLOCKS_TEMP){
      if(Debug && top_N_most_sim_itemIDs_dev) {
        LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
        LOG("num_loops : " <<num_loops) ;
        LOG("spot : " <<spot) ;
        LOG("num_entries : " <<num_entries) ;
        //save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
      }
      //checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  num_entries *  SIZE_OF(float), cudaMemcpyHostToDevice));
      //checkCudaErrors(cudaDeviceSynchronize());
      if(top_N_most_sim_itemIDs_dev){
        get_cosine_similarity_host_kernel<<<CUDA_NUM_BLOCKS_TEMP, CUDA_NUM_THREADS_TEMP>>>(spot, (int)num_entries, ratings_rows, 
          csr_format_ratingsMtx_userID_dev,
          coo_format_ratingsMtx_itemID_dev,
          coo_format_ratingsMtx_rating_dev,
          cosine_similarity_dev, compare_values, 
          top_N_most_sim_itemIDs_dev, top_N_most_sim_item_similarity_dev,  
          ratings_cols, Top_N, isBad);
        // get_cosine_similarity_host_kernel_debug(spot, (int)num_entries, ratings_rows, 
        //   csr_format_ratingsMtx_userID_dev,
        //   coo_format_ratingsMtx_itemID_dev,
        //   coo_format_ratingsMtx_rating_dev,
        //   cosine_similarity_dev, compare_values, 
        //   top_N_most_sim_itemIDs_dev, top_N_most_sim_item_similarity_dev,  
        //   ratings_cols, Top_N, isBad);
      }else{
        get_cosine_similarity_host_kernel<<<CUDA_NUM_BLOCKS_TEMP, CUDA_NUM_THREADS_TEMP>>>(spot, (int)num_entries, ratings_rows, 
          csr_format_ratingsMtx_userID_dev,
          coo_format_ratingsMtx_itemID_dev,
          coo_format_ratingsMtx_rating_dev,
          cosine_similarity_dev, compare_values, isBad);   
      }

      CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
      if(isBad_host){
        if(top_N_most_sim_itemIDs_dev){
          //
        }else{
          get_cosine_similarity_host_kernel_debug(spot, (int)num_entries, ratings_rows, 
            csr_format_ratingsMtx_userID_dev,
            coo_format_ratingsMtx_itemID_dev,
            coo_format_ratingsMtx_rating_dev,
            cosine_similarity_dev, compare_values, isBad);  
        }
        LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
        LOG("num_loops : " <<num_loops) ;
        LOG("spot : " <<spot) ;
        LOG("num_entries : " <<num_entries) ;
        save_device_array_to_file(cosine_similarity_dev, (int)num_entries, "cosine_similarity_dev", strPreamble(blank));
        ABORT_IF_NEQ(0, 1, "isBad");
      };

      
      checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  num_entries *  SIZE_OF(float), cudaMemcpyDeviceToHost));
      if(Debug ) {
        checkCudaErrors(cudaDeviceSynchronize());
        gpu_set_all<float>(cosine_similarity_dev, num_entries, (float)0.0);
        //LOG("Here") ;
        //save_host_array_to_file<float>(cosine_similarity_host + spot, (int)num_entries, "cosine_similarity_host");
      }
      num_gpu_blocks = num_gpu_blocks - CUDA_NUM_BLOCKS_TEMP;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
      if(Debug && top_N_most_sim_itemIDs_dev) {
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 + (program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        if(print) LOG("get_cosine_similarity_host average loop run time : "<<readable_time(program_time / (double)num_loops));
        //ABORT_IF_NEQ(0, 1, "");
      }
    } // end while

    if(Debug && 0) {
      LOG("num_gpu_blocks : " <<num_gpu_blocks) ;
      LOG("num_loops : " <<num_loops) ;
      LOG("spot : " <<spot) ;
    }
    // spot is the number of entries done so far
    // total - (done) = left to go 
    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  (num_below_diag - spot) *  SIZE_OF(float), cudaMemcpyHostToDevice));
    if(top_N_most_sim_itemIDs_dev){
      get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>(spot, (int)(num_below_diag - spot), ratings_rows, 
        csr_format_ratingsMtx_userID_dev,
        coo_format_ratingsMtx_itemID_dev,
        coo_format_ratingsMtx_rating_dev,
        cosine_similarity_dev, compare_values, 
        top_N_most_sim_itemIDs_dev, top_N_most_sim_item_similarity_dev,  
          ratings_cols, Top_N,  isBad);
    }else{
      get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>(spot, (int)(num_below_diag - spot), ratings_rows, 
        csr_format_ratingsMtx_userID_dev,
        coo_format_ratingsMtx_itemID_dev,
        coo_format_ratingsMtx_rating_dev,
        cosine_similarity_dev, compare_values, isBad);       
    }
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if(isBad_host){
      save_device_array_to_file(cosine_similarity_dev, (int)num_entries, "cosine_similarity_dev", strPreamble(blank));
      ABORT_IF_NEQ(0, 1, "isBad");
    };
    checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  (num_below_diag - spot) *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cosine_similarity_dev));
  }else{
    if(Debug) LOG("get_cosine_similarity_host num_gpu_blocks <= CUDA_NUM_BLOCKS") ;
    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_below_diag * SIZE_OF(float)));

    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host,  num_below_diag *  SIZE_OF(float), cudaMemcpyHostToDevice));
    if(top_N_most_sim_itemIDs_dev){
      get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>(0, (int)num_below_diag, ratings_rows, 
        csr_format_ratingsMtx_userID_dev,
        coo_format_ratingsMtx_itemID_dev,
        coo_format_ratingsMtx_rating_dev,
        cosine_similarity_dev, compare_values, 
        top_N_most_sim_itemIDs_dev, top_N_most_sim_item_similarity_dev,  
          ratings_cols, Top_N, isBad);
    }else{
      get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS_TEMP>>>(0, (int)num_below_diag, ratings_rows, 
        csr_format_ratingsMtx_userID_dev,
        coo_format_ratingsMtx_itemID_dev,
        coo_format_ratingsMtx_rating_dev,
        cosine_similarity_dev, compare_values, isBad);      
    }
    checkCudaErrors(cudaMemcpy(cosine_similarity_host , cosine_similarity_dev,  num_below_diag *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cosine_similarity_dev));
    CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isBad));
    if(isBad_host){
      ABORT_IF_NEQ(0, 1, "isBad");
    };
  };


  if(Debug){

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
{ABORT_IF_NEQ(0, 1, "Function Not Ready.");
  /*
  bool Debug = false;
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) LOG("called get_cosine_similarity_host") ;
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;


  //long long int num_gpu_blocks = GET_BLOCKS(ratings_rows * ratings_rows);
  long long int num_gpu_blocks = GET_BLOCKS(num_below_diag);

  if(print) LOG("total loops : " <<ceil((num_below_diag) / (CUDA_NUM_BLOCKS*CUDA_NUM_THREADS))) ;


  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    const long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot = (long long int)0;

    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_entries * SIZE_OF(float)));
    float * cosine_similarity_dev_1;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev_1, num_entries * SIZE_OF(float)));
    float* cosine_similarity_dev_ptrs[2] = {cosine_similarity_dev, cosine_similarity_dev_1};
    if(Debug) LOG("cosine_similarity_dev allocated") ;

    
    #ifdef _OPENMP
      const int nthreads = 2;
      omp_set_num_threads(nthreads);
      omp_lock_t *locks = (omp_lock_t *)malloc(nthreads * SIZE_OF(omp_lock_t));
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
              this_, false, isBad);

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
          checkCudaErrors(cudaMemcpy(cosine_similarity_host + temp * num_entries, this_,  num_entries *  SIZE_OF(float), cudaMemcpyDeviceToHost));
          
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
    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host + spot,  (num_below_diag - spot) *  SIZE_OF(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(spot, num_below_diag - spot, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, false, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host + spot, cosine_similarity_dev,  (num_below_diag - spot) *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cosine_similarity_dev));
    checkCudaErrors(cudaFree(cosine_similarity_dev_1));
  }else{
    if(Debug) LOG("get_cosine_similarity_host num_gpu_blocks <= CUDA_NUM_BLOCKS") ;
    float * cosine_similarity_dev;
    checkCudaErrors(cudaMalloc((void**)&cosine_similarity_dev, num_below_diag * SIZE_OF(float)));

    // checkCudaErrors(cudaMemcpy(cosine_similarity_dev, cosine_similarity_host,  num_below_diag *  SIZE_OF(float), cudaMemcpyHostToDevice));
    get_cosine_similarity_host_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(0, num_below_diag, ratings_rows, 
      csr_format_ratingsMtx_userID_dev,
      coo_format_ratingsMtx_itemID_dev,
      coo_format_ratingsMtx_rating_dev,
      cosine_similarity_dev, false, isBad);
    checkCudaErrors(cudaMemcpy(cosine_similarity_host , cosine_similarity_dev,  num_below_diag *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cosine_similarity_dev));
  };


  if(Debug){

  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  #ifdef _OPENMP
    //free(locks);
  #endif
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  if(print) LOG("finished call to get_cosine_similarity_host") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("get_cosine_similarity_host run time : "<<readable_time(program_time)<<std::endl);
  */
}


template<typename Dtype>
__global__ void gpu_permute_kernel(Dtype* A, const int* P, long long int rows, long long int cols, bool permute_rows, 
 const long long int start, const int size) 
{
  //A is rows by cols stored in col major ordering

  if(permute_rows){
    //pvt is an array length rows
    CUDA_KERNEL_LOOP(j, size /*cols*/) {
      long long int ind=(long long int)0;
      Dtype temp=(Dtype)0.0;

      for(long long int i=(long long int)0; i < (rows - (long long int)1); i+=(long long int)1){
          // get next index
        ind = (long long int)(P[i]);
        while(ind<i)
          ind = (long long int)(P[ind]);

          // swap elements in array
        temp = A[i + ((long long int)j + start) * rows];
        A[i + ((long long int)j + start) * rows] = A[ind + ((long long int)j + start) * rows];
        A[ind + ((long long int)j + start) * rows] = temp;
      };
    };
  } else{
    //pvt is an array length cols
    CUDA_KERNEL_LOOP(i, size /*rows*/) {
      int ind=(long long int)0;
      Dtype temp=(Dtype)0.0;

      for(long long int j=(long long int)0; j<(cols-(long long int)1); j+=(long long int)1){
              // get next index
        ind = (long long int)(P[j]);
        while(ind<j)
          ind = (long long int)(P[ind]);

              // swap elements in array
        temp = A[((long long int)i + start) + j * rows];
        A[((long long int)i + start) + j * rows] = A[((long long int)i + start) + ind * rows];
        A[((long long int)i + start) + ind * rows] = temp;
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
    num_gpu_blocks = GET_BLOCKS(cols);
  }else{
    num_gpu_blocks = GET_BLOCKS(rows);
  };
  
  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //   ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // }; 
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops = (long long int)0;
    long long int num_entries = CUDA_NUM_BLOCKS * CUDA_NUM_THREADS;
    long long int spot = (long long int)0;
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
      gpu_permute_kernel<Dtype><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, spot, (int)num_entries);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    gpu_permute_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, spot, (int)(permute_rows ? (cols - spot) : (rows - spot)));
  }else{
    if(too_big(permute_rows ? cols : rows) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    gpu_permute_kernel<Dtype><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(A,  P, rows, cols,  permute_rows, 0, (int)(permute_rows ? cols : rows));
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
    checkCudaErrors(cudaMalloc((void**)&indicies, M * SIZE_OF(int)));
    gpu_set_as_index(indicies, M);
    gpu_shuffle_array<int>(dn_handle, M, indicies);
    if(row_major_ordering){
      gpu_permute<float>(x, indicies, N, M, !shuffle_rows); 
    }else{
      gpu_permute<float>(x, indicies, M, N, shuffle_rows); 
    };
    checkCudaErrors(cudaFree(indicies));
  }else{
    checkCudaErrors(cudaMalloc((void**)&indicies, N * SIZE_OF(int)));
    gpu_set_as_index(indicies, N);
    gpu_shuffle_array<int>(dn_handle, N, indicies);
    if(row_major_ordering){
      gpu_permute<float>(x, indicies, N, M, shuffle_rows); 
    }else{
      gpu_permute<float>(x, indicies, M, N, !shuffle_rows); 
    };
    checkCudaErrors(cudaFree(indicies));
  }

} 

__global__ void gpu_normalize_mtx_rows_or_cols_kernel(const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool normalize_rows, bool* isBad)
{
  float temp;
  if(normalize_rows){
    CUDA_KERNEL_LOOP(i, M ) {
      float norm = (float)0.0;
      if(row_major_ordering){
        for( long long int j = (long long int)0; j < N; j+=(long long int)1){
          norm += pow(x[(long long int)i * N + j], (float)2.0);
        }
        norm = sqrt(norm);
        for(long long int j = (long long int)0; j < N; j+=(long long int)1){
          x[(long long int)i * N + j] = x[(long long int)i * N + j] / norm;
          if (::isinf(x[(long long int)i * N + j]) || ::isnan(x[(long long int)i * N + j])){
            isBad[0] = true;
          };
        }

      }else{
        for(long long int j = (long long int)0; j < N; j+=(long long int)1){
          norm += pow(x[(long long int)i + M * j], (float)2.0);
        }
        norm = sqrt(norm);
        for(long long int j = (long long int)0; j < N; j+=(long long int)1){
          x[(long long int)i + M * j] = x[(long long int)i + M * j] / norm;
          if (::isinf(x[(long long int)i + M * j]) || ::isnan(x[(long long int)i + M * j])){
            isBad[0] = true;
          };
        }
      };
    }
  }else{
    CUDA_KERNEL_LOOP(j, N ) {
      float norm = (float)0.0;
      if(row_major_ordering){
        for(long long int i = (long long int)0; i < M; i+=(long long int)1){
          norm += pow(x[i * N + (long long int)j], (float)2.0);
        }
        norm = sqrt(norm);
        for(long long int i = (long long int)0; i < M; i+=(long long int)1){
          x[i * N + (long long int)j] = x[i * N + (long long int)j] / norm;
          if (::isinf(x[i * N + (long long int)j]) || ::isnan(x[i * N + (long long int)j])){
            isBad[0] = true;
          };
        }
      }else{
        for(long long int i = (long long int)0; i < M; i+=(long long int)1){
          norm += pow(x[i + M * (long long int)j], (float)2.0);
        }
        norm = sqrt(norm);
        for(long long int i = (long long int)0; i < M; i+=(long long int)1){
          x[i + M * (long long int)j] = x[i + M * (long long int)j] / norm;
          if (::isinf(x[i + M * (long long int)j]) || ::isnan(x[i + M * (long long int)j])){
            isBad[0] = true;
          };
        }
      };
    }
  }
}

void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool normalize_rows)
{
  std::string blank = "";
  long long int num_gpu_blocks;
  if(normalize_rows){
    num_gpu_blocks = GET_BLOCKS(M);
  }else{
    num_gpu_blocks = GET_BLOCKS(N);
  };
  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  }; 
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));
  gpu_normalize_mtx_rows_or_cols_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(M, N, row_major_ordering, x, normalize_rows, isBad);
  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    if(row_major_ordering){
      save_device_mtx_to_file<float>(x, N, M, "x_bad", true, strPreamble(blank));
    }else{
      save_device_mtx_to_file<float>(x, M, N, "x_bad", false, strPreamble(blank));
    }
    ABORT_IF_NEQ(0, 1, "isBad");
  };
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS(m * num_latent_factors);

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

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS(m * num_latent_factors);

  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //     ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };

  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
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
    if(too_big(m * num_latent_factors) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    gpu_mult_US_in_SVD_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(m, U, S, (long long int)0, m * num_latent_factors, right_multiply_by_S, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  if(too_big(M * N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  // x is M by M in memory
  // y is M by N where N < M 
  const long long int num_gpu_blocks = GET_BLOCKS(M * N);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  gpu_copy_and_transpose_kernel<float><<<num_gpu_blocks, CUDA_NUM_THREADS>>>(M, N, x, y);
}

template <>
void gpu_get_num_latent_factors<float>(cublasHandle_t dn_handle, const long long int m, float* S, 
  long long int* num_latent_factors, const float percent) 
{
  if(too_big(m) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}

  bool Debug = false;
  float sum;
  if(Debug) {
    LOG("here!");
    LOG("m : "<< m);
  }
  gpu_asum<float>(dn_handle, m, S, &sum);
  if(Debug) LOG("here!");

  float* S_host = NULL;
  S_host = (float *)malloc(SIZE_OF(float)*m); 
  checkErrors(S_host);
  CUDA_CHECK(cudaMemcpy(S_host, S, m * SIZE_OF(float), cudaMemcpyDeviceToHost));

  if(Debug) LOG("here!");

  float sum_so_far;
  num_latent_factors[0] = m;
  for(int j = 0; j < (int)m; j++){
    if(Debug) {
      LOG("S_host[ "<<j<<" ] : "<< S_host[j]);
    }
    sum_so_far += S_host[j];
    if((sum_so_far / sum) >= percent) {
      if(Debug) LOG("num_latent_factors = "<< j+1);
      num_latent_factors[0] = (long long int)(j+1);
      break;
    }
  }
  free(S_host);
  if(Debug) LOG("here!");
}

template <>
void gpu_get_latent_factor_mass<float>(cublasHandle_t dn_handle, const long long int m, float* S, 
  const long long int num_latent_factors, float* percent) 
{
  if(too_big(m) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  if(num_latent_factors > m ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}

  bool Debug = false;
  float sum;
  if(Debug) {
    LOG("here!");
    LOG("m : "<< m);
  }
  gpu_asum<float>(dn_handle, m, S, &sum);
  if(Debug) LOG("here!");

  float* S_host = NULL;
  S_host = (float *)malloc(SIZE_OF(float)*m); 
  checkErrors(S_host);
  CUDA_CHECK(cudaMemcpy(S_host, S, m * SIZE_OF(float), cudaMemcpyDeviceToHost));

  if(Debug) LOG("here!");

  float sum_so_far;
  for(int j = 0; j < (int)num_latent_factors; j++){
    if(Debug) {
      LOG("S_host[ "<<j<<" ] : "<< S_host[j]);
    }
    sum_so_far += S_host[j];
  }
  percent[0] = sum_so_far / sum;
  free(S_host);
  if(Debug) LOG("here!");
}

template <>
void preserve_first_m_rows<float>(const long long int old_lda, const long long int new_lda, 
                                  const long long int sda, float* V) 
{
  if(new_lda > old_lda) {
    LOG("WARNING: new_lda > old_lda");
    return;
  }
  for(int j = 0; j < sda; j++){
    checkCudaErrors(cudaMemcpy(V + new_lda * j, V + old_lda * j, 
     new_lda * SIZE_OF(float), cudaMemcpyDeviceToDevice));
  }
  float* V_copy = NULL;
  V_copy = (float *)malloc(SIZE_OF(float) * new_lda * sda); 
  checkErrors(V_copy);
  CUDA_CHECK(cudaMemcpy(V_copy, V, new_lda * sda * SIZE_OF(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(V));
  CUDA_CHECK(cudaMalloc ( (void**)&V  , SIZE_OF(float) * new_lda * sda ) );
  CUDA_CHECK(cudaMemcpy(V, V_copy, new_lda * sda * SIZE_OF(float), cudaMemcpyHostToDevice));
  free(V_copy);

}


int first_tiny_sv(const long long int n, float* SV, float epsilon) 
{
  float* SV_copy = NULL;
  SV_copy = (float *)malloc(SIZE_OF(float) * n); 
  checkErrors(SV_copy);
  CUDA_CHECK(cudaMemcpy(SV_copy, SV, n * SIZE_OF(float), cudaMemcpyDeviceToHost));

  for(long long int i = (long long int)0; i < n; i+=(long long int)1){
    if(SV_copy[i] < epsilon) {
      free(SV_copy);
      return (int)i + 1;
    }
  }
  free(SV_copy);
  return ((int)n + 1);

}



template <>
void gpu_orthogonal_decomp<float>(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle,
                                  const long long int m, const long long int n, 
                                  long long int* num_latent_factors, const float percent,
                                  float* A, float* U, float* V, bool S_with_U, float* S) 
{
  /*
    A is m by n stored in col-maj ordering

    solution     A      =       U       *      S      *       V^T
              m by n          m by m         m by n         n by n

              ..but S only has min(m,n) non zero entries so if m < n

    solution     A      =       U*S      *       V^T
              m by n          m by n           n by n

  */
  if(too_big(n * m) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  bool Debug = false;
  std::string blank = "";
  const long long int min_dim = std::min(m,n);


  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

  float* A_host_copy = NULL;
  A_host_copy = (float *)malloc(SIZE_OF(float)*m*n); 
  checkErrors(A_host_copy);
  CUDA_CHECK(cudaMemcpy(A_host_copy, A, m*n * SIZE_OF(float), cudaMemcpyDeviceToHost));

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

  CUDA_CHECK(cudaMalloc ((void**)&d_S    , SIZE_OF(float)*std::min(lda,sda)));
  CUDA_CHECK(cudaMalloc ((void**)&devInfo, SIZE_OF(int)));
  CUDA_CHECK(cudaMalloc ((void**)&d_rwork, SIZE_OF(float)*(std::min(lda,sda) - 1)));



  // query working space of SVD
  cusolver_status = cusolverDnSgesvd_bufferSize(
              dn_solver_handle,
              lda,
              sda,
              &lwork );
  CUSOLVER_CHECK(cusolver_status);
  //assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

  CUDA_CHECK(cudaMalloc((void**)&d_work , SIZE_OF(float) * lwork));

  

  // compute SVD
  signed char jobu  = 'N'; // min(m,n) columns of U
  signed char jobvt = 'A'; // min(m,n) rows of VT
  if(Debug && 0){
    LOG("gpu_orthogonal_decomp requires " <<(std::min(lda,sda) + std::min(lda,sda) - 1 + lwork ) * SIZE_OF(float) + SIZE_OF(int)<< " bytes of memory");
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

  CUDA_CHECK(cudaMemcpy(A, A_host_copy, m*n * SIZE_OF(float), cudaMemcpyHostToDevice));
  free(A_host_copy);

  if(cusolver_status != CUSOLVER_STATUS_SUCCESS){
    if(n > m){
      //we have to solve the transpose problem
      if(Debug) LOG("Solving the tranpose problem...");
    }
    LOG("buffersize = "<<lwork);
    LOG("cusolverDnSgesvd ("<<dn_solver_handle<<","<<jobu<<","<<jobvt<<", lda = "<<lda<<", sda = "<<sda<<","<<A<<", lda = "<<lda<<","
          <<d_S<<","<<d_U<<", ldu = "<<lda<<","<<d_VT<<", ldvt = "<<sda/*std::min(m,n)*/<<","<<d_work<<", lwork = "<<lwork<<","<<d_rwork<<","<<
      devInfo<<")");
      //typeid(a).name()
    LOG("cusolverDnSgesvd ("<<typeid(dn_solver_handle).name()<<","<<typeid(jobu).name()<<","<<typeid(jobvt).name()<<", lda = "<<typeid(lda).name()<<", sda = "<<typeid(sda).name()<<","<<typeid(A).name()<<", lda = "<<typeid(lda).name()<<","
          <<typeid(d_S).name()<<","<<typeid(d_U).name()<<", ldu = "<<typeid(lda).name()<<","<<typeid(d_VT).name()<<", ldvt = "<<typeid(sda).name()/*std::min(m,n)*/<<","<<typeid(d_work).name()<<", lwork = "<<typeid(lwork).name()<<","<<typeid(d_rwork).name()<<","<<
      typeid(devInfo).name()<<")");  

    save_device_mtx_to_file<float>(A, m, n, "A"); 
    save_device_mtx_to_file<float>(d_U, lda, min_dim, "d_U");
    save_device_mtx_to_file<float>(d_VT, sda, min_dim, "d_VT");
    save_device_array_to_file<float>(d_S, min_dim, "singular_values"); 
  }

  CUSOLVER_CHECK(cusolver_status);
  checkCudaErrors(cudaDeviceSynchronize());

  if(Debug && 0){
    checkCudaErrors(cudaDeviceSynchronize());
    save_device_mtx_to_file<float>(U, m, min_dim, "U");
    save_device_mtx_to_file<float>(V, n, min_dim, "V");
    save_device_array_to_file<float>(d_S, min_dim, "singular_values");
      // LOG("Press Enter to continue.") ;
      // std::cin.ignore();
  }
  int first_tiny_sv_ = first_tiny_sv(min_dim, d_S, (float)0.00001);

  if(n > m){
      //the tranpose problem was solved
      //sda = m;
    if(Debug) LOG("need to transpose "<<m<<" by "<<m<<" mtx U");
    transpose_in_place<float>(handle, sda,  sda, U);
    // A^T * U
    gpu_gemm<float>(handle, false, true, m, n, m, (float)1.0, U, A, (float)0.0, V);
    if(Debug && 0){
      checkCudaErrors(cudaDeviceSynchronize());
      save_device_mtx_to_file<float>(U, m, min_dim, "U_2");
      save_device_mtx_to_file<float>(V, n, min_dim, "V_2");\
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }
    if(S_with_U){
      if(first_tiny_sv_ < min_dim){
        LOG("WARNING WILL DIVIDE BY ~ZERO");
        LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<min_dim);
        save_device_array_to_file<float>(d_S, min_dim, "singular_values");
      }
      gpu_mult_US_in_SVD<float>(m, min_dim, U, d_S, true);
      gpu_div_US_in_SVD<float>(n, min_dim, V, d_S, true);
    }
  }else{
      //sda = n;
    if(Debug) LOG("need to transpose "<<n<<" by "<<m<<" mtx V");
    transpose_in_place<float>(handle, sda,  sda, V);
    gpu_gemm<float>(handle, false, false, n, m, n, (float)1.0, V, A, (float)0.0, U);
    if(!S_with_U){
      if(first_tiny_sv_ < min_dim){
        LOG("WARNING WILL DIVIDE BY ~ZERO");
        LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<min_dim);
        save_device_array_to_file<float>(d_S, min_dim, "singular_values");
      }
      if(Debug) LOG("Divide U by S and mult V by S");
      gpu_div_US_in_SVD<float>(m, min_dim, U, d_S, true);
      gpu_mult_US_in_SVD<float>(n, min_dim, V, d_S, true);
    }
  };

  if(S != NULL){
    CUDA_CHECK(cudaMemcpy(S, d_S, min_dim * SIZE_OF(float), cudaMemcpyDeviceToDevice));
  }




  

  gpu_get_num_latent_factors<float>(handle, min_dim, d_S, num_latent_factors, percent);


  if(Debug){
    float max_S = (float)0.0;
    CUDA_CHECK(cudaMemcpy(&max_S, d_S, SIZE_OF(float), cudaMemcpyDeviceToHost));
    LOG("Largest Singular Value : "<<max_S);
    //checkCudaErrors(cudaDeviceSynchronize());
    //LOG("num_latent_factors : "<<num_latent_factors[0]) ;

    // print_gpu_mtx_entries<float>(A, m, n);
    // print_gpu_array_entries<float>(S, std::min(m,n));
    // print_gpu_mtx_entries<float>(U, m, min_dim);
    // print_gpu_mtx_entries<float>(V, n, min_dim);

    save_device_array_to_file<float>(d_S, min_dim, "singular_values", strPreamble(blank));
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }    
  


  if (d_S    ) checkCudaErrors(cudaFree(d_S));
  if (devInfo) checkCudaErrors(cudaFree(devInfo));
  if (d_work ) checkCudaErrors(cudaFree(d_work));
  if (d_rwork) checkCudaErrors(cudaFree(d_rwork));

  if(Debug){
    float * R;
    checkCudaErrors(cudaMalloc((void**)&R, m * n * SIZE_OF(float)));

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

    // op(A) is K by M
    // op(B) is N by K
    // C is N by M
    // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C

        // U^T * U
    // gpu_gemm<float>(handle, false, true, min_dim, min_dim, m /*num_latent_factors[0]*/,
    //                 (float)1.0, U, U, (float)0.0, R);

    // save_device_mtx_to_file<float>(R, min_dim, min_dim, "UTU", false);

    // // V^T * V
    // gpu_gemm<float>(handle, false, true, min_dim, min_dim, n /*num_latent_factors[0]*/,
    //                 (float)1.0, V, V, (float)0.0, R);

    // save_device_mtx_to_file<float>(R, min_dim, min_dim, "VTV", false);


      // U * V^T
    gpu_gemm<float>(handle, true, false, n, m, min_dim /*num_latent_factors[0]*/,
                    (float)1.0, V, U, (float)0.0, R);

    gpu_axpby<float>(handle, m * n, (float)1.0, A,
      (float)(-1.0), R);

    //save_device_mtx_to_file<float>(R, m, n, "svd_error", false);
    float epsilon = 0.0001;
    //float range_A    = gpu_range<float>(m * n,  A);
    float error      = gpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R);
    if(error > epsilon){
      LOG("SVD max error = "<<ToString<float>(error)) ;
      ABORT_IF_EQ(0, 0, "gpu_orthogonal_decomp factoring incorrectly");
    }
    checkCudaErrors(cudaFree(R));
    // LOG("A mtx range of values = "<<range_A) ;
    
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;

    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

  }





  if(Debug) LOG("finished call to gpu_orthogonal_decomp") ;
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(Debug) LOG("gpu_orthogonal_decomp run time : "<<readable_time(program_time)<<std::endl);
}  

void gpu_orthogonal_decomp_test(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle) 
{
  const long long int m = 5;
  const long long int n = 3;
  const long long int min_dim = std::min(m,n);

  float* A;
  float* U;
  float* V;
  checkCudaErrors(cudaMalloc((void**)&A,  m*n * SIZE_OF(float)));
  checkCudaErrors(cudaMalloc((void**)&U,  m*min_dim * SIZE_OF(float)));
  checkCudaErrors(cudaMalloc((void**)&V,  min_dim*n * SIZE_OF(float)));

  gpu_rng_gaussian<float>(m*n, (float)0.0, (float)1.0, A);
  save_device_mtx_to_file<float>(A, m, n, "A");

  long long int num_latent_factors;
  const float percent = (float)0.95;

  gpu_orthogonal_decomp<float>(handle, dn_solver_handle, m, n, &num_latent_factors, percent,
  A, U, V, 1);

  checkCudaErrors(cudaFree(A));
  checkCudaErrors(cudaFree(U));
  checkCudaErrors(cudaFree(V));
}


template <>
void gpu_block_orthogonal_decomp_from_host<float>(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle,
                                                  const long long int m, const long long int n, 
                                                  long long int* num_latent_factors, const float percent,
                                                  float* A, float* U, float* V, bool row_major_ordering,
                                                  long long int block_rows, bool S_with_U, float* S)
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
  LOG("gpu_block_orthogonal_decomp_from_host called...");
  if(row_major_ordering) LOG("We assume A exists on host in row major order");
  LOG("WARNING: Returns U in row major order and V in column major order") ;

  if(Debug){
    LOG("handle  = "<<handle<< "," << 
      "dn_solver_handle  = "<<dn_solver_handle<< "," << 
      "m  = "<<m<< "," << 
      "n  = "<<n<< "," << 
      "num_latent_factors  = "<<num_latent_factors<< "," << 
      "percent  = "<<percent<< "," << 
      "A  = "<<A<< "," << 
      "U  = "<<U<< "," << 
      "V  = "<<V<< "," << 
      "row_major_ordering  = "<<row_major_ordering<< "," << 
      "block_rows  = "<<block_rows<< "," << 
      "S_with_U  = "<<S_with_U<< "," << 
      "S  = "<<S);                                                
  }

  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  float* S_temp = NULL;
  if(S == NULL){
    if(Debug){ 
      LOG("S == NULL");
    }
    S_temp = (float *)malloc(m *  SIZE_OF(float)); 
    checkErrors(S_temp);
    S = S_temp;
    cpu_set_all(S, m, (float)0.0);
  }



  bool divides_evenly = true;

  int num_blocks = (int)(m / block_rows);
  const long long int left_over = m % block_rows;
  if((int)(left_over) != 0){ 
    num_blocks += 1;
    divides_evenly = false;
  }

  const long long int min_dim = std::min(block_rows, n);
  if(Debug){ 
    LOG("min_dim : "<< min_dim);
    LOG("rows : "<< m);
    LOG("cols : "<< n);
    LOG("block_rows : "<< block_rows);
    LOG("num_blocks : "<< num_blocks);
    LOG("divides_evenly : "<< divides_evenly);
  }


  /*
    A in row major ordering is equivalent to A^T in column major ordering
    A in column major ordering is equivalent to A^T in row major ordering
  */
  // int nthreads = 1;
  // #ifdef _OPENMP
  //   int nProcessors = omp_get_max_threads();
  //   nthreads = (int)std::min(nProcessors, num_blocks);
  //   //omp_set_num_threads(nthreads);
  //   omp_set_num_threads(1);
  //   omp_lock_t printlock;

  //   omp_init_lock(&printlock);
  // #endif
  // #pragma omp parallel shared(nthreads)
  // {
    float* A_dev_;
    float* U_dev_; 
    float* V_dev_;
    float* S_dev_;
    if(too_big(block_rows * n * SIZE_OF(float)) ) {ABORT_IF_EQ(0, 0, "Long long long int too big");}
    checkCudaErrors(cudaMalloc((void**)&A_dev_,  block_rows * n * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&U_dev_,  block_rows * min_dim * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&V_dev_,  n * min_dim * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&S_dev_,  min_dim * SIZE_OF(float)));

    if(Debug){
      gpu_set_all<float>(A_dev_, block_rows * n, (float)0.0);
      gpu_set_all<float>(U_dev_, block_rows * min_dim, (float)0.0);
      gpu_set_all<float>(V_dev_, min_dim * n, (float)0.0);
      gpu_set_all<float>(S_dev_, min_dim, (float)0.0);

      cpu_set_all<float>(U, m * m, (float)0.0);
      cpu_set_all<float>(V, m * n, (float)0.0);
      cpu_set_all<float>(S, m, (float)0.0);  

      LOG("gpu_block_orthogonal_decomp_from_host requires " <<(block_rows * n + block_rows * min_dim + n * min_dim + min_dim)  * SIZE_OF(float)<< " bytes of memory");     
    }

       

    // int th_id = 0;
    // #ifdef _OPENMP
    //   th_id = omp_get_thread_num();
    // #endif
    // for(long long int block = (long long int)th_id; block < (long long int)num_blocks; block += (long long int)nthreads){
    for(long long int block = (long long int)0; block < (long long int)num_blocks; block+=(long long int)1){   
      long long int start = block * (block_rows * m + min_dim);

      if(block < (long long int)(num_blocks - 1) || divides_evenly || num_blocks == 1){
        checkCudaErrors(cudaMemcpy(A_dev_,  A + (block * block_rows) * n,  block_rows * n * SIZE_OF(float), cudaMemcpyHostToDevice));
        if(Debug && 0) {checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");}
        // A + (block * block_rows) * n is in row major ordering
        // A_dev_ is [A + (block * block_rows) * n] ^ T in column major ordering
        // A = U * V^T -> A^T = V * U^T
        if(row_major_ordering){
          gpu_orthogonal_decomp<float>(handle, dn_solver_handle,
                                      n, block_rows, num_latent_factors, percent, 
                                      A_dev_, V_dev_, U_dev_, !S_with_U, S_dev_); 
        }else{
          gpu_orthogonal_decomp<float>(handle, dn_solver_handle,
                                      block_rows, n, num_latent_factors, percent, 
                                      A_dev_, U_dev_, V_dev_, S_with_U, S_dev_);           
        }


        //gpu_swap_ordering(n, min_dim, V_dev_, false);
        //gpu_swap_ordering(block_rows, min_dim, U_dev_, false);

        // gpu_scale(handle, n * min_dim, (float)1.0 / (float)num_blocks, V_dev_);
        // gpu_scale(handle, block_rows * min_dim, (float)num_blocks, U_dev_);
        // gpu_scale(handle, min_dim, pow((float)num_blocks, (float)2.0), S_dev_);
        if(Debug && 0){ 
          LOG("block : "<<block);
          LOG("start : "<<start);
          LOG("m : "<<m);
          LOG("n : "<<n);
          LOG("min_dim : "<< min_dim);
          LOG("num_latent_factors : "<<num_latent_factors[0]);
          LOG("percent : "<<percent);
          LOG("block_rows : "<< block_rows);
          LOG("S_with_U : "<< S_with_U);
          LOG("num_blocks : "<< num_blocks);
          LOG("divides_evenly : "<< divides_evenly);

          //save_host_array_to_file<float>(S, m, "S_before", strPreamble(blank));
          //print_host_mtx<float>(A, m, n, "A", 1, strPreamble(blank));
          //print_host_mtx<float>(V, n, m, "V", 0, strPreamble(blank)); 
          //print_host_mtx<float>(U, m, m, "U", 0, strPreamble(blank));  
          //save_host_mtx_to_file<float>(A, block_rows * (block + (long long int)1), n, "A", 1, strPreamble(blank));
          //save_host_mtx_to_file<float>(V, n, m, "V_before", 0, strPreamble(blank)); 
          //save_host_mtx_to_file<float>(U, m, m, "U_before", 0, strPreamble(blank)); 
          //std::string tit = "S_dev_before";
          //save_device_array_to_file<float>(S_dev_, min_dim, strPreamble(blank));
          //print_gpu_mtx_entries<float>(A_dev_, n, block_rows, "AT_dev", 0, strPreamble(blank));
          //print_gpu_mtx_entries<float>(V_dev_, n, m, "V_dev", 0, strPreamble(blank)); 
          //print_gpu_mtx_entries<float>(U_dev_, m, m, "U_dev", 0, strPreamble(blank));  
          save_device_mtx_to_file<float>(A_dev_, (int)n,          (int)block_rows, "AT_dev", 0, strPreamble(blank));
          save_device_mtx_to_file<float>(V_dev_, (int)n,          (int)min_dim,    "V_dev",  0, strPreamble(blank)); 
          save_device_mtx_to_file<float>(U_dev_, (int)block_rows, (int)min_dim,    "U_dev",  0, strPreamble(blank));  
          checkCudaErrors(cudaDeviceSynchronize());
          // LOG("Press Enter to continue.") ;
          // std::cin.ignore();
        }

        checkCudaErrors(cudaMemcpy(S + block * min_dim,  S_dev_,  min_dim * SIZE_OF(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(V + block * n * min_dim,  V_dev_,  n * min_dim * SIZE_OF(float), cudaMemcpyDeviceToHost));
        //copy_device_mtx_into_host_submtx<float>(block_rows, min_dim, U_dev_, U + start, m);
        
        if(Debug && 0){ 
          //save_host_array_to_file<float>(S, m, "S", strPreamble(blank));
          //print_host_mtx<float>(A, m, n, "A", 1, strPreamble(blank));
          //print_host_mtx<float>(V, n, m, "V", 0, strPreamble(blank)); 
          //print_host_mtx<float>(U, m, m, "U", 0, strPreamble(blank));  
          //save_host_mtx_to_file<float>(A, block_rows * (block + (long long int)1), n, "A_after", 1, strPreamble(blank));
          //save_host_mtx_to_file<float>(V, n, m, "V", 0, strPreamble(blank)); 
          save_host_mtx_to_file<float>(V + block * n * min_dim, (int)n, (int)min_dim, "V", 0, strPreamble(blank)); 
          //save_host_mtx_to_file<float>(U, m, m, "U", 0, strPreamble(blank)); 
          save_host_mtx_to_file<float>(U + start, (int)block_rows, (int)1, "U", 0, strPreamble(blank)); 
          //LOG("S_dev_after : ");
          //save_device_array_to_file<float>(S_dev_, min_dim, strPreamble(blank));
          //print_gpu_mtx_entries<float>(A_dev_, n, block_rows, "AT_dev", 0, strPreamble(blank));
          //print_gpu_mtx_entries<float>(V_dev_, n, m, "V_dev", 0, strPreamble(blank)); 
          //print_gpu_mtx_entries<float>(U_dev_, m, m, "U_dev", 0, strPreamble(blank));  
          //save_device_mtx_to_file<float>(A_dev_, n, block_rows, "AT_dev_after", 0, strPreamble(blank));
          //save_device_mtx_to_file<float>(V_dev_, n, min_dim, "V_dev_after", 0, strPreamble(blank)); 
          //save_device_mtx_to_file<float>(U_dev_, block_rows, min_dim, "U_dev_after", 0, strPreamble(blank));  

          checkCudaErrors(cudaDeviceSynchronize());
          // LOG("Press Enter to continue.") ;
          // std::cin.ignore();
          //ABORT_IF_EQ(0, 0, "return 0");
        }

      }else{
        if(1) LOG("Doing Left over Block");
        const long long int min_dim_left_over = std::min(left_over,n);
        if (min_dim_left_over <(long long int)2) LOG("WARNING SMALL DIM LEFT OVER");
        checkCudaErrors(cudaMemcpy(A_dev_,  A + block * block_rows * n,  left_over * n * SIZE_OF(float), cudaMemcpyHostToDevice));
 
        if(row_major_ordering){
          gpu_orthogonal_decomp<float>(handle, dn_solver_handle,
                                       n, left_over, num_latent_factors, percent, 
                                       A_dev_, V_dev_, U_dev_, !S_with_U, S_dev_);
        }else{
          gpu_orthogonal_decomp<float>(handle, dn_solver_handle,
                                      left_over, n, num_latent_factors, percent, 
                                      A_dev_, U_dev_, V_dev_, S_with_U, S_dev_);           
        }
        //gpu_swap_ordering(n, min_dim_left_over, V_dev_, false);
        //gpu_swap_ordering(left_over, min_dim_left_over, U_dev_, false);

        // gpu_scale(handle, n * min_dim_left_over, (float)1.0 / (float)num_blocks, V_dev_);
        // gpu_scale(handle, block_rows * min_dim_left_over, (float)num_blocks, U_dev_);
        // gpu_scale(handle, min_dim_left_over, pow((float)num_blocks, (float)2.0), S_dev_);

        checkCudaErrors(cudaMemcpy(V + block * n * min_dim,  V_dev_,  n * min_dim_left_over * SIZE_OF(float), cudaMemcpyDeviceToHost));
        //copy_device_mtx_into_host_submtx<float>(block_rows, min_dim, U_dev_, U + start, m);
        checkCudaErrors(cudaMemcpy(S + block * min_dim,  S_dev_,  left_over * SIZE_OF(float), cudaMemcpyDeviceToHost));
      }
      if(Debug && 0){
        checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
        //omp_set_lock(&printlock);
        //LOG(std::endl<<"block : "<<block);
        //print_host_mtx<float>(A, m, n, "A", 1, strPreamble(blank));
        //print_host_array(S, (int)m, "S", strPreamble(blank));
        //print_host_mtx<float>(U, m, m, "U", 0, strPreamble(blank));
        //print_host_mtx<float>(V, n, m, "V", 0, strPreamble(blank));
        //print_gpu_mtx_entries<float>(U_dev_, block_rows, min_dim, "U", 0, strPreamble(blank));
        //print_gpu_mtx_entries<float>(V_dev_, n, min_dim, "V", 0, strPreamble(blank));
        //omp_unset_lock(&printlock);
      } 
    }


    checkCudaErrors(cudaFree(A_dev_));
    checkCudaErrors(cudaFree(U_dev_)); 
    checkCudaErrors(cudaFree(V_dev_));
    checkCudaErrors(cudaFree(S_dev_));
  //}


  if(too_big(m) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  int* order  = NULL;
  order = (int *)malloc(m *  SIZE_OF(int)); 
  checkErrors(order);
  if(Debug && 0) {
    LOG("Here");
    checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  //gpu_set_as_index<int>(std::min(m,n), order);
  cpu_set_as_index<int>(order, m, 1);
  if(Debug){
    //print_host_array(order, m, "order", strPreamble(blank));
    //print_host_array(S, m, "S", strPreamble(blank));
    save_host_arrays_side_by_side_to_file_(order, order, S, (int)m, "order_S_0");
  }
  cpu_scal<float>(m, (float)(-1.0), S);
  if(Debug && 0) {
    print_host_array<float>(S, m, "S", strPreamble(blank));
    checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  thrust::sort_by_key(thrust::host, S, S + m, order);
  if(Debug && 0) {
    print_host_array<float>(S, m, "S", strPreamble(blank));
    checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  cpu_scal<float>(m, (float)(-1.0), S);
  if(Debug) {
    //print_host_array<float>(S, m, "S", strPreamble(blank));
    save_host_arrays_side_by_side_to_file_(order, order, S, (int)m, "order_S_1");
    //checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  cpu_get_num_latent_factors<float>(m, S, num_latent_factors, percent);
  //LOG("num_latent_factors : "<<num_latent_factors[0]);
  if(num_latent_factors[0] > m){
    // sanity check
    ABORT_IF_EQ(0, 0, "num_latent_factors = "<<num_latent_factors[0]<<" > "<<m<<" = m");
  }
  //print_host_array<float>(S, m, "S", strPreamble(blank));
  if(Debug && 0){
    print_host_array(order, m, ("order"), strPreamble(blank));
    print_host_array<float>(S, m, "S", strPreamble(blank));
    //print_host_mtx<float>(U, m, m, "U", 0, strPreamble(blank));
    //print_host_mtx<float>(V, n, m, "V", 0, strPreamble(blank));
    //checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  //gpu_permute<float>(V, indicies, n, std::min(m,n), 0); 
  
  // permute columns
  //cpu_permute(Dtype* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) <- taking in A in col major ordering
  cpu_permute(V, order, n, m, false);
  //cpu_permute(U, order, m, m, false);

  if(Debug && 0){
    //print_host_mtx<float>(U, m, m, "U", 0, strPreamble(blank));
    //print_host_mtx<float>(V, n, m, "V", 0, strPreamble(blank));
    save_host_mtx_to_file<float>(U, m, m, "U", 0, strPreamble(blank));
    save_host_mtx_to_file<float>(V, n, m, "V", 0, strPreamble(blank));
    save_host_array_to_file<float>(S, m, "S");
    checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");
  }
  free(order);
  //cpu_swap_ordering(n, m, V, 0);

  if(Debug){
    //cpu_swap_ordering(m, m, U, 0);

    float mean_abs_nonzero_ = (float)0.0;
    cpu_mean_abs_nonzero(m * n, V, &mean_abs_nonzero_, true);

    checkCudaErrors(cudaDeviceSynchronize());
    LOG("num_latent_factors : "<<num_latent_factors[0]) ;
    //save_host_mtx_to_file<float>(U, m, m, "U", 1, strPreamble(blank));
    //save_host_mtx_to_file<float>(V, n, m, "V", 0, strPreamble(blank));
    
    //save_device_mtx_to_file<float>(U, m, m, "U_3");
    //save_device_mtx_to_file<float>(V, n, m, "V_3");
 
    int start = 0;
    float epsilon = 0.0001;
    long long int num = std::min((long long int)5, m * n);
    if((int)(m * n - num) > 0){
      getRandIntsBetween(&start, 0, (int)(m * n - num), 1);
    }
    long long int start_ = (long long int)start;
    LOG("start_ : "<<start_) ;
    LOG("num : "<<num) ;
    LOG("m * n : "<<m * n) ;

    float* R  = NULL;
    R = (float *)malloc(num *  SIZE_OF(float)); 
    checkErrors(R);
    //cpu_set_all<float>(R, num, (float)0.0);

    /*
      A is m by n stored in row-maj ordering where m<<n
      V is n by m stored in row-maj ordering
      (V^T is m by n)
      U is m by m stored in row-maj ordering
    
      // M, N, K
      //M number of rows of matrix op(A) and C.
      //N is number of columns of matrix op(B) and C.]
      //K is number of rows of op(B) and columns of op(A).

      // op(A) is M by K
      // op(B) is K by N
      // C is M by N
      // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
      // performs C=alpha op ( B ) op ( A ) + beta C
    */

    /*
      A in row major ordering is equivalent to A^T in column major ordering
      A in column major ordering is equivalent to A^T in row major ordering
    */

    // cpu_gemm<float>(true, false, m, m, m /*num_latent_factors[0]*/,
    //  (float)1.0, U, U, (float)0.0, R);
    // print_host_mtx<float>(R, m, m, "UTU", 1, strPreamble(blank));

    // cpu_gemm<float>(false, true, m, m, m /*num_latent_factors[0]*/,
    //  (float)1.0, U, U, (float)0.0, R);
    // print_host_mtx<float>(R, m, m, "UUT", 1, strPreamble(blank));

    // cpu_gemm<float>(true, false, m, m, n /*num_latent_factors[0]*/,
    //  (float)1.0, V, V, (float)0.0, R);
    // print_host_mtx<float>(R, m, m, "VTV", 1, strPreamble(blank));

    // cpu_gemm<float>(false, true, n, n, m /*num_latent_factors[0]*/,
    //  (float)1.0, V, V, (float)0.0, R);
    // print_host_mtx<float>(R, n, n, "VVT", 1, strPreamble(blank));

    cpu_gemm<float>(true, false, m, n, m /*num_latent_factors[0]*/,
     (float)1.0, U, V, (float)0.0, R, start_, num);

    save_host_arrays_side_by_side_to_file<float>(A + start_, R, (int)num, "real_factored", strPreamble(blank));

    cpu_axpby<float>(num, (float)1.0, A + start_, (float)(-1.0), R);

    //print_host_mtx<float>(R, m, n, "svd_error", 1, strPreamble(blank));

    //float range_A    = gpu_range<float>(m * n,  A);
    float error      = cpu_abs_max<float>(num, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
    
    // LOG("A mtx range of values = "<<range_A) ;
    LOG("SVD max error = "<<ToString<float>(error)) ;
    if(error > epsilon){
      ABORT_IF_EQ(0, 0, "gpu_block_orthogonal_decomp_from_host factoring incorrectly");
    }
    free(R);
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
  

    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
    
  } 
  //save_host_array_to_file<float>(S, m, "S");
  if(S == NULL){
    free(S_temp);
  }

  if(Debug) LOG("finished call to gpu_block_orthogonal_decomp_from_host") ;
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  /*if(Debug)*/ LOG("gpu_block_orthogonal_decomp_from_host run time : "<<readable_time(program_time)<<std::endl);
}


void gpu_block_orthogonal_decomp_from_host_test(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle) 
{
  const long long int m = 27000;
  const long long int n = 26744;
  const long long int block_rows = 200;
  bool S_with_U = false;
  const long long int min_dim = std::min(m,n);
  const long long int max_dim = std::max(m,n);
  std::string blank = "";

  bool row_major_ordering = true;

  float* A = NULL;
  float* U = NULL;
  float* V = NULL;
  float* S = NULL;
  A = (float *)malloc(m*n *  SIZE_OF(float)); 
  U = (float *)malloc(m*m *  SIZE_OF(float)); 
  V = (float *)malloc(n*m *  SIZE_OF(float)); 
  S = (float *)malloc(m *  SIZE_OF(float)); 
  checkErrors(A);
  checkErrors(U);
  checkErrors(V);
  checkErrors(S);

  cpu_set_all<float>(U, m*m, (float)0.0);
  cpu_set_all<float>(V, m*n, (float)0.0);
  cpu_set_all<float>(S, m, (float)0.0);

  host_rng_uniform<float>(m*n, (float)(-1.0), (float)1.0, A);

  //save_host_mtx_to_file<float>(A, m, n, "A");

  long long int num_latent_factors;
  const float percent = (float)0.95;

  gpu_block_orthogonal_decomp_from_host<float>(handle, dn_solver_handle, m, n, &num_latent_factors, percent,
                                               A, U, V, row_major_ordering, block_rows, S_with_U, S);

  //cpu_swap_ordering(const long long int rows, const long long int cols, Dtype *A, const bool row_major_ordering)
  cpu_swap_ordering(m, m, U, 0);
  cpu_swap_ordering(n, m, V, 0);

  if(0){
    print_host_mtx<float>(U, m, m, "U", 1, strPreamble(blank));
    print_host_mtx<float>(V, n, m, "V", 1, strPreamble(blank));
  }

  float* R  = NULL;
  R = (float *)malloc(max_dim * max_dim *  SIZE_OF(float)); 
  checkErrors(R);

  /*
    A is m by n stored in row-maj ordering where m<<n
    V is n by m stored in row-maj ordering
    (V^T is m by n)
    U is m by m stored in row-maj ordering
  
    // M, N, K
    //M number of rows of matrix op(A) and C.
    //N is number of columns of matrix op(B) and C.]
    //K is number of rows of op(B) and columns of op(A).

    // op(A) is M by K
    // op(B) is K by N
    // C is M by N
    // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C
  */

  /*
    A in row major ordering is equivalent to A^T in column major ordering
    A in column major ordering is equivalent to A^T in row major ordering
  */

  /*
    cpu_gemm<float>(true, false, m, m, m, // <- num_latent_factors[0]
     (float)1.0, U, U, (float)0.0, R);
    print_host_mtx<float>(R, m, m, "UTU", 1, strPreamble(blank));

    cpu_gemm<float>(false, true, m, m, m, // <- num_latent_factors[0]
     (float)1.0, U, U, (float)0.0, R);
    print_host_mtx<float>(R, m, m, "UUT", 1, strPreamble(blank));

    cpu_gemm<float>(true, false, m, m, n, // <- num_latent_factors[0]
     (float)1.0, V, V, (float)0.0, R);
    print_host_mtx<float>(R, m, m, "VTV", 1, strPreamble(blank));

    cpu_gemm<float>(false, true, n, n, m, // <- num_latent_factors[0]
     (float)1.0, V, V, (float)0.0, R);
    print_host_mtx<float>(R, n, n, "VVT", 1, strPreamble(blank));
  */


  cpu_gemm<float>(false, true, m, n, m /*num_latent_factors[0]*/,
   (float)1.0, U, V, (float)0.0,
   R);

  cpu_axpby<float>(m * n, (float)1.0, A, (float)(-1.0), R);

  //print_host_mtx<float>(R, m, n, "svd_error", 1, strPreamble(blank));

  //float range_A    = gpu_range<float>(m * n,  A);
  float error      = cpu_abs_max<float>(m * n, (const float*)R); 
  //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
  free(R);
  // LOG("A mtx range of values = "<<range_A) ;
  LOG("SVD max error = "<<ToString<float>(error)) ;
  // LOG("SVD max error over range of values = "<<error/range_A) ;
  // LOG("SVD expected absolute error = "<<error_expt) ;
  // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
  // LOG("Press Enter to continue.") ;
  // std::cin.ignore();

  free(A);
  free(U);
  free(V);
  free(S);


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
  if(too_big( (long long int)lda * (long long int)n) ) ABORT_IF_EQ(0, 0, "too big");
  CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

  checkCudaErrors(cudaMalloc(&info, SIZE_OF(int)));
  checkCudaErrors(cudaMalloc(&buffer, SIZE_OF(float)*bufferSize));
  checkCudaErrors(cudaMalloc(&A, SIZE_OF(float)*lda*n));
  checkCudaErrors(cudaMalloc(&ipiv, SIZE_OF(int)*n));


  // prepare a copy of A because getrf will overwrite A with L
  checkCudaErrors(cudaMemcpy(A, Acopy, SIZE_OF(float)*lda*n, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemset(info, 0, SIZE_OF(int)));

  // start = second();
  // start = second();

  CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info));
  checkCudaErrors(cudaMemcpy(&h_info, info, SIZE_OF(int), cudaMemcpyDeviceToHost));

  if ( 0 != h_info ){
    fprintf(stderr, "Error: LU factorization failed\n");
  }

  checkCudaErrors(cudaMemcpy(x, b, SIZE_OF(float)*n, cudaMemcpyDeviceToDevice));
  CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
  checkCudaErrors(cudaDeviceSynchronize());

  // stop = second();
  // time_solve = stop - start;
  // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

  if (info  ) { checkCudaErrors(cudaFree(info  )); }
  if (buffer) { checkCudaErrors(cudaFree(buffer)); }
  if (A     ) { checkCudaErrors(cudaFree(A)); }
  if (ipiv  ) { checkCudaErrors(cudaFree(ipiv)); }

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
  if(too_big( (long long int)lda * (long long int)n) ) ABORT_IF_EQ(0, 0, "too big");
  CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

  checkCudaErrors(cudaMalloc(&info, SIZE_OF(int)));
  checkCudaErrors(cudaMalloc(&buffer, SIZE_OF(double)*bufferSize));
  checkCudaErrors(cudaMalloc(&A, SIZE_OF(double)*lda*n));
  checkCudaErrors(cudaMalloc(&ipiv, SIZE_OF(int)*n));


  // prepare a copy of A because getrf will overwrite A with L
  checkCudaErrors(cudaMemcpy(A, Acopy, SIZE_OF(double)*lda*n, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemset(info, 0, SIZE_OF(int)));

  // start = second();
  // start = second();

  CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
  checkCudaErrors(cudaMemcpy(&h_info, info, SIZE_OF(int), cudaMemcpyDeviceToHost));

  if ( 0 != h_info ){
    fprintf(stderr, "Error: LU factorization failed\n");
  }

  checkCudaErrors(cudaMemcpy(x, b, SIZE_OF(double)*n, cudaMemcpyDeviceToDevice));
  CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
  checkCudaErrors(cudaDeviceSynchronize());
  // stop = second();

  // time_solve = stop - start;
  // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

  if (info  ) { checkCudaErrors(cudaFree(info  )); }
  if (buffer) { checkCudaErrors(cudaFree(buffer)); }
  if (A     ) { checkCudaErrors(cudaFree(A)); }
  if (ipiv  ) { checkCudaErrors(cudaFree(ipiv)); }

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
    if(too_big( (long long int)lda * (long long int)n) ) ABORT_IF_EQ(0, 0, "too big");
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

    checkCudaErrors(cudaMalloc(&info, SIZE_OF(int)));
    checkCudaErrors(cudaMalloc(&buffer, SIZE_OF(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, SIZE_OF(float)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, SIZE_OF(float)*n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, SIZE_OF(float)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, SIZE_OF(int)));

    // start = second();
    // start = second();

    // compute QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, SIZE_OF(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
      fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, SIZE_OF(float)*n, cudaMemcpyDeviceToDevice));

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

  if(too_big( (long long int)lda * (long long int)n) ) ABORT_IF_EQ(0, 0, "too big");

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

  checkCudaErrors(cudaMalloc(&info, SIZE_OF(int)));
  checkCudaErrors(cudaMalloc(&buffer, SIZE_OF(double)*bufferSize));
  checkCudaErrors(cudaMalloc(&A, SIZE_OF(double)*lda*n));
  checkCudaErrors(cudaMalloc ((void**)&tau, SIZE_OF(double)*n));

  // prepare a copy of A because getrf will overwrite A with L
  checkCudaErrors(cudaMemcpy(A, Acopy, SIZE_OF(double)*lda*n, cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemset(info, 0, SIZE_OF(int)));

  // start = second();
  // start = second();

  // compute QR factorization
  CUSOLVER_CHECK(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

  checkCudaErrors(cudaMemcpy(&h_info, info, SIZE_OF(int), cudaMemcpyDeviceToHost));

  if ( 0 != h_info ){
    fprintf(stderr, "Error: LU factorization failed\n");
  }

  checkCudaErrors(cudaMemcpy(x, b, SIZE_OF(double)*n, cudaMemcpyDeviceToDevice));

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

  CUDA_CHECK(cudaMemcpy(&first_entry_index, csr + first_row, SIZE_OF(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_entry_index, csr + last_row + 1, SIZE_OF(int), cudaMemcpyDeviceToHost));

  return last_entry_index - first_entry_index;


}

int gpu_get_first_coo_index(const int first_row, const int* csr)
{
  int first_entry_index;

  CUDA_CHECK(cudaMemcpy(&first_entry_index, csr + first_row, SIZE_OF(int), cudaMemcpyDeviceToHost));

  return first_entry_index;


}


template<typename Dtype>
__global__ void sparse_error_kernel(const long long int rows, const long long int cols, const Dtype* dense_mtx_A, 
 const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, Dtype* coo_errors, const int num_sparse_entries, 
 bool* isBad, Dtype* coo_A)
{
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  const int row_skip = csr_rows_B[0];
  CUDA_KERNEL_LOOP(j, num_sparse_entries) {
    int count = 0;
    long long int row = (long long int)0;
    // while(count < j + 1){
    //   count += csr_rows_B[row + 1] - csr_rows_B[row]; //num thing in that row
    //   row +=1;
    // }
    for(row = (long long int)0; row < rows; row+=(long long int)1){
      count += csr_rows_B[(int)row + 1] - csr_rows_B[row]; //add the number of things in this row to the count
      if(count >= j + 1) break; //you're in the right row
    }
    // count >= j + 1
    long long int col = (long long int)(coo_cols_B[/*row_skip + */ j]);
    coo_errors[j] = coo_entries_B[/*row_skip + */ j] - dense_mtx_A[row + col * rows];
    //coo_errors[j] = -coo_entries_B[/*row_skip + */ j] + dense_mtx_A[row + col * rows];

    if(coo_A != NULL){
      //coo_A[j] = (Dtype)row;
      coo_A[j] = dense_mtx_A[row + col * rows];
    }
    
    if (::isinf(coo_errors[j]) || ::isnan(coo_errors[j])){
      isBad[0] = true;
    };
  };
}

void sparse_error(const long long int rows, const long long int cols, const float* dense_mtx_A, 
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
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  // if(coo_A == NULL){
  //   Debug = false;
  // }else{
  //   Debug = true;
  // }

  if(Debug) LOG("sparse_error called");


  const long long int num_gpu_blocks = GET_BLOCKS((long long int)num_sparse_entries);

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
                                                                  isBad, coo_A);
  if(Debug) {
    save_device_array_to_file<float>(coo_errors, num_sparse_entries, "coo_errors_rows");
    LOG("Press Enter to continue.") ;
    std::cin.ignore();
  };    
  if(Debug && 0)LOG("Here!");
  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
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
  CUDA_CHECK(cudaMalloc((void**)&csrRowPtrA_temp, (m+1) * SIZE_OF(float)));
  checkCudaErrors(cudaMemcpy(csrRowPtrA_temp, csrRowPtrA, (m+1)  *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
  gpu_add_scalar<int>(m + 1, (int)(-1) * (int)first_ind, csrRowPtrA_temp);

  float* full_A;
  CUDA_CHECK(cudaMalloc((void**)&full_A, (long long int)m * (long long int)k * SIZE_OF(float)));
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

  checkCudaErrors(cudaFree(full_A));
  checkCudaErrors(cudaFree(csrRowPtrA_temp));
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
  std::string blank = "";
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
  if(too_big( (long long int)m * (long long int)k) ) ABORT_IF_EQ(0, 0, "too big");
  if(too_big( (long long int)m * (long long int)n) ) ABORT_IF_EQ(0, 0, "too big");
  if(too_big( (long long int)n * (long long int)k) ) ABORT_IF_EQ(0, 0, "too big");

  int * csrRowPtrA_temp;
  CUDA_CHECK(cudaMalloc((void**)&csrRowPtrA_temp, (m+1) * SIZE_OF(float)));
  checkCudaErrors(cudaMemcpy(csrRowPtrA_temp, csrRowPtrA, (m+1)  *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
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
        save_device_mtx_to_file<float>(B, n, m, "B", false, strPreamble(blank));
      }else{
        LOG("B is "<<m<<" by "<< n);
        save_device_mtx_to_file<float>(B, m, n, "B", false, strPreamble(blank));
      };
      LOG("C is "<<k<<" by "<< n);
    }else{
      LOG("A is "<<m<<" by "<< k<<" in CSR format");
      if(TransB){
        LOG("B is "<<n<<" by "<< k <<" in dense format but it will be tranposed");
        save_device_mtx_to_file<float>(B, n, m, "B", false, strPreamble(blank));
      }else{
        LOG("B is "<<k<<" by "<< n);
        save_device_mtx_to_file<float>(B, m, n, "B", false, strPreamble(blank));
      };
      LOG("C is "<<m<<" by "<< n);
    };
    checkCudaErrors(cudaDeviceSynchronize());
    save_device_array_to_file<int>(csrRowPtrA, m + 1, "csrRowPtrA", strPreamble(blank));
    save_device_array_to_file<int>(csrColIndA, nnz, "csrColIndA", strPreamble(blank));
    save_device_array_to_file<float>(csrValA, nnz, "csrValA", strPreamble(blank));
    
    //save_device_mtx_to_file<float>(C, k, n, "C");

    checkCudaErrors(cudaDeviceSynchronize());
  // LOG("Press Enter to continue.") ;
  // std::cin.ignore();
  }

  cusparseOperation_t cuTransA =
  (TransA == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
  (TransB == false) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseScsrmm2(handle, cuTransA, cuTransB, 
                                  m, n, k, nnz, alpha, descrA, 
                                  csrValA, csrRowPtrA_temp, csrColIndA, 
                                  B, ldb, beta, C, ldc) );

  checkCudaErrors(cudaFree(csrRowPtrA_temp));

  bool isBad = gpu_isBad<float>(C, (TransA == true) ? ((long long int)n * (long long int)k) : ( (long long int)m * (long long int)n) );
  if(isBad){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  if(Debug) LOG("gpu_spXdense_MMM call finished");
}

template<typename Dtype>
__global__ void gpu_logarithmic_histogram_abs_val_kernel(const long long int n, Dtype* error, Dtype* probability, 
                                                          int min_pow, int max_pow, int non_zero_count)
{
  CUDA_KERNEL_LOOP(i, (int)n) {
    if( (Dtype)0.0 < abs(error[i]) && abs(error[i]) < (Dtype)pow((Dtype)10.0, (Dtype)min_pow) ) {
      probability[0] += (Dtype)1.0 / (Dtype)non_zero_count;
    }else if((Dtype)0.0 < abs(error[i])){
      int count = 1;
      for(int j = min_pow + 1; j <= max_pow; j++){
        if( (Dtype)pow((Dtype)10.0, (Dtype)(j - 1))<= abs(error[i]) && abs(error[i]) < (Dtype)pow((Dtype)10.0, (Dtype)j) ) {
          probability[count] += (Dtype)1.0 / (Dtype)non_zero_count;
          break;
        }else{
          count++;
        }
      }
    }
  }
}

template <typename Dtype>
void gpu_logarithmic_histogram_abs_val(const long long int n, Dtype* error_dev, Dtype* probability, 
                                      int min_pow, int max_pow, int non_zero_count)
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

  Dtype* probability_dev;
  checkCudaErrors(cudaMalloc((void**)&probability_dev, (max_pow - min_pow + 1) * SIZE_OF(Dtype)));
  gpu_set_all(probability_dev, (long long int)(max_pow - min_pow + 1), (Dtype)0.0);

  gpu_logarithmic_histogram_abs_val_kernel<<<1, 1>>>( n, error_dev, probability_dev, min_pow, max_pow, non_zero_count);

  CUDA_CHECK(cudaMemcpy(probability, probability_dev, (max_pow - min_pow + 1) * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
  cudaFree(probability_dev);
}

template void gpu_logarithmic_histogram_abs_val<float>(const long long int n, float* error_dev, float* probability, 
                                                int min_pow, int max_pow, int non_zero_count);




template <>
void gpu_R_error<float>(cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr, cusolverDnHandle_t dn_solver_handle,
                      const long long int batch_size_t, const long long int batch_size_ACU, 
                      const long long int num_latent_factors, const long long int ratings_cols,
                      const int nnz, const int first_coo_ind, const bool compress, 
                      float* training_entries, float* coo_errors, const float testing_fraction,
                      const float *coo_format_ratingsMtx_rating_dev_batch, 
                      const int *csr_format_ratingsMtx_userID_dev_batch, 
                      const int *coo_format_ratingsMtx_itemID_dev_batch,
                      float *V, float *U_t, float *R_t, float *U_ACU, float *R_ACU, 
                      float training_rate, float regularization_constant, const int increment_index, int training_iteration,
                      float* testing_error_on_training_entries, float* testing_error_on_testing_entries, 
                      float* total_iterations, bool S_with_U, float *SV, float* logarithmic_histogram)
{
  bool Debug  = false;
  bool print_ = false;
  if(print_) {
    LOG("gpu_R_error called");
  }
  LOG("initial training_rate : "<<training_rate);
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  //gpu_sum_of_squares_test(); 

  //============================================================================================
  // Initialize  U_t randomly 
  //============================================================================================ 
  /*
    R_t is sparse batch_size_t by ratings_cols with nnz entries
    V is dense ratings_cols by batch_size_ACU (or num_latent_factors)

    Here we want to further sparsify R_t to measure 
    the prediction on entries we already know

    gpu_spXdense_MMM performs
    C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C


    notes: 

    V^T * V   is the identity mtx if compress==false
    V   * V^T is not the identity mtx 
  */
  float *coo_R = NULL;
  checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * SIZE_OF(float)));

  float *training_entries_cpy = NULL; 
  bool testing = (testing_fraction > (float)0.0);
  if(testing){
    gpu_get_rand_bools<float>(nnz,  training_entries, (float)1.0 - testing_fraction /*probability of 1*/);
    checkCudaErrors(cudaMalloc((void**)&training_entries_cpy, nnz * SIZE_OF(float)));
    checkCudaErrors(cudaMemcpy(training_entries_cpy, training_entries, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));
    if(Debug && 0){
      save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, nnz, "ratings_before_hadamard");
    }

    gpu_hadamard<float>(nnz, coo_format_ratingsMtx_rating_dev_batch, training_entries );

    if(Debug && 0){
      save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, nnz, "ratings_after_hadamard");
    }
  }else{
    training_entries = (float *)coo_format_ratingsMtx_rating_dev_batch;
    if(Debug){
      LOG("training_entries : "<<training_entries);
    }
  }

  float* km_errors;
  int* km_selection;
  if(!testing){
    checkCudaErrors(cudaMalloc((void**)&km_errors, batch_size_t * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&km_selection, batch_size_t * SIZE_OF(int)));
  }


  float alpha = (float)1.0;
  float beta = (float)1.0;
  // float *alpha_dev;
  // float *beta_dev;
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, SIZE_OF(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, SIZE_OF(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, SIZE_OF(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, SIZE_OF(float), cudaMemcpyHostToDevice));
  // if(alpha_dev) checkCudaErrors(cudaFree(alpha_dev));
  // if(beta_dev) checkCudaErrors(cudaFree(beta_dev));

  // float *U_t_check;
  // CUDA_CHECK( cudaMalloc( (void**)&U_t_check, batch_size_t * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float) ) );
  // float *U_ACU_check;
  // CUDA_CHECK( cudaMalloc( (void**)&U_ACU_check, batch_size_ACU * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float) ) );
  // float *V_check;
  // CUDA_CHECK( cudaMalloc( (void**)&V_check, ratings_cols * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float) ) );

  float* U_t_check   = NULL;
  float* U_ACU_check  = NULL;
  float* V_check     = NULL;
  U_t_check  = (float *)  malloc(batch_size_t * (compress ? num_latent_factors : batch_size_ACU) *  SIZE_OF(float)); 
  checkErrors(U_t_check);
  if(!testing) {
    U_ACU_check = (float *)  malloc(batch_size_ACU * (compress ? num_latent_factors : batch_size_ACU) *  SIZE_OF(float)); 
    V_check    = (float *)  malloc(ratings_cols * (compress ? num_latent_factors : batch_size_ACU) *  SIZE_OF(float)); 
    checkErrors(U_ACU_check);
    checkErrors(V_check);
  }

  if(Debug && 0){


    if(compress){
      LOG("num_latent_factors : "<<num_latent_factors);
      //CUDA_CHECK( cudaMalloc( (void**)&U_t_check, batch_size_t * num_latent_factors * SIZE_OF(float) ) );
      LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
        <<"m = "<<batch_size_t<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
        <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
        <<"&alpha, sp_descr,"<<std::endl<<"                                                "
        <<"training_entries,csr_format_ratingsMtx_userID_dev_batch, coo_format_ratingsMtx_itemID_dev_batch,"<<std::endl<<"                                                "
        <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
        <<"U_t, ldc = "<<batch_size_t<<" );"  );
    }else{
      //CUDA_CHECK( cudaMalloc( (void**)&U_t_check, batch_size_t * batch_size_ACU  * SIZE_OF(float) ) );
      LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
        <<"m = "<<batch_size_t<<" , n = "<<batch_size_ACU<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
        <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
        <<"&alpha, sp_descr,"<<std::endl<<"                                                "
        <<"training_entries,csr_format_ratingsMtx_userID_dev_batch, coo_format_ratingsMtx_itemID_dev_batch,"<<std::endl<<"                                                "
        <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
        <<"U_t, ldc = "<<batch_size_t<<" );"  );    
    }

    // gpu_spXdense_MMM_check<float>(dn_handle, false, false, batch_size_t,
    //                              (compress == false) ? batch_size_ACU : num_latent_factors, 
    //                              ratings_cols, first_coo_ind, alpha, 
    //                              training_entries, 
    //                              csr_format_ratingsMtx_userID_dev_batch, 
    //                              coo_format_ratingsMtx_itemID_dev_batch,
    //                              V, beta, U_t_check);
  }



  // gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_t,
  //  (compress == false) ? batch_size_ACU : num_latent_factors, 
  //  ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
  //  training_entries, 
  //  csr_format_ratingsMtx_userID_dev_batch, 
  //  coo_format_ratingsMtx_itemID_dev_batch,
  //  V, ratings_cols, &beta, U_t, batch_size_t, Debug);
  bool knowledgeable = false;
  //if(!testing && increment_index == 0){
  if(increment_index == 0) {
    knowledgeable = false;
  }

  //if(testing || increment_index == 0){

    if(!knowledgeable){
      gpu_rng_gaussian<float>(batch_size_t * (compress ? num_latent_factors : batch_size_ACU), (float)0.0, (float)1.0, U_t);
    }else{
      LOG("Starting from an educated guess...");
      //gpu_rng_gaussian<float>(batch_size_t * ratings_cols, (float)0.0, (float)0.00007, R_t);
      //gpu_set_all(R_t, batch_size_t * ratings_cols, (float)0.0);
      if(Debug && 0){
        save_device_mtx_to_file<float>(R_t, batch_size_t, ratings_cols, "R_t_0", false, strPreamble(blank));
      }
      if(testing){
        gpu_fill_training_mtx_if(batch_size_t, ratings_cols, false,
                                csr_format_ratingsMtx_userID_dev_batch,
                                coo_format_ratingsMtx_itemID_dev_batch,
                                training_entries, training_entries_cpy,
                                R_t);
        if(Debug && 0){
          save_device_mtx_to_file<float>(R_t, batch_size_t, ratings_cols, "R_t_1", false, strPreamble(blank));
        }
      }else{
        gpu_fill_training_mtx(batch_size_t, ratings_cols, false,
                              csr_format_ratingsMtx_userID_dev_batch,
                              coo_format_ratingsMtx_itemID_dev_batch,
                              training_entries,
                              R_t);
      }
      // M, N, K
      // M number of columns of matrix op(A) and C.
      // N is number of rows of matrix op(B) and C.]
      // K is number of columns of op(B) and rows of op(A).

      // op(A) is K by M
      // op(B) is N by K
      // C is N by M
      // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
      // performs C=alpha op ( B ) op ( A ) + beta C
      gpu_gemm<float>(dn_handle, false, false,
                      compress ? num_latent_factors : batch_size_ACU, 
                      batch_size_t, ratings_cols,
                      (float)1.0, V, R_t, (float)0.0, U_t);
      if(Debug && 0){
        save_device_mtx_to_file<float>(U_t, batch_size_t, compress ? num_latent_factors : batch_size_ACU, "U_t_0", false, strPreamble(blank));
        save_device_mtx_to_file<float>(V, ratings_cols, compress ? num_latent_factors : batch_size_ACU, "V", false, strPreamble(blank));
      }
      // U_t is batch_size_t * (compress ? num_latent_factors : batch_size_ACU)
      // V is ratings_cols * (compress ? num_latent_factors : batch_size_ACU)
      // R_t is batch_size_t * ratings_cols
      if(!S_with_U){
        if(SV != NULL){
          int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_ACU, SV, (float)0.0001);
          if(Debug && 0){
            LOG("first_tiny_sv_ : "<<first_tiny_sv_)
          }
          if( first_tiny_sv_ < ( (int)(compress ? num_latent_factors : batch_size_ACU) ) ){
            LOG("WARNING WILL DIVIDE BY ~ZERO");
            long long int temp = (compress == false) ? batch_size_ACU : num_latent_factors;
            LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
          } 

          gpu_div_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true );//right div
          if(Debug && 0){
            save_device_array_to_file<float>(SV, compress ? num_latent_factors : batch_size_ACU, "SV", strPreamble(blank));
            save_device_mtx_to_file<float>(U_t, batch_size_t, compress ? num_latent_factors : batch_size_ACU, "U_t_1", false, strPreamble(blank));
          }
          gpu_div_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true );//right div
          if(Debug && 0){
            save_device_mtx_to_file<float>(U_t, batch_size_t, compress ? num_latent_factors : batch_size_ACU, "U_t_2", false, strPreamble(blank));
          }
        }else{
          // hmm
        }
      }
    }
  //}
  
  float largest_sv = (float)0.0;

  if(Debug){
    LOG("S_with_U : "<<S_with_U)
    LOG("SV != NULL : "<<(SV != NULL))
  }
  //============================================================================================
  // Handle column normalization in U_t
  //============================================================================================ 

  if(S_with_U){
    if(SV != NULL){
      int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_ACU, SV, (float)0.0001);
      if(Debug){
        LOG("first_tiny_sv_ : "<<first_tiny_sv_)
      }
      if(first_tiny_sv_ < (int)(compress ? num_latent_factors : batch_size_ACU)){
        LOG("WARNING WILL DIVIDE BY ~ZERO");
        long long int temp = (compress == false) ? batch_size_ACU : num_latent_factors;
        LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
      }
      gpu_div_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true );//right div

      //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
      //                              bool row_major_ordering, float* x, bool normalize_rows);
      gpu_normalize_mtx_rows_or_cols(batch_size_t, compress ? num_latent_factors : batch_size_ACU,  
                                        false, U_t, false);

      gpu_mult_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true ); //right mult
      checkCudaErrors(cudaMemcpy(&largest_sv, SV, SIZE_OF(float), cudaMemcpyDeviceToHost));
      if(Debug){LOG("largest_sv : "<<largest_sv);}
    }else{
      largest_sv = gpu_norm(dn_handle, batch_size_t, U_t);
      //more thinking here:force order
      float current_largest_sv = gpu_norm(dn_handle, batch_size_t, U_t);
      gpu_scale(dn_handle, batch_size_t * ( compress ? num_latent_factors : batch_size_ACU), largest_sv/current_largest_sv, U_t);
    }
  }else{
    //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
    //                              bool row_major_ordering, float* x, bool normalize_rows);
    if(Debug) LOG("here");
    gpu_normalize_mtx_rows_or_cols(batch_size_t, compress ? num_latent_factors : batch_size_ACU, false, U_t, false);      
  } 



  // V^T * V
  // gpu_gemm<float>(dn_handle, false, true, (compress == false) ? batch_size_ACU : num_latent_factors,
  //                 (compress == false) ? batch_size_ACU : num_latent_factors, ratings_cols, 
  //                 (float)1.0, V, V, (float)0.0, );




  if(Debug){
    //save_device_mtx_to_file<float>(U_t, batch_size_t, (compress ? num_latent_factors : batch_size_ACU), "U_t", false, strPreamble(blank));
    //save_device_mtx_to_file<float>(V, ratings_cols, (compress ? num_latent_factors : batch_size_ACU), "V", false, strPreamble(blank));
    
    //save_device_arrays_side_by_side_to_file<float>(U_t, U_t_check, batch_size_t * (compress ? num_latent_factors : batch_size_ACU), "U_t");
    //save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_batch, training_entries, nnz, "R_t");

    LOG("compress : "<<compress)
    //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, coo_errors, nnz, "ratings_testing_errors_v2");
  }

  int i = 0;
  bool not_done = true;
  float error;
  float min_error_so_far = (float)10000.0;
  int min_error_iter = i;
  int unchanged_error_count = 0;
  float min_training_rate = training_rate / ((float)10000.0);

  int num_steps =  log10((int)round(training_rate / min_training_rate));

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;
  int max_its = 50;
  if(testing){
    max_its = 20000;
  }

  int training_its = 1;
  training_its = 4*(training_its + 1);

  float epsilon = (float)0.09;
  if(testing){
    epsilon = (float)0.2;
  }

  float* error_vector = NULL;
  error_vector = (float *)malloc((max_its + training_its) * SIZE_OF(float));
  checkErrors(error_vector);
  float* micro_km_error = NULL;
  micro_km_error = (float *)malloc((training_its) * SIZE_OF(float));
  checkErrors(micro_km_error);

  int num_its_per_step = max_its / num_steps;
  if(Debug){
    LOG("training_rate : " <<training_rate);
    LOG("min_training_rate : " <<ToString(min_training_rate));
    LOG("(training_rate / min_training_rate) : "<<(training_rate / min_training_rate));
    LOG("num_steps : " <<num_steps);
    LOG("not_done : "<< not_done);
    LOG("i < max_its : "<< (i < max_its));
    LOG("training_rate >= min_training_rate : "<< (training_rate >= min_training_rate));
  }
  bool has_slowed = false;
  bool has_slowed_last = false;
  int max_num_slow = 150; // 50 **submitted
  if(!testing) max_num_slow = 10;
  int has_slowed_count = 0;

  float training_rate_U_ACU = (float)0.00005; // submitted version uses (float)1.0 / (float)45000.0;
  float training_rate_V = (float)0.0001; // submitted version uses (float)0.00001

  float avg_chg_during_U_ACU_update_it = (float)0.0; 
  float avg_chg_during_V_update_it = (float)0.0; 
  float avg_chg_during_U_t_update_it = (float)0.0; 

  int U_ACU_update_it_count = 0;
  int V_update_it_count = 0;
  int U_t_update_it_count = 0;

  float* Beta = NULL;
  float radius_around_min = (float)0.0001;// (float)0.01; //****subbmitted
  if(!testing) {
    radius_around_min = (float)0.005;
    checkCudaErrors(cudaMalloc((void**)&Beta, (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float)));
    gpu_set_as_func_of_index<float>(Beta, (compress ? num_latent_factors : batch_size_ACU), (float)1.0,  (float)1.0 - training_rate * regularization_constant);
    if(1){
      save_device_array_to_file<float>(Beta, (compress ? num_latent_factors : batch_size_ACU), "V_regularizing_array");
    }
  }

  bool updated_min_error_so_far = false;
  int first_updating_iteration = 0;
  bool bad_run = false;





  //========================================================================================================================================================================================
  //========================================================================================================================================================================================
  //========================================================================================================================================================================================
  //========================================================================================================================================================================================
  //========================================================================================================================================================================================

  while(not_done && i < max_its /*&& training_rate >= min_training_rate*/){

    //============================================================================================
    // Compute  R_t_predict = U_t * V^T 
    //============================================================================================ 
    // M, N, K
    // M number of columns of matrix op(A) and C.
    // N is number of rows of matrix op(B) and C.]
    // K is number of columns of op(B) and rows of op(A).

    // op(B) is N by K
    // op(A) is K by M
    // C is N by M
    // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C
    gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_t, 
                    compress ? num_latent_factors : batch_size_ACU,
                    (float)1.0, V, U_t, (float)0.0, R_t);

    if(1){
      bool isBad = gpu_isBad<float>(R_t, batch_size_t * ratings_cols);
      if(isBad){
        ABORT_IF_NEQ(0, 1, "isBad");
      };
      //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, coo_errors, nnz, "ratings_testing_errors_v3");
    }

    //============================================================================================
    // Compute  Error = R_t_actual -  R_t_predict  <-- (Error is sparse)
    //============================================================================================ 
    /*
        Here we want to measure the error on the places we know 
        that were predicted in the previous step
    */ 
    sparse_error(batch_size_t, ratings_cols, R_t, 
                 csr_format_ratingsMtx_userID_dev_batch, 
                 coo_format_ratingsMtx_itemID_dev_batch,
                 training_entries, 
                 coo_errors, nnz, coo_R);
    if(0){
      bool isBad = gpu_isBad<float>(coo_errors, nnz);
      if(isBad){
        ABORT_IF_NEQ(0, 1, "isBad");
      };
    }
    if(testing) gpu_hadamard<float>(nnz, training_entries_cpy, coo_errors );
    float training_error_temp;// = gpu_sum_of_squares<float>(nnz, coo_errors);
    if( Debug)  LOG("gpu_R_error iteration : "<<i);
    long long int nnz_ = (long long int)((float)nnz * ((float)1.0 - testing_fraction));

    bool do_ = true;
    float temp;// = training_error_temp / (float)(nnz_);
    gpu_mean_abs_nonzero(nnz, coo_errors, &temp);

    if(i > 0){
      if(Debug){
        LOG("temp : "<<temp);
        //LOG("temp < epsilon : "<<(temp < epsilon));
        //LOG("::isinf(temp) : "<<::isinf(temp));
        //LOG("::isnan(temp) : "<<::isnan(temp));
        //LOG("temp < epsilon || ::isinf(temp) || ::isnan(temp) : "<<(temp < epsilon || ::isinf(temp) || ::isnan(temp)));
      }
      if(temp < epsilon || ::isinf(temp) || ::isnan(temp) || i == max_its - 1){
        if(!testing && !has_slowed){
          has_slowed = true;
          max_its = i + training_its;
          if(print_) LOG("forced has_slowed = true at iteration "<< i);
          bad_run = true;
        }else{
          if(print_ && 0) {
            LOG("gpu_R_error finished at iteration : "<<i);
          }
          error = temp;
          error_vector[i] = error;
          do_ = false;
          not_done = false;
          break;
        }
      }
      if(Debug) {
        LOG("std::abs(temp - min_error_so_far) : "<<std::abs(temp - min_error_so_far));
        LOG("((float)0.05 *  min_error_so_far)) : "<<((float)0.05 *  min_error_so_far));
        LOG("unchanged_error_count : "<<unchanged_error_count);
      }      
      if( (std::abs(temp - error_vector[i - 1 - unchanged_error_count]) < (radius_around_min *  min_error_so_far)) ){
        unchanged_error_count++;
        // have we stopped improving?
        if(unchanged_error_count > max_num_slow && training_rate > min_training_rate){
          if(testing || has_slowed){
            training_rate = training_rate / (float)2.0; // 10.0 sumitted
            max_num_slow -=5;
            if(print_) {
              LOG("gpu_R_error has slow learning at iteration : "<<i)
              LOG("gpu_R_error reducing training_rate : "<<i);
              LOG("training_rate : "<<ToString(training_rate));
              LOG("max_num_slow : "<<max_num_slow);
            }            
          }else{
            if(print_) {
              LOG("gpu_R_error has slow learning at iteration : "<<i);
            }
            has_slowed = true;
            max_num_slow = 50;
            radius_around_min = (float)0.01;
            max_its = i + training_its;
          }
          unchanged_error_count = 0;
        }else if(unchanged_error_count > 20 && training_rate <= min_training_rate){
          if(!testing && !has_slowed){
            has_slowed = true;
            max_its = i + training_its;
            if(print_) LOG("forced has_slowed = true");
            bad_run = true;
            unchanged_error_count = 0;
          }else{
            if(print_) LOG("gpu_R_error error unchanged for too long : "<<i);
            error = temp;
            error_vector[i] = error;
            do_ = false;
            not_done = false;
            break;            
          }
        }
      }else{
        unchanged_error_count = 0;
      }
      if (temp > (float)1.3 * min_error_so_far && 
        !( (U_ACU_update_it_count == 1 || V_update_it_count == 1) || !(U_ACU_update_it_count == 1 && V_update_it_count == 1) ) ){
        // have we gotten worse?
        // has_slowed = false;
        if(print_) {
          LOG("gpu_R_error jumped over minimum iteration : "<<i);
          LOG("min_error_so_far : "<<min_error_so_far);
          LOG("new error : "<<temp);
          LOG("new error / min_error_so_far : "<<temp / min_error_so_far);
          LOG("gpu_R_error reducing training_rate : "<<i);
          LOG("training_rate : "<<ToString(training_rate));
        }
        //we jumped over the minimum
        checkCudaErrors(cudaMemcpy(U_t, U_t_check, batch_size_t * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyHostToDevice));
        if(!testing) {
          checkCudaErrors(cudaMemcpy(U_ACU, U_ACU_check, batch_size_ACU * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(V, V_check, ratings_cols * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyHostToDevice));
          //gpu_rng_gaussian<float>(batch_size_ACU * (compress ? num_latent_factors : batch_size_ACU), (float)0.0, (float)0.00007, U_ACU, 1, dn_handle);
          //gpu_rng_gaussian<float>(ratings_cols * (compress ? num_latent_factors : batch_size_ACU), (float)0.0, (float)0.00007, V, 1, dn_handle);
          gpu_gemm<float>(dn_handle, true, false, 
                ratings_cols, batch_size_ACU, 
                compress ? num_latent_factors : batch_size_ACU,
                (float)1.0,
                V, U_ACU, 
                (float)0.0,
                R_ACU);
        }
        temp = min_error_so_far;
        if(training_rate > min_training_rate){
          training_rate = training_rate / (float)10.0;
        }else if(i > 50){
          if(print_) LOG("gpu_R_error done : "<<i);
          error = temp;
          error_vector[i] = error;
          do_ = false;
          not_done = false;
          break;          
        }
      }


    }else{
      if(print_) LOG("initial error : "<<temp);
    }

    
    if(do_){
      float last_error = error;
      error = temp;
      if(Debug) LOG("gpu_R_error error at iteration "<<i<<" : "<< error); 
      temp = min_error_so_far;
      min_error_so_far = std::min(error, min_error_so_far);
      if((U_ACU_update_it_count == 1 || V_update_it_count == 1) && !updated_min_error_so_far){
        min_error_so_far = error;
        first_updating_iteration = i;
        if(print_) {
          LOG("first update to U_ACU or V made iteration : "<<i);
          LOG("min_error_so_far : "<<min_error_so_far);
        }

        updated_min_error_so_far = true;
      }
      if(min_error_so_far != temp){
        // store a backup
        checkCudaErrors(cudaMemcpy(U_t_check, U_t, batch_size_t * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyDeviceToHost));
        if(!testing) {
          checkCudaErrors(cudaMemcpy(U_ACU_check, U_ACU, batch_size_ACU * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyDeviceToHost));
          checkCudaErrors(cudaMemcpy(V_check, V, ratings_cols * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyDeviceToHost));
        }   
        min_error_iter = i;
        if(Debug){
          LOG("min_error_so_far : "<<min_error_so_far);
        }
      }
      error_vector[i] = error;

      if( (has_slowed_last != has_slowed) && !testing){
        has_slowed_last = has_slowed;
        float multiplier_ = (float)2.0 ; /// (float)10.0 ;
        if(has_slowed){
          training_rate = training_rate * multiplier_;
        }else{
          training_rate = training_rate / multiplier_;
        }
        //training_rate_V = (float)0.00001; // **submitted**
        training_rate_V = training_rate / (float)100.0; 
        //training_rate_U_ACU = (float)1.0 / (float)45000.0; // **submitted**
        training_rate_U_ACU = training_rate / (float)10.0; 
        if(1) {
          LOG("iteration : "<<i);
          LOG("training_rate : "<< ToString(training_rate));
          LOG("training_rate_U_ACU : "<< ToString(training_rate_U_ACU));
          LOG("training_rate_V : "<< ToString(training_rate_V));
        }
      }
      
      if( Debug && 0){
        LOG("gpu_R_error average error : "<< error); 
        //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

        float *training_entries_temp; 
        checkCudaErrors(cudaMalloc((void**)&training_entries_temp, nnz * SIZE_OF(float)));
        checkCudaErrors(cudaMemcpy(training_entries_temp, training_entries, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));

        temp = gpu_sum_of_squares_of_diff(dn_handle, nnz, 
                coo_format_ratingsMtx_rating_dev_batch, 
                training_entries_temp);
        //LOG("gpu_R_error error normalized by should be max error: "<< training_error_temp / (float)(nnz_)<<std::endl); 
        //LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 


        std::string title = ("ratings_errors_v" + ToString<int>(i)).c_str();
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, 
         coo_R, coo_errors, nnz, "ratings_errors_v");

        checkCudaErrors(cudaFree(training_entries_temp));
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
      bool reg_ = true;
      if(reg_){
        beta = (float)1.0 - training_rate * regularization_constant;
      }else{
        beta = (float)1.0;
      }
      beta = (float)1.0;

      bool updated_ = false;
      //(int)(std::min(training_iteration+ 2, 100) 
      // LOG("(int)(std::min(training_iteration + 2, 100) :  "<< (int)(std::min(training_iteration + 2, 100)));
      // LOG("i % (int)(std::min(training_iteration + 2, 100) :  "<< i % (int)(std::min(training_iteration + 2, 100)));
      bool update_u = true;
      if(has_slowed){
        has_slowed_count++;
        update_u = update_u && ( ( i % 2 ) != 0);
        if(Debug) LOG("error : "<< error);
      }
      if(testing || update_u ){
        // Update U

        // store a backup
        //checkCudaErrors(cudaMemcpy(U_t_check, U_t, batch_size_t * (compress ? num_latent_factors : batch_size_ACU) * SIZE_OF(float), cudaMemcpyDeviceToDevice));
        if(Debug) LOG("Here! it "<< i);
        gpu_spXdense_MMM<float>(sp_handle, false, false, 
                               batch_size_t, 
                               compress ? num_latent_factors : batch_size_ACU, 
                               ratings_cols,
                               nnz, first_coo_ind, &training_rate, sp_descr, 
                               coo_errors, 
                               csr_format_ratingsMtx_userID_dev_batch, 
                               coo_format_ratingsMtx_itemID_dev_batch,
                               V, ratings_cols, &beta, U_t, batch_size_t, 0);

        //============================================================================================
        // Handle column normalization in U_t
        //============================================================================================     
        if(S_with_U){
          if(SV != NULL){
            int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_ACU, SV, (float)0.0001);
            if(Debug){
              LOG("first_tiny_sv_ : "<<first_tiny_sv_)
            }
            if(first_tiny_sv_ < (int)((compress == false) ? batch_size_ACU : num_latent_factors)){
              LOG("WARNING WILL DIVIDE BY ~ZERO");
              long long int temp = (compress == false) ? batch_size_ACU : num_latent_factors;
              LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
            }
            gpu_div_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true );//right div

            //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
            //                              bool row_major_ordering, float* x, bool normalize_rows);
            gpu_normalize_mtx_rows_or_cols(batch_size_t, compress ? num_latent_factors : batch_size_ACU,  
                                              false, U_t, false);

            gpu_mult_US_in_SVD<float>(batch_size_t, compress ? num_latent_factors : batch_size_ACU, U_t, SV, true ); //right mult
          }else{
            //more thinking here:force order
            float current_largest_sv = gpu_norm(dn_handle, batch_size_t, U_t);
            gpu_scale(dn_handle, batch_size_t * ( compress ? num_latent_factors : batch_size_ACU), largest_sv/current_largest_sv, U_t);
          }
        }else{
          //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
          //                              bool row_major_ordering, float* x, bool normalize_rows);
          
          gpu_normalize_mtx_rows_or_cols(batch_size_t, compress ? num_latent_factors : batch_size_ACU, false, U_t, false); 
          if(Debug) LOG("here");     
        } 


        float change_in_error = (error - last_error); // new should be smaller so we want negative values
        if(updated_min_error_so_far){
          if(( (i - 1) % 4 ) != 0){
            if(U_ACU_update_it_count > 0){
              cpu_incremental_average(U_ACU_update_it_count, &avg_chg_during_U_ACU_update_it, change_in_error);
              if(Debug) LOG("avg_chg_during_U_ACU_update_it : "<< avg_chg_during_U_ACU_update_it);
            }
          }else{
            if(V_update_it_count > 0){
              cpu_incremental_average(V_update_it_count, &avg_chg_during_V_update_it, change_in_error);
              if(Debug) LOG("avg_chg_during_V_update_it : "<< avg_chg_during_V_update_it);
            }
          }
        }
        if(has_slowed) U_t_update_it_count++;
      }else{
        updated_ = true;

        float change_in_error = (error - last_error); // new should be smaller so we want negative values
        cpu_incremental_average(U_t_update_it_count, &avg_chg_during_U_t_update_it, change_in_error);
        if(Debug) {
          LOG("avg_chg_during_U_t_update_it : "<< avg_chg_during_U_t_update_it);
          LOG("avgerage sum of the updates : "<< avg_chg_during_U_t_update_it + avg_chg_during_V_update_it + avg_chg_during_U_ACU_update_it);
        }
    


        // LOG("(int)(std::min(training_iteration + 2, 100) :  "<< (int)(std::min(training_iteration + 2, 100)));
        // LOG("i % (int)(std::min(training_iteration + 2, 100) :  "<< i % (int)(std::min(training_iteration + 2, 100))); 
        // bool update_u_gu = ( ( i % update_u_inc ) == ( i % (update_u_inc -1) ) );  
        if(Debug) {
          LOG("it : "<< i);
          LOG("( i % 4 ) : "<< ( i % 4 ));
        }
        bool update_u_gu = ( i % 4 ) != 0;   
        if(update_u_gu){
          if(Debug) LOG("update U_ACU! it "<< i);


          gpu_dense_nearest_row<float>(batch_size_ACU, compress ? num_latent_factors : batch_size_ACU, U_ACU, 
                                       batch_size_t, U_t, 
                                       km_selection, km_errors, false);
          
          if(Debug) micro_km_error[U_ACU_update_it_count] = gpu_expected_value(batch_size_t,  km_errors);
          //training_rate = 0.01;
          gpu_calculate_KM_error_and_update(batch_size_ACU, compress ? num_latent_factors : batch_size_ACU, U_ACU, 
                                           batch_size_t, U_t, csr_format_ratingsMtx_userID_dev_batch,
                                           km_selection, training_rate_U_ACU, (float)1.0, training_iteration);
          //training_rate = temp_training_rate;

          if(0){
            save_device_array_to_file<float>(km_errors, batch_size_t, "micro_km_errors");
            save_device_array_to_file<int>(km_selection, batch_size_t, "micro_km_selection");

            gpu_sparse_nearest_row<float>(batch_size_ACU, ratings_cols, R_ACU, 
                                         batch_size_t, nnz, csr_format_ratingsMtx_userID_dev_batch, 
                                         coo_format_ratingsMtx_itemID_dev_batch,
                                         training_entries, km_selection, km_errors, false);
            micro_km_error[U_ACU_update_it_count] = gpu_sum(batch_size_t,  km_errors);
            
            save_host_array_to_file<float>(micro_km_error, U_ACU_update_it_count + 1, "micro_km_errors_through_iteration");
            save_host_array_to_file<float>(error_vector, i + 1, "training_error_thru_iterations", strPreamble(blank));

            save_device_mtx_to_file<float>(U_t, batch_size_t, (compress ? num_latent_factors : batch_size_ACU), "U_t", false, strPreamble(blank));
            save_device_mtx_to_file<float>(U_ACU, (compress ? num_latent_factors : batch_size_ACU), (compress ? num_latent_factors : batch_size_ACU), "U_ACU", false, strPreamble(blank));
          }
          U_ACU_update_it_count++;
        }else{

          // update V
          //============================================================================================
          // Update  V = V * (1 - alpha * lambda) + alpha * Error^T * U_training 
          //============================================================================================ 
          if(Debug) LOG("update V! it : "<< i);
          //training_rate = 0.1;

          gpu_mult_US_in_SVD<float>(ratings_cols, compress ? num_latent_factors : batch_size_ACU, V, Beta, true ); //right mult

          
          gpu_spXdense_MMM<float>(sp_handle, true, false, batch_size_t, compress ? num_latent_factors : batch_size_ACU, 
                                  ratings_cols, nnz, first_coo_ind, &training_rate_V, sp_descr, 
                                  coo_errors, 
                                  csr_format_ratingsMtx_userID_dev_batch, 
                                  coo_format_ratingsMtx_itemID_dev_batch,
                                  U_t, batch_size_t, &beta, V, ratings_cols, false);
          if(0) save_host_array_to_file<float>(error_vector, i + 1, "training_error_thru_iterations", strPreamble(blank));
          
          V_update_it_count++;
          //============================================================================================
          // Handle column normalization in V
          //============================================================================================     
          /*
            if(S_with_U){
              //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
              //                              bool row_major_ordering, float* x, bool normalize_rows);
              
              gpu_normalize_mtx_rows_or_cols(ratings_cols, compress ? num_latent_factors : batch_size_ACU, false, V, false); 
              if(Debug) LOG("here");  
            }else{
              if(SV != NULL){
                int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_ACU, SV, (float)0.00001);
                if(first_tiny_sv_ < (int)(compress ? num_latent_factors : batch_size_ACU)){
                  LOG("WARNING WILL DIVIDE BY ~ZERO");
                  long long int temp = (compress == false) ? batch_size_ACU : num_latent_factors;
                  LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
                  save_device_array_to_file(SV, compress ? num_latent_factors : batch_size_ACU, "SV", strPreamble(blank));
                }
                gpu_div_US_in_SVD<float>(ratings_cols, compress ? num_latent_factors : batch_size_ACU, V, SV, true );//right div

                //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
                //                              bool row_major_ordering, float* x, bool normalize_rows);
                gpu_normalize_mtx_rows_or_cols(ratings_cols, compress ? num_latent_factors : batch_size_ACU,  
                                                  false, V, false);

                gpu_mult_US_in_SVD<float>(ratings_cols, compress ? num_latent_factors : batch_size_ACU, V, SV, true ); //right mult
              }else{
                //more thinking here:force order
                float current_largest_sv = gpu_norm(dn_handle, ratings_cols, V);
                gpu_scale(dn_handle, ratings_cols * ( compress ? num_latent_factors : batch_size_ACU), largest_sv/current_largest_sv, V);
              }   
            } 
          */
        }
        //if(!testing && updated_){
        // M, N, K
        // M number of columns of matrix op(A) and C.
        // N is number of rows of matrix op(B) and C.]
        // K is number of columns of op(B) and rows of op(A).

        // op(B) is N by K
        // op(A) is K by M
        // C is N by M
        // cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc) 
        // performs C=alpha op ( B ) op ( A ) + beta C
        gpu_gemm<float>(dn_handle, true, false, 
                        ratings_cols, batch_size_ACU, 
                        compress ? num_latent_factors : batch_size_ACU,
                        (float)1.0,
                        V, U_ACU, 
                        (float)0.0,
                        R_ACU);

        //print_gpu_array_entries(U_ACU, 5, strPreamble(blank));

        long long int num_latent_factors_temp = num_latent_factors;
        gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                    batch_size_ACU, ratings_cols, 
                                    &num_latent_factors_temp, (float)1.0,
                                    R_ACU, 
                                    U_ACU, V, S_with_U, SV);

        //print_gpu_array_entries(U_ACU, 5, strPreamble(blank));

        float temp_lsv = largest_sv;
        checkCudaErrors(cudaMemcpy(&largest_sv, SV, SIZE_OF(float), cudaMemcpyDeviceToHost));
        if(Debug) LOG("largest_sv : "<<largest_sv);
        largest_sv = temp_lsv;
        //}
      }


      i += 1 ; 
    } // end do_

  } // end while
  
  if(!testing && print_ ){
    LOG("avg_chg_during_U_ACU_update_it : "<< avg_chg_during_U_ACU_update_it);
    LOG("avg_chg_during_V_update_it : "<< avg_chg_during_V_update_it);
    LOG("avg_chg_during_U_t_update_it : "<< avg_chg_during_U_t_update_it);
    LOG("avgerage sum of the updates : "<< avg_chg_during_U_t_update_it + avg_chg_during_V_update_it + avg_chg_during_U_ACU_update_it);
  }
  if(bad_run){
    save_host_array_to_file<float>(error_vector, i + 1, "training_error_thru_iterations_bad_run", strPreamble(blank) );
  }
  save_host_array_to_file<float>(error_vector, i + 1, "training_error_thru_iterations", strPreamble(blank));
  if(!testing){
    checkCudaErrors(cudaFree(Beta));

    //save_host_array_to_file<float>(error_vector + first_updating_iteration, i + 1 - first_updating_iteration, "training_error_thru_iterations", strPreamble(blank));
    //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_batch, training_entries, coo_errors, nnz, "ratings_testing_errors_v3");
    if(coo_R != NULL && Debug) save_device_arrays_side_by_side_to_file<float>(training_entries, coo_R, coo_errors, nnz, "training_actual_prediction_errors");
    
  }

  if(print_){
    LOG("number of training entries : "<<(float)nnz * ((float)1.0 - testing_fraction));
    LOG("gpu_R_error total iterations : "<<i + 1);
    LOG("gpu_R_error MSQER on training entries: "<<error);    
  }
  
  
  
  if(testing){

    // gpu_sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
    //  const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
    //  const Dtype* coo_entries_B, int* selection, Dtype* error, bool row_major_ordering)
    
    sparse_error(batch_size_t, ratings_cols, R_t, 
                 csr_format_ratingsMtx_userID_dev_batch, 
                 coo_format_ratingsMtx_itemID_dev_batch,
                 coo_format_ratingsMtx_rating_dev_batch, 
                 coo_errors, nnz, coo_R);

    gpu_reverse_bools<float>(nnz,  training_entries_cpy);         // zeros for training entries, ones for testing entries
    gpu_hadamard<float>(nnz, training_entries_cpy, coo_errors );  // only the testing errors 
    gpu_hadamard<float>(nnz, coo_format_ratingsMtx_rating_dev_batch, training_entries_cpy ); // only the testing entries

    if(1 && coo_R != NULL) {
      save_device_arrays_side_by_side_to_file<float>(training_entries_cpy, coo_R, coo_errors, nnz, "testing_actual_prediction_errors");
    }
    
    float error_test;// = gpu_sum_of_squares<float>(nnz, coo_errors);
    float mean_guess_error;// = gpu_sum_of_squares<float>(nnz, training_entries_cpy);
    gpu_mean_abs_nonzero(nnz, coo_errors, &error_test);
    gpu_mean_abs_nonzero(nnz, training_entries_cpy, &mean_guess_error);
    float nnz_ = (float)nnz * testing_fraction; 

    float* log_hist = (float *)malloc(7 * SIZE_OF(float)); 
    checkErrors(log_hist);
    cpu_set_all<float>(log_hist, 7, (float)0.0);
    gpu_logarithmic_histogram_abs_val(nnz, coo_errors, log_hist, (int)(0 - 3), 3, (int) nnz_);
    if(logarithmic_histogram != NULL){
      cpu_incremental_average_array<float>((long long int)(increment_index + 1), logarithmic_histogram, log_hist, 7);
    }

    if(print_){
      if(1) {
        //LOG("testing_error_on_testing_entries["<<0<<"] : "<<testing_error_on_testing_entries[0]);
        LOG("number of testing entries : "<<nnz_);
        //LOG("increment_index : "<<increment_index);
      }
      //print_host_array(log_hist, 7, "Logarithmic Histogram of Errors From 10^(-3) to 10^3", strPreamble(blank));

      LOG("gpu_R_error error on testing entries: "<<error_test /*/ nnz_*/); 
      //float expected_dist_two_guassian = gpu_expected_dist_two_guassian<float>(dn_handle, nnz_);
      //LOG("expected distance between two "<<nnz_<<" dimenstional guassian vectors : "<<expected_dist_two_guassian);
      //LOG("Testing error norm over E[|N(0,1) - N(0,1)|]: "<< std::sqrt(error_test) / expected_dist_two_guassian ); 
      LOG("Error of guessing the mean(zeros): "<<mean_guess_error); 
      LOG("Testing error over error of guessing the mean(zeros): "<< error_test / mean_guess_error ); 
      //LOG("Testing error norm over norm of testing only entries: "<< std::sqrt(error_test) / std::sqrt(mean_guess_error) );    
    }
    free(log_hist);
    cpu_incremental_average((long long int)(increment_index + 1), testing_error_on_testing_entries, error_test);

    //checkCudaErrors(cudaMemcpy(training_entries, training_entries_cpy, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(training_entries_cpy));
  }else{
    float km_errors_training;
    gpu_mean_abs_nonzero(batch_size_t, km_errors, &km_errors_training);
    if(print_) LOG("km_errors_training : "<<km_errors_training); 
    cpu_isBad(&km_errors_training, 1);
    cpu_incremental_average((long long int)(increment_index + 1), testing_error_on_testing_entries, km_errors_training);

    checkCudaErrors(cudaFree(km_errors));
    checkCudaErrors(cudaFree(km_selection));
    // checkCudaErrors(cudaFree(U_ACU_check));
    // checkCudaErrors(cudaFree(V_check));    
    free(U_ACU_check);
    free(V_check);
  } 
  cpu_incremental_average((long long int)(increment_index + 1), testing_error_on_training_entries, error); 
  cpu_incremental_average((long long int)(increment_index + 1), total_iterations, (float)(i + 1));
  //testing_error_on_training_entries[0] += (error - testing_error_on_training_entries[0]) / ((float) (increment_index + 1));
  //total_iterations[0] += (long long int)(((float)i - (float)total_iterations[0]) / ((float) (increment_index + 1)));

  if(coo_R != NULL) {
    checkCudaErrors(cudaFree(coo_R));
    //checkCudaErrors(cudaFree(U_t_check));
  }
  free(error_vector);
  free(micro_km_error);
  // checkCudaErrors(cudaFree(U_t_check));
  free(U_t_check);


  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   

  if(print_) LOG("gpu_R_error run time : "<<readable_time(program_time)<<std::endl);
}





template <>
void gpu_R_error_training<float>(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
  const long long int batch_size_training, const long long int batch_size_CU, 
  const long long int num_latent_factors, const long long int ratings_cols,
  const int nnz, const int first_coo_ind, const bool compress, float* coo_errors, 
  const float *coo_format_ratingsMtx_rating_dev_training, 
  const int *csr_format_ratingsMtx_userID_dev_training_batch, 
  const int *coo_format_ratingsMtx_itemID_dev_training,
  const float *V, float *U_training, float *R_training, 
  float training_rate, float regularization_constant, bool S_with_U, float *SV)
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
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, SIZE_OF(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, SIZE_OF(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, SIZE_OF(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, SIZE_OF(float), cudaMemcpyHostToDevice));
  // if(alpha_dev) checkCudaErrors(cudaFree(alpha_dev));
  // if(beta_dev) checkCudaErrors(cudaFree(beta_dev));

  float *U_training_check;
  if(compress){
    CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * num_latent_factors * SIZE_OF(float) ) );
  }else{
    CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * batch_size_CU  * SIZE_OF(float) ) );  
  }

  if(Debug){


    if(compress){
      LOG("num_latent_factors : "<<num_latent_factors);
      //CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * num_latent_factors * SIZE_OF(float) ) );
      LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
        <<"m = "<<batch_size_training<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
        <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
        <<"&alpha, sp_descr,"<<std::endl<<"                                                "
        <<"coo_format_ratingsMtx_rating_dev_training,csr_format_ratingsMtx_userID_dev_training_batch, coo_format_ratingsMtx_itemID_dev_training,"<<std::endl<<"                                                "
        <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
        <<"U_training, ldc = "<<batch_size_training<<" );"  );
    }else{
      //CUDA_CHECK( cudaMalloc( (void**)&U_training_check, batch_size_training * batch_size_CU  * SIZE_OF(float) ) );
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

  // gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_training,
  //  (compress == false) ? batch_size_CU : num_latent_factors, 
  //  ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
  //  coo_format_ratingsMtx_rating_dev_training, 
  //  csr_format_ratingsMtx_userID_dev_training_batch, 
  //  coo_format_ratingsMtx_itemID_dev_training,
  //  V, ratings_cols, &beta, U_training, batch_size_training, Debug);
  gpu_rng_gaussian<float>(batch_size_training * ( compress ? num_latent_factors : batch_size_CU ), (float)0.0, (float)1.0, U_training);

  float largest_sv = gpu_norm(dn_handle, batch_size_training, U_training);

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

  if(Debug){
    LOG("largest_sv : "<< largest_sv);
    save_device_mtx_to_file<float>(U_training, batch_size_training, compress ? num_latent_factors : batch_size_CU, "U_training");
    save_device_mtx_to_file<float>(V, ratings_cols, compress ? num_latent_factors : batch_size_CU, "V");
    //save_device_arrays_side_by_side_to_file<float>(U_training, U_training_check, batch_size_training * (compress ? num_latent_factors : batch_size_CU), "U_training");
  }

  int i = 0;
  bool not_done = true;
  float error;

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;

  while(not_done && i < 10000){
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
    if(S_with_U){
      if(SV != NULL){
        int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_CU, SV, (float)0.0001);
        if(first_tiny_sv_ < (compress ? num_latent_factors : batch_size_CU)){
          LOG("WARNING WILL DIVIDE BY ~ZERO");
          long long int temp = (compress ? num_latent_factors : batch_size_CU);
          LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
        }
        gpu_div_US_in_SVD<float>(batch_size_training, compress ? num_latent_factors : batch_size_CU, U_training, SV, true /*right div*/);

        //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
        //                              bool row_major_ordering, float* x, bool normalize_rows);
        gpu_normalize_mtx_rows_or_cols(batch_size_training, compress ? num_latent_factors : batch_size_CU,  
                                          false, U_training, false);

        gpu_mult_US_in_SVD<float>(batch_size_training, compress ? num_latent_factors : batch_size_CU, U_training, SV, true /*right mult*/);
      }else{
        //more thinking here:force order
        float current_largest_sv = gpu_norm(dn_handle, batch_size_training, U_training);
        gpu_scale(dn_handle, batch_size_training * ( compress ? num_latent_factors : batch_size_CU), largest_sv/current_largest_sv, U_training);
      }
    }else{
      if(Debug){
        LOG("calling gpu_normalize_mtx_rows_or_cols");
      }
      //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
      //                              bool row_major_ordering, float* x, bool normalize_rows);
      gpu_normalize_mtx_rows_or_cols(batch_size_training, compress ? num_latent_factors : batch_size_CU,  
                                        false, U_training, false);      
    }  

    gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_training, 
                    compress ? num_latent_factors : batch_size_CU,
                    (float)1.0, V, U_training, (float)0.0, R_training);
    bool isBad = gpu_isBad<float>(R_training, batch_size_training * ratings_cols);
    if(isBad){
      ABORT_IF_NEQ(0, 1, "isBad");
    };

    float *coo_R;
    if(Debug){
      checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * SIZE_OF(float)));
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

    bool do_ = true;
    float temp = training_error_temp / nnz;
    float epsilon = 0.00001;
    if(i > 1){
      //if(i % 1000 ==0) LOG("gpu_R_error iteration : "<<i);
      //have we stopped improving?
      if(error - temp < epsilon){
        do_ = false;
        if (abs(error - temp) > epsilon){
          LOG("gpu_R_error jumped over minimum iteration : "<<i);
          //we jumped over the minimum
          checkCudaErrors(cudaMemcpy(U_training, U_training_check, batch_size_training * (compress ? num_latent_factors : batch_size_CU) * SIZE_OF(float), cudaMemcpyDeviceToDevice));
          training_rate = training_rate / (float)10.0;
        }else{
          //we've learned enough
          not_done = false;
        }
      }
    }
    if(do_){
      error = temp;


      if( Debug){
        LOG("gpu_R_error average error : "<< error); 
        //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

        checkCudaErrors(cudaFree(coo_R));
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


      checkCudaErrors(cudaMemcpy(U_training_check, U_training, batch_size_training * (compress ? num_latent_factors : batch_size_CU) * SIZE_OF(float), cudaMemcpyDeviceToDevice));

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

      
    } // end do_
    i += 1 ;
  } // end while

  if(1) {
    //LOG("gpu_R_error call finished");
    LOG("gpu_R_error_training total iterations : "<<i);
    LOG("gpu_R_error_training error : "<<error);     
  }
  if(Debug){
    save_device_mtx_to_file<float>(U_training, batch_size_training, (compress ? num_latent_factors : batch_size_CU), "U_training");
    save_device_mtx_to_file<float>(V, ratings_cols, (compress ? num_latent_factors : batch_size_CU), "V");
    //save_device_arrays_side_by_side_to_file<float>(U_training, U_training_check, batch_size_training * (compress ? num_latent_factors : batch_size_CU), "U_training");
  }
  checkCudaErrors(cudaFree(U_training_check));
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("gpu_R_error_training run time : "<<readable_time(program_time)<<std::endl);
}

template <>
void gpu_R_error_testing<float>(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
  const long long int batch_size_testing, const long long int batch_size_CU, 
  const long long int num_latent_factors, const long long int ratings_cols,
  const int nnz, const int first_coo_ind, const bool compress, 
  float* testing_entries, float* coo_testing_errors, const float testing_fraction,
  const float *coo_format_ratingsMtx_rating_dev_testing, 
  const int *csr_format_ratingsMtx_userID_dev_testing_batch, 
  const int *coo_format_ratingsMtx_itemID_dev_testing,
  const float *V, float *U_testing, float *R_testing, 
  float training_rate, float regularization_constant, int increment_index,
  float* testing_error_on_training_entries, float* testing_error_on_testing_entries, 
  long long int* total_iterations, bool S_with_U, float *SV)
{
  bool Debug = false;
  LOG("gpu_R_error_testing called");
  std::string blank = "";

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
  checkCudaErrors(cudaMalloc((void**)&testing_entries_cpy, nnz * SIZE_OF(float)));
  checkCudaErrors(cudaMemcpy(testing_entries_cpy, testing_entries, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));


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
  // CUDA_CHECK(cudaMalloc((void**)&alpha_dev, SIZE_OF(float)));
  // CUDA_CHECK(cudaMalloc((void**)&beta_dev, SIZE_OF(float)));
  // checkCudaErrors(cudaMemcpy(alpha_dev, &alpha, SIZE_OF(float), cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(beta_dev, &beta, SIZE_OF(float), cudaMemcpyHostToDevice));
  // if(alpha_dev) checkCudaErrors(cudaFree(alpha_dev));
  // if(beta_dev) checkCudaErrors(cudaFree(beta_dev));

  float *U_testing_check;
  CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * (compress ? num_latent_factors : batch_size_CU) * SIZE_OF(float) ) );

  if(Debug){


    if(compress){
      LOG("num_latent_factors : "<<num_latent_factors);
      //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * num_latent_factors * SIZE_OF(float) ) );
      LOG("gpu_spXdense_MMM<float>(handle, TransA = "<<false<<" , TransB = "<< false<<","<<std::endl<<"                                                "
        <<"m = "<<batch_size_testing<<" , n = "<<num_latent_factors<<" , k = "<<ratings_cols<<","<<std::endl<<"                                                "
        <<"nnz = "<<  nnz<<" , first_coo_ind = "<<  first_coo_ind<<","<<std::endl<<"                                                "
        <<"&alpha, sp_descr,"<<std::endl<<"                                                "
        <<"testing_entries,csr_format_ratingsMtx_userID_dev_testing_batch, coo_format_ratingsMtx_itemID_dev_testing,"<<std::endl<<"                                                "
        <<"V, ldb = "<<ratings_cols<<", &beta,"<<std::endl<<"                                                "
        <<"U_testing, ldc = "<<batch_size_testing<<" );"  );
    }else{
      //CUDA_CHECK( cudaMalloc( (void**)&U_testing_check, batch_size_testing * batch_size_CU  * SIZE_OF(float) ) );
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



  // gpu_spXdense_MMM<float>(sp_handle, false, false, batch_size_testing,
  //  (compress == false) ? batch_size_CU : num_latent_factors, 
  //  ratings_cols, nnz, first_coo_ind, &alpha, sp_descr, 
  //  testing_entries, 
  //  csr_format_ratingsMtx_userID_dev_testing_batch, 
  //  coo_format_ratingsMtx_itemID_dev_testing,
  //  V, ratings_cols, &beta, U_testing, batch_size_testing, Debug);

  gpu_rng_gaussian<float>(batch_size_testing * (compress ? num_latent_factors : batch_size_CU), (float)0.0, (float)1.0, U_testing);

  float largest_sv;
  if(S_with_U){
    largest_sv = gpu_norm(dn_handle, batch_size_testing, U_testing);
  }

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

  if(Debug){
    save_device_mtx_to_file<float>(U_testing, batch_size_testing, (compress ? num_latent_factors : batch_size_CU), "U_testing", false, strPreamble(blank));
    save_device_mtx_to_file<float>(V, ratings_cols, (compress ? num_latent_factors : batch_size_CU), "V", false, strPreamble(blank));
    //save_device_arrays_side_by_side_to_file<float>(U_testing, U_testing_check, batch_size_testing * (compress ? num_latent_factors : batch_size_CU), "U_testing");
    //save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_testing, testing_entries, nnz, "R_testing");

    if(S_with_U){ LOG("largest_sv : "<<largest_sv) }
    LOG("compress : "<<compress)
    save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, coo_testing_errors, nnz, "ratings_testing_errors_v2");
  }

  int i = 0;
  bool not_done = true;
  float error;

  //float training_rate = 0.1;
  //float regularization_constant = 0.01;

  while(not_done && i < 10000){
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
    if(Debug){
      LOG("S_with_U : "<<S_with_U)
      LOG("SV != NULL : "<<(SV != NULL))
    }
    if(S_with_U){
      if(SV != NULL){
        int first_tiny_sv_ = first_tiny_sv(compress ? num_latent_factors : batch_size_CU, SV, (float)0.0001);
        if(Debug){
          LOG("first_tiny_sv_ : "<<first_tiny_sv_)
        }
        if(first_tiny_sv_ < (compress == false) ? batch_size_CU : num_latent_factors){
          LOG("WARNING WILL DIVIDE BY ~ZERO");
          long long int temp = (compress == false) ? batch_size_CU : num_latent_factors;
          LOG("first_tiny_sv_ : " << first_tiny_sv_<< " < "<<temp);
        }
        gpu_div_US_in_SVD<float>(batch_size_testing, compress ? num_latent_factors : batch_size_CU, U_testing, SV, true /*right div*/);

        //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
        //                              bool row_major_ordering, float* x, bool normalize_rows);
        gpu_normalize_mtx_rows_or_cols(batch_size_testing, compress ? num_latent_factors : batch_size_CU,  
                                          false, U_testing, false);

        gpu_mult_US_in_SVD<float>(batch_size_testing, compress ? num_latent_factors : batch_size_CU, U_testing, SV, true /*right mult*/);
      }else{
        //more thinking here:force order
        float current_largest_sv = gpu_norm(dn_handle, batch_size_testing, U_testing);
        gpu_scale(dn_handle, batch_size_testing * ( compress ? num_latent_factors : batch_size_CU), largest_sv/current_largest_sv, U_testing);
      }
    }else{
      //void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
      //                              bool row_major_ordering, float* x, bool normalize_rows);
      if(Debug) LOG("here");
      gpu_normalize_mtx_rows_or_cols(batch_size_testing, compress ? num_latent_factors : batch_size_CU, false, U_testing, false);      
    } 

    gpu_gemm<float>(dn_handle, true, false, ratings_cols, batch_size_testing, 
                    compress ? num_latent_factors : batch_size_CU,
                    (float)1.0, V, U_testing, (float)0.0, R_testing);

    bool isBad = gpu_isBad<float>(R_testing, batch_size_testing * ratings_cols);
    if(isBad){
      ABORT_IF_NEQ(0, 1, "isBad");
    };


    if(Debug && 0){
      //checkCudaErrors(cudaMalloc((void**)&coo_R, nnz * SIZE_OF(float)));
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

    bool do_ = true;
    float temp = training_error_temp / (float)(nnz_);
    float epsilon = 0.00001;
    if(i > 1){
      //if(i % 1000 ==0) LOG("gpu_R_error iteration : "<<i);
      //have we stopped improving?
      if(error - temp < epsilon){
        do_ = false;
        if (abs(error - temp) > epsilon){
          LOG("gpu_R_error jumped over minimum iteration : "<<i);
          //we jumped over the minimum
          checkCudaErrors(cudaMemcpy(U_testing, U_testing_check, batch_size_testing * (compress ? num_latent_factors : batch_size_CU) * SIZE_OF(float), cudaMemcpyDeviceToDevice));
          training_rate = training_rate / (float)10.0;
        }else{
          //we've learned enough
          not_done = false;
        }
      }
    }
    
    if(do_){
      error = temp;
      if( Debug && 0){
        LOG("gpu_R_error average error : "<< error); 
        //LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));

        float *testing_entries_temp; 
        checkCudaErrors(cudaMalloc((void**)&testing_entries_temp, nnz * SIZE_OF(float)));
        checkCudaErrors(cudaMemcpy(testing_entries_temp, testing_entries, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));

        temp = gpu_sum_of_squares_of_diff(dn_handle, nnz, 
          coo_format_ratingsMtx_rating_dev_testing, 
          testing_entries_temp);
        LOG("gpu_R_error error normalized by should be max error: "<< training_error_temp / (float)(nnz_)<<std::endl); 
        //LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 


        std::string title = ("ratings_testing_errors_v" + ToString<int>(i)).c_str();
        save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_testing, testing_entries, 
         coo_R, coo_testing_errors, nnz, "ratings_testing_errors_v");

        checkCudaErrors(cudaFree(testing_entries_temp));
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
      // store a backup
      checkCudaErrors(cudaMemcpy(U_testing_check, U_testing, batch_size_testing * (compress ? num_latent_factors : batch_size_CU) * SIZE_OF(float), cudaMemcpyDeviceToDevice));


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

      
    } // end do_
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

  save_device_arrays_side_by_side_to_file<float>(coo_testing_errors, testing_entries_cpy, nnz, "testing_entry_errors");

  float error_test = gpu_sum_of_squares<float>(nnz, coo_testing_errors);
  float mean_guess_error = gpu_sum_of_squares<float>(nnz, testing_entries_cpy);
  float nnz_ = (float)nnz * testing_fraction; 

  if(1) {
    //LOG("gpu_R_error call finished");
    LOG("gpu_R_error_testing total iterations : "<<i);
    LOG("gpu_R_error_testing MSQER on training entries: "<<error); 

    LOG("gpu_R_error_testing MSQER on testing entries: "<<error_test / nnz_); 

    //float expected_dist_two_guassian = gpu_expected_dist_two_guassian<float>(dn_handle, nnz_);
    //LOG("expected distance between two "<<nnz_<<" dimenstional guassian vectors : "<<expected_dist_two_guassian);
    //LOG("Testing error norm over E[|N(0,1) - N(0,1)|]: "<< std::sqrt(error_test) / expected_dist_two_guassian ); 

    LOG("Testing error norm over norm of testing only entries: "<< std::sqrt(error_test) / std::sqrt(mean_guess_error) );    
  }

  //checkCudaErrors(cudaMemcpy(testing_entries, testing_entries_cpy, nnz * SIZE_OF(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(testing_entries_cpy));
  if(Debug) {
    checkCudaErrors(cudaFree(coo_R));
    //checkCudaErrors(cudaFree(U_testing_check));
  }
  checkCudaErrors(cudaFree(U_testing_check));
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(0) {
    LOG("testing_error_on_testing_entries["<<0<<"] : "<<testing_error_on_testing_entries[0]);
    LOG("nnz_ : "<<nnz_);
    LOG("increment_index : "<<increment_index);
  }
  testing_error_on_training_entries[0] += (error - testing_error_on_training_entries[0]) / ((float) (increment_index + 1));
  testing_error_on_testing_entries[0] += (error_test / nnz_ - testing_error_on_testing_entries[0]) / ((float) (increment_index + 1));
  total_iterations[0] += (long long int)(((float)i - (float)total_iterations[0]) / ((float) (increment_index + 1)));
  LOG("gpu_R_error_testing run time : "<<readable_time(program_time)<<std::endl);
}






template <typename Dtype>
__global__ void gpu_dense_nearest_row_kernel(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype* error, bool row_major_ordering, bool* isBad)
{
  CUDA_KERNEL_LOOP(row_B, rows_B) {
    // For this row_B, which row_A is closest?
    Dtype closest_A_row_dist = (Dtype)10000.0;
    int   closest_A_row      = 0;
    for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
      Dtype temp = (Dtype)0.0;
      for(long long int col = (long long int)0; col < (long long int)cols; col+=(long long int)1){
        if(row_major_ordering){
          gpu_incremental_average<Dtype>(col + (long long int)(1), &temp, 
            std::abs(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - dense_mtx_B[row_B + (long long int)cols * (long long int)col]));
          //gpu_incremental_average<Dtype>((long long int)(count), &temp, pow(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - dense_mtx_B[row_B + (long long int)cols * (long long int)col], (Dtype)2.0));
          //temp += pow(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - dense_mtx_B[row_B + (long long int)cols * (long long int)col], (Dtype)2.0);
        }else{
          gpu_incremental_average<Dtype>(col + (long long int)(1), &temp, 
            std::abs(dense_mtx_A[row_A + (long long int)rows_A * (long long int)col] - dense_mtx_B[row_B + (long long int)rows_B * (long long int)col]));
          //temp += pow(dense_mtx_A[row_A + (long long int)rows_A * (long long int)col] - dense_mtx_B[row_B + (long long int)rows_B * (long long int)col], (Dtype)2.0);
        }
      }
      if(temp < closest_A_row_dist || row_A == (long long int)0){
        closest_A_row_dist = temp;
        closest_A_row      = (int)row_A;
      }
    }
    selection[row_B] = closest_A_row;
    error[row_B] = closest_A_row_dist;

    if (::isinf(error[row_B]) || ::isnan(error[row_B])){
      isBad[0] = true;
    };
  }
}


template <typename Dtype>
void gpu_dense_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype* error, bool row_major_ordering)
{
  bool Debug = false;
  if(Debug) LOG("gpu_dense_nearest_row called");
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS((long long int)rows_B);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      if(Debug){
        LOG("num_gpu_blocks : "<<num_gpu_blocks);
        LOG("num_loops : "<<num_loops);
        LOG("spot : "<<spot);
      }
      gpu_dense_nearest_row_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                          num_entries, dense_mtx_B, selection, error, row_major_ordering, isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_dense_nearest_row_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                  rows_B - spot, dense_mtx_B, selection, error, row_major_ordering, isBad);
  }else{
    if(too_big(rows_B) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    if(Debug){
      LOG("num_gpu_blocks : "<<num_gpu_blocks);
      LOG("CUDA_NUM_THREADS : "<<CUDA_NUM_THREADS);
      LOG("rows_A : "<<rows_A);
      LOG("cols : "<<cols);
      LOG("rows_B : "<<rows_B);
      LOG("row_major_ordering : "<<row_major_ordering);
    }
    gpu_dense_nearest_row_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                    rows_B, dense_mtx_B, selection, error, row_major_ordering, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    save_device_arrays_side_by_side_to_file(selection, error, rows_B, "selection_errors");
            save_device_mtx_to_file<float>(dense_mtx_A, rows_A, cols, "dense_mtx_A", false, strPreamble(blank));
            save_device_mtx_to_file<float>(dense_mtx_B, rows_B, cols, "dense_mtx_B", false, strPreamble(blank));
    ABORT_IF_NEQ(0, 1, "isBad");
  };

  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(Debug) LOG("gpu_dense_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}

template void gpu_dense_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const float* dense_mtx_B, int* selection, float* error, bool row_major_ordering);





template <typename Dtype>
__global__ void gpu_sparse_nearest_row_kernel(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection, Dtype* error, bool row_major_ordering, bool* isBad)
{
  const int row_skip = csr_rows_B[0];
  CUDA_KERNEL_LOOP(row_B, rows_B) {
    // For this row_B, which row_A is closest?
    Dtype closest_A_row_dist = (Dtype)10000.0;
    int   closest_A_row      = 0;
    for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
      Dtype temp = (Dtype)0.0;
      int count = 0;
      for(long long int coo_index = (long long int)(csr_rows_B[row_B]); coo_index < (long long int)(csr_rows_B[row_B + 1]); coo_index+=(long long int)1){
        int true_coo_index = coo_index - row_skip;
        int col = coo_cols_B[true_coo_index];
        count++;
        if(row_major_ordering){
          gpu_incremental_average<Dtype>((long long int)(count), &temp, std::abs(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - coo_entries_B[true_coo_index]));
          //gpu_incremental_average<Dtype>((long long int)(count), &temp, pow(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - coo_entries_B[true_coo_index], (Dtype)2.0));
          //temp += pow(dense_mtx_A[row_A * (long long int)cols + (long long int)col] - coo_entries_B[true_coo_index], (Dtype)2.0);
        }else{
          gpu_incremental_average<Dtype>((long long int)(count), &temp, std::abs(dense_mtx_A[row_A + (long long int)rows_A * (long long int)col] - coo_entries_B[true_coo_index]));
          //temp += pow(dense_mtx_A[row_A + (long long int)rows_A * (long long int)col] - coo_entries_B[true_coo_index], (Dtype)2.0);
        }
      }
      temp *= ( ((Dtype)count) / ((Dtype)num_sparse_entries) );
      if(temp < closest_A_row_dist || row_A == (long long int)0){
        closest_A_row_dist = temp;
        closest_A_row      = (int)row_A;
      }
    }
    selection[row_B] = closest_A_row;
    error[row_B] = closest_A_row_dist;

    if (::isinf(error[row_B]) || ::isnan(error[row_B])){
      isBad[0] = true;
    };
  }
}


template <typename Dtype>
void gpu_sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection, Dtype* error, bool row_major_ordering)
{
  bool Debug = false;
  if(Debug) LOG("gpu_sparse_nearest_row called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int num_gpu_blocks = GET_BLOCKS((long long int)rows_B);

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
      ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  };

  
  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS){
      if(Debug){
        LOG("num_gpu_blocks : "<<num_gpu_blocks);
        LOG("num_loops : "<<num_loops);
        LOG("spot : "<<spot);
      }
      gpu_sparse_nearest_row_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                          num_entries, num_sparse_entries, csr_rows_B + spot, coo_cols_B, 
                                          coo_entries_B, selection, error, row_major_ordering, isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    gpu_sparse_nearest_row_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                  rows_B - spot, num_sparse_entries, csr_rows_B + spot, coo_cols_B, 
                                  coo_entries_B, selection, error, row_major_ordering, isBad);
  }else{
    if(too_big(rows_B) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    if(Debug){
      LOG("num_gpu_blocks : "<<num_gpu_blocks);
      LOG("CUDA_NUM_THREADS : "<<CUDA_NUM_THREADS);
      LOG("rows_A : "<<rows_A);
      LOG("cols : "<<cols);
      LOG("rows_B : "<<rows_B);
      LOG("num_sparse_entries : "<<num_sparse_entries);
      LOG("row_major_ordering : "<<row_major_ordering);
    }
    gpu_sparse_nearest_row_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A,  cols, dense_mtx_A, 
                                    rows_B, num_sparse_entries, csr_rows_B, coo_cols_B, 
                                    coo_entries_B, selection, error, row_major_ordering, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };

  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(Debug) LOG("gpu_sparse_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}

template void gpu_sparse_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const float* coo_entries_B, int* selection, float* error, bool row_major_ordering);








template <typename Dtype>
__global__ void gpu_calculate_KM_error_and_update_kernel(const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                                    const int rows_B, const Dtype* dense_mtx_B, const int* csr_rows_B, 
                                                    int* selection, Dtype training_rate, Dtype regularization_constant, int increment_index,
                                                    long long int start, const int num, bool* isBad)
{
  bool row_major_ordering = false;
  CUDA_KERNEL_LOOP(index, num) {
    long long int row_A = ((long long int)index + start) % ((long long int) rows_A);
    long long int col   = ((long long int)index + start) / ((long long int) rows_A);
    Dtype temp = (Dtype)0.0;
    long long int count = (long long int)0;
    for(long long int row_B = (long long int)0; row_B < (long long int)rows_B; row_B +=(long long int)1){
      if(selection[row_B] == (int)row_A){
        // if(row_major_ordering){
        //   temp += dense_mtx_B[row_B * (long long int)cols + col];
        // }else{
        //   temp += dense_mtx_B[row_B + (long long int)rows_B * col];
        // }
        count += (long long int)(csr_rows_B[(int)row_B + 1] - csr_rows_B[(int)row_B]);
      }
    }
    for(long long int row_B = (long long int)0; row_B < (long long int)rows_B; row_B +=(long long int)1){
      if(selection[row_B] == (int)row_A){
        float frac = ((float)(csr_rows_B[(int)row_B + 1] - csr_rows_B[(int)row_B])) / ((float)count);
        if(row_major_ordering){
          temp += frac * dense_mtx_B[row_B * (long long int)cols + col];
        }else{
          temp += frac * dense_mtx_B[row_B + (long long int)rows_B * col];
        }
        //count++;
      }
    }
    if(count > (long long int)0){
      Dtype old_val;
      long long int index_;
      if(row_major_ordering){
        index_ = row_A * (long long int)cols + col;
      }else{
        index_ = row_A + (long long int)rows_A * col;
      }
      old_val = dense_mtx_A[index_];
      dense_mtx_A[index_] = ((Dtype)1.0 - training_rate * regularization_constant) * old_val + training_rate * (temp /*/ ((Dtype)count)*/);

      //gpu_incremental_average<Dtype>(min((long long int)(increment_index + 1), (long long int)(100)), dense_mtx_A + index_, temp);


      if (::isinf(dense_mtx_A[index_]) || ::isnan(dense_mtx_A[index_])){
        isBad[0]= true;
      }
    }

  }
}

template<typename Dtype>
void gpu_calculate_KM_error_and_update(const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                      const int rows_B, const Dtype* dense_mtx_B, const int* csr_rows_B, 
                                      int* selection, Dtype training_rate, Dtype regularization_constant, int increment_index)
{
  bool Debug = false;
  if(Debug) LOG("gpu_calculate_KM_error_and_update called");
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  bool isBad_host = false;
  bool* isBad;
  CUDA_CHECK(cudaMalloc((void**)&isBad, SIZE_OF(bool)));
  CUDA_CHECK(cudaMemcpy(isBad, &isBad_host, SIZE_OF(bool), cudaMemcpyHostToDevice));

  long long int total = (long long int)rows_A * (long long int)cols;
  long long int num_gpu_blocks = GET_BLOCKS(total);

  // if (num_gpu_blocks > CUDA_NUM_BLOCKS){
  //     ABORT_IF_NEQ(0, 1, "Max Blocks Exceeded");
  // };

  if (num_gpu_blocks > CUDA_NUM_BLOCKS){
    long long int num_loops   = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS * CUDA_NUM_THREADS);
    long long int spot        = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    while (num_gpu_blocks > CUDA_NUM_BLOCKS)
    {
      gpu_calculate_KM_error_and_update_kernel<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS>>>(rows_A, cols, dense_mtx_A, 
                                                    rows_B, dense_mtx_B, csr_rows_B, selection, training_rate, regularization_constant, increment_index, spot, num_entries, isBad);
      num_gpu_blocks = num_gpu_blocks - (long long int)CUDA_NUM_BLOCKS;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
    };
    // spot is the number of entries done so far
    // total - (done) = left to go 
    int left_over = (int)(total - spot);
    gpu_calculate_KM_error_and_update_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A, cols, dense_mtx_A, 
                                                    rows_B, dense_mtx_B, csr_rows_B, selection, training_rate, regularization_constant, increment_index, spot, left_over, isBad);
  }else{
    if(too_big(total)) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    gpu_calculate_KM_error_and_update_kernel<<<num_gpu_blocks, CUDA_NUM_THREADS>>>(rows_A, cols, dense_mtx_A, 
                                                    rows_B, dense_mtx_B, csr_rows_B, selection, training_rate, regularization_constant, increment_index, (long long int)0, (int)total, isBad);
  };

  CUDA_CHECK(cudaMemcpy(&isBad_host, isBad, SIZE_OF(bool), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(isBad));
  if(isBad_host){
    ABORT_IF_NEQ(0, 1, "isBad");
  };
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(Debug) LOG("gpu_calculate_KM_error_and_update run time : "<<readable_time(program_time)<<std::endl);
}

template void gpu_calculate_KM_error_and_update<float>(const int rows_A, const int cols, float* dense_mtx_A, 
    const int rows_B, const float* dense_mtx_B, const int* csr_rows_B, int* selection, 
    float training_rate, float regularization_constant, int increment_index);


