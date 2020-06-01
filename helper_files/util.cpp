#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <iterator> 
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <bitset>
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

//using namespace Eigen;

typedef boost::minstd_rand base_generator_type;


bool too_big(const long long int li){
  bool Debug = false;
  if(Debug){
    LOG("INT_MIN : "<<INT_MIN);
    LOG("INT_MAX : "<<INT_MAX);
    LOG("compared to : "<<li);
  }
  if (li >= (long long int)INT_MIN && li <= (long long int)INT_MAX) {
    return false;
  } else {
    return true;
  }
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, SIZE_OF(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}


template<typename Dtype>
bool cpu_isBad(const Dtype* A, long long int size, std::string var_name, std::string file_line) 
{
  bool isBad = false;
  std::string blank = "";
  for(long long int i = (long long int)0; i < size; i += (long long int)1){
    if (::isinf(A[i]) || ::isnan(A[i])){
      isBad = true;
    };
  };
  if(isBad){
    save_host_array_to_file(A, (int)size, var_name + "_isBad_array", file_line );
    ABORT_IF_EQ(0, 0, "isBad");
  }
  return isBad;
}

template bool cpu_isBad<float>(const float* A, long long int size, std::string var_name, std::string file_line);

//============================================================================================
// manipulate memory
//============================================================================================


template <typename Dtype>
void host_copy(const long long int N, const Dtype* X, Dtype* Y) 
{
  if (X != Y) {
    memcpy(Y, X, SIZE_OF(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }

}

template void host_copy<int>(const long long int N, const int* X, int* Y);
template void host_copy<unsigned int>(const long long int N, const unsigned int* X, unsigned int* Y);
template void host_copy<float>(const long long int N, const float* X, float* Y);
template void host_copy<double>(const long long int N, const double* X, double* Y);

template <typename Dtype>
void copy_device_mtx_into_host_submtx(const long long int M, const long long int N, const Dtype* X, Dtype* Y, const long long int inc_Y)
{
  bool Debug = false;
  std::string blank = "";
  if(Debug){
    LOG("copy_device_mtx_into_host_submtx called");
    LOG("X is on the device "<<M<<" by "<<N<<" in memory = "<<M * N);
    LOG("Y is on host "<<inc_Y<<" by "<<N<<" in memory = "<<inc_Y * N);
    LOG("M : "<< M);
    LOG("N : "<< N);
    LOG("inc_Y : "<< inc_Y);
    print_gpu_mtx_entries<Dtype>(X, M, N, "X", 0, strPreamble(blank));
  }

  if(inc_Y == M){
    checkCudaErrors(cudaMemcpy(Y,  X,  M * N * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
    if(Debug){
      checkCudaErrors(cudaDeviceSynchronize());
      print_host_mtx<Dtype>(Y, inc_Y, N, "Y", 0, strPreamble(blank));
      checkCudaErrors(cudaDeviceSynchronize());
      print_gpu_mtx_entries<Dtype>(X, M, N, "X", 0, strPreamble(blank));
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }else{
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min(nProcessors, (int)N);
      if(Debug){
        LOG("copy_device_mtx_into_host_submtx done");
        LOG("nthreads : "<< nthreads);
      }
      omp_set_num_threads(nthreads);
      omp_lock_t printlock;
      omp_init_lock(&printlock);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int j = (long long int)th_id; j < N; j += (long long int)nthreads){
      //for(long long int j = (long long int)0; j < (long long int)N; j+=(long long int)1){
        long long int y_skip = j * inc_Y;
        long long int x_skip = j * M;
        if(Debug){
          omp_set_lock(&printlock);
          LOG("th_id : "<< th_id);
          LOG("j : "<< j);
          LOG("N : "<< N);
          LOG("y_skip : "<< y_skip);
          LOG("y_skip  + M: "<< y_skip + M <<" <= "<<inc_Y * N);
          LOG("x_skip : "<< x_skip);
          LOG("x_skip  + M: "<< x_skip + M <<" <= "<<M * N);
        }
        Dtype* Y_temp = Y + y_skip;
        const Dtype* X_temp = X + x_skip;
        checkCudaErrors(cudaMemcpy(Y_temp,  X_temp,  M * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));
        if(Debug){
          checkCudaErrors(cudaDeviceSynchronize());
          print_host_mtx<Dtype>(Y, inc_Y, N, "Y", 0, strPreamble(blank));
          omp_unset_lock(&printlock);
        }
      }
    }
  }
  if(Debug){
    LOG("copy_device_mtx_into_host_submtx done");
  }
}

template void copy_device_mtx_into_host_submtx<float>(const long long int M, const long long int N, const float* X, float* Y, const long long int inc_Y);

//============================================================================================
// math functions
//============================================================================================







template <>
void cpu_incremental_average_array<float>(const long long int increment_index, float* old_avgs, float* new_vals, int num) {
  int nthreads = 1;
  if(increment_index < (long long int)1){
    ABORT_IF_EQ(0, 0, "oops!");
  }
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, num);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    if(increment_index == (long long int)1){
      for(int j = th_id; j < num; j++){
        old_avgs[j] = new_vals[j];
      }
    }else{
      for(int j = th_id; j < num; j++){
        old_avgs[j] += (new_vals[j] - old_avgs[j]) / ((float)(increment_index));
      }
    }
  }
}

template<typename Dtype>
void cpu_incremental_average(const long long int increment_index, Dtype* old_avg, Dtype new_val) {
  bool Debug = false;
  if(increment_index < (long long int)1){
    ABORT_IF_EQ(0, 0, "oops!");
  }
  if(increment_index == (long long int)1){
    old_avg[0] = new_val;
  }else{
    if(Debug){
      LOG("old avg : "<< old_avg[0]);
      LOG("new value : "<< new_val);
      LOG("(new_val - old_avg[0]) : "<< (new_val - old_avg[0]));
      LOG("(Dtype)(increment_index) : "<< (Dtype)(increment_index));
      LOG("( (new_val - old_avg[0]) / ((Dtype)(increment_index)) ) : "<< ( (new_val - old_avg[0]) / ((Dtype)(increment_index)) ));
    }
    old_avg[0] += ( (new_val - old_avg[0]) / ((Dtype)(increment_index)) );
    if(Debug)LOG("new avg : "<< old_avg[0]);
  }
}

template void cpu_incremental_average<int>(const long long int increment_index, int* old_avg, int new_val);
template void cpu_incremental_average<float>(const long long int increment_index, float* old_avg, float new_val);
template void cpu_incremental_average<double>(const long long int increment_index, double* old_avg, double new_val);

template <>
void cpu_incremental_average<long long int>(const long long int increment_index, long long int* old_avg, long long int new_val) {
  if(increment_index < (long long int)1){
    ABORT_IF_EQ(0, 0, "oops!");
  }
  if(increment_index == (long long int)1){
    old_avg[0] = (long long int)new_val;
  }else{
    old_avg[0] = (long long int)(((float)new_val - (float)old_avg[0]) / ((float)(increment_index)));
  }
}

template<typename Dtype>
void cpu_mean_abs_nonzero(const long long int n, const Dtype* x, Dtype* y, bool Debug, std::string vec_name) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

  y[0] = (Dtype)0.0;
  long long int z = (long long int)0;
  long long int count_nonzero = (long long int)0;
  for(long long int i = (long long int)0; i < n; i+=(long long int)1) {
    if(x[i] != (Dtype)0.0){
      count_nonzero+=(long long int)1;
      cpu_incremental_average((long long int)(count_nonzero), y, std::abs(x[i]));
    }
  }
  z = n - count_nonzero;

  if(Debug){
    LOG(z <<" out of "<< n<<" entries are zero in "<<vec_name<<".");
    LOG( ((Dtype)z) / ((Dtype)n)  <<" of the entries are zero in "<<vec_name<<".");
  }
}
template void cpu_mean_abs_nonzero<int>(const long long int n, const int* x, int* y, bool Debug, std::string vec_name);
template void cpu_mean_abs_nonzero<float>(const long long int n, const float* x, float* y, bool Debug, std::string vec_name);
template void cpu_mean_abs_nonzero<double>(const long long int n, const double* x, double* y, bool Debug, std::string vec_name);

template <typename Dtype>
Dtype cpu_asum(const long long int n, const Dtype* x) {
  Dtype s = (Dtype)0.0;
  for (long long int k = (long long int)0; k < n; k+=(long long int)1){
    s += abs(x[k]);
    //LOG(INFO) << x[k];
  };
  return s;
}

template int cpu_asum<int>(const long long int n, const int* x);
template float cpu_asum<float>(const long long int n, const float* x);
template double cpu_asum<double>(const long long int n, const double* x);

template <typename Dtype>
void cum_asum(const long long int n, Dtype* x) {
  Dtype s = (Dtype)0.0;
  for (long long int k = (long long int)0; k < n; k+=(long long int)1){
    s += abs(x[k]);
    x[k] = s;
    //LOG(INFO) << x[k];
  };
}

template void cum_asum<int>(const long long int n, int* x);
template void cum_asum<float>(const long long int n, float* x);
template void cum_asum<double>(const long long int n, double* x);

// template <>
// float cpu_asum<float>(const long long int n, const float* x) {
//   if(too_big(n) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
//   return cblas_sasum((int)n, x, 1);
// }

// template <>
// double cpu_asum<double>(const long long int n, const double* x) {
//   if(too_big(n) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
//   return cblas_dasum((int)n, x, 1);
// }

template <>
float cpu_sum<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 0,"Long long long int too big");}
  float sum = thrust::reduce(thrust::host, x, x + n, (float)0., thrust::plus<float>());
  return sum;
}

// square<Dtype> computes the square of a number f(x) -> x*x 
template <typename Dtype> 
struct square_ 
{ 
  Dtype operator()(const Dtype& x) const {
   return x * x; 
 } 
};

template <>
float cpu_sum_of_squares<float>(const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  square_<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float s = thrust::transform_reduce(thrust::host, x, x+n, unary_op, init, binary_op) ; 
  return s/* /(float)n*/;
}

template <typename Dtype>
void cpu_scal(const long long int N, const Dtype alpha, Dtype *X) {
  bool Debug = false;
  if(Debug) {
    LOG("INT_MAX : "<<INT_MAX);
    LOG("N : "<<N);
    LOG("(int)N : "<<(int)N);
    LOG("alpha : "<<alpha);
  }

  if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  //cblas_sscal((int)N, alpha, X, 1);


  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, N);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int k = (long long int)th_id; k < N; k += (long long int)nthreads){
    //for (long long int k = (long long int)0; k < N; k+=(long long int)1){
      X[k] = X[k] * alpha;
    };
  }
}

template void cpu_scal<float>(const long long int N, const float alpha, float *X);
template void cpu_scal<double>(const long long int N, const double alpha, double *X);

// template <>
// void cpu_scal<double>(const long long int N, const double alpha, double *X) {
//   if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
//   cblas_dscal((int)N, alpha, X, 1);
// }


int gcd(int a, int b) 
{
  return b == 0 ? a : gcd(b, a % b);
}


template <typename Dtype>
void cpu_permute(Dtype* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) 
{
  bool Debug = false;
  // A is rows by cols stored in col major ordering
  // this function permutes in place
  if(Debug) LOG("cpu_permute called");
  if(permute_rows){
    if(Debug) LOG("permute_rows is true") ;
    //pvt is an array length rows
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min((long long int)nProcessors, cols);
      omp_set_num_threads(nthreads);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int j = (long long int)th_id; j < cols; j += (long long int)nthreads){
      //for(long long int  j = 0; j <cols; j+=(long long int)1) {
        long long int ind=(long long int)0;
        Dtype temp = (Dtype)0;

        for(long long int i = (long long int)0; i < rows - (long long int)1; i+=(long long int)1){
          // get next index
          ind = (long long int)P[i];
          while(ind < i){
            ind = (long long int)P[ind];
          }
          // swap elements in array
          temp = A[i + j * rows];
          A[i + j * rows] = A[ind + j * rows];
          A[ind + j * rows] = temp;
          if (::isinf(A[i + j * rows]) || ::isnan(A[i + j * rows])){
            LOG("A["<<i<<", "<<j<<"]"<<A[i + j * rows]);
          }
          if (::isinf(A[ind + j * rows]) || ::isnan(A[ind + j * rows])){
            LOG("A["<<ind<<", "<<j<<"]"<<A[ind + j * rows]);
          }
        };
      };
    }
  } else{
    if(Debug) LOG("permute_rows is false");
    //pvt is an array length cols
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min((long long int)nProcessors, rows);
      omp_set_num_threads(nthreads);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int i = (long long int)th_id; i < rows; i += (long long int)nthreads){
      //for(long long int i = (long long int)0; i<rows; i+=(long long int)1) {
        long long int ind = (long long int)0;
        Dtype temp = (Dtype)0.0;

        for(long long int j=(long long int)0; j < cols - (long long int)1; j+=(long long int)1){
          // get next index
          ind = (long long int)P[j];
          while(ind < j){
            ind = (long long int)P[ind];
          }
          // swap elements in array
          if(Debug) LOG("i + j * rows: "<<i<<" + "<<j<<" * "<<rows<<" = "<<i + j * rows) ;
          if(Debug) LOG("i + ind * rows: "<<i<<" + "<<ind<<" * "<<rows<<" = "<<i + ind * rows) ;
          temp = A[i + j * rows];
          A[i + j * rows] = A[i + ind * rows];
          A[i + ind * rows] = temp;
          if (::isinf(A[i + j * rows]) || ::isnan(A[i + j * rows])){
            LOG("A["<<i<<", "<<j<<"]"<<A[i + j * rows]);
          }
          if (::isinf(A[i + ind * rows]) || ::isnan(A[i + ind * rows])){
            LOG("A["<<i<<", "<<ind<<"]"<<A[i + ind * rows]);
          }
        };
      };
    }
  }
}
template void cpu_permute<float>(float* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) ;
template void cpu_permute<double>(double* A, const int* P, const long long int rows, const long long int cols, bool permute_rows);


// Non-square matrix transpose of matrix of size r x c and base address A 
template <>
void MatrixInplaceTranspose<float>(float *A, long long int r, long long int c, bool row_major_ordering) 
{ //ABORT_IF_NEQ(0, 1, "Function Not Supported Yet");
  /*
    int HASH_SIZE = 128;

    int size = r*c - 1; 
    float t; // holds element to be replaced, eventually becomes next element to move 
    int next; // location of 't' to be moved 
    int cycleBegin; // holds start of cycle 
    int i; // iterator 
    bitset<HASH_SIZE> b; // hash to mark moved elements 

    b.reset(); 
    b[0] = b[size] = 1; 
    i = 1; // Note that A[0] and A[size-1] won't move 
    while (i < size) 
    { 
        cycleBegin = i; 
        t = A[i]; 
        do
        { 
            // Input matrix [r x c] 
            // Output matrix  
            // i_new = (i*r)%(N-1) 
            next = (i*r)%size; 
            swap(A[next], t); 
            b[i] = 1; 
            i = next; 
        } 
        while (i != cycleBegin); 

        // Get Next Move (what about querying random location?) 
        for (i = 1; i < size && b[i]; i++) 
            ; 
        std::cout << endl; 
    } 
  */
  if(too_big(r * c) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  if(row_major_ordering){
    mkl_simatcopy('R', 'T', r, c, (float)1.0, A, c, r);
  }else{
    mkl_simatcopy('C', 'T', r, c, (float)1.0, A, r, c);
  }
} 

template <typename Dtype>
void cpu_axpby(const long long int N, const Dtype alpha, const Dtype* X, const Dtype beta, Dtype* Y) {
  bool Debug = false;
  if(Debug) {
    LOG("INT_MAX : "<<INT_MAX);
    LOG("N : "<<N);
    LOG("(int)N : "<<(int)N);
    LOG("alpha : "<<alpha);
  }

  if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  //cblas_sscal((int)N, alpha, X, 1);


  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, N);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int k = (long long int)th_id; k < N; k += (long long int)nthreads){
    //for (long long int k = (long long int)0; k < N; k+=(long long int)1){
      Y[k] = X[k] * alpha + Y[k] * beta;
    };
  }
}

template void cpu_axpby<float>(const long long int N, const float alpha, const float* X, const float beta, float* Y);
template void cpu_axpby<double>(const long long int N, const double alpha, const double* X, const double beta, double* Y);

// template <>
// void cpu_axpby<float>(const long long int N, const float alpha, const float* X, const float beta, float* Y) {
//   /*
//     y := a*x + b*y
//     where:
//     a and b are scalars
//     x and y are vectors each with n elements.
//   */
//   if(too_big(N) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
//   cblas_saxpby((int)N, alpha, X, 1, beta, Y, 1);
// }

// template <>
// void cpu_axpby<double>(const long long int N, const double alpha, const double* X, const double beta, double* Y) {
//   /*
//     y := a*x + b*y
//     where:
//     a and b are scalars
//     x and y are vectors each with n elements.
//   */
//   if(too_big(N) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}
//   cblas_daxpby((int)N, alpha, X, 1, beta, Y, 1);
// }


void cpu_axpby_test() {

  const long long int N = (long long int)5;
  const float alpha = (float)(-1.0);
  const float beta = (float)1.0;
  std::string blank = "";

  float* X = NULL;
  float* Y = NULL;
  X = (float *)malloc(N *  SIZE_OF(float)); 
  Y = (float *)malloc(N *  SIZE_OF(float));  
  checkErrors(X);
  checkErrors(Y);

  host_rng_uniform<float>(N, (float)(-10.0), (float)10.0, X);
  host_rng_uniform<float>(N, (float)(-10.0), (float)10.0, Y);

  if(1){
    print_host_array<float>(X, N, "X", strPreamble(blank));
    print_host_array<float>(Y, N, "X", strPreamble(blank));
  }

  cpu_axpby<float>(N, alpha, X, beta, Y);

  if(1){
    print_host_array<float>(X, N, "X", strPreamble(blank));
    print_host_array<float>(Y, N, "X", strPreamble(blank));
  }
  free(X);
  free(Y);  
}

template <typename Dtype>
void cpu_gemm(const bool TransA, const bool TransB, 
                     const long long int M, const long long int N, const long long int K,
                     const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                     Dtype* C, long long int start_, long long int num_) 
{
  bool Debug = false;
  if(Debug) LOG("cpu_gemm called.");
  std::string blank = "";
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    M, N, K
    M number of rows of matrix op(A) and C.
    N is number of columns of matrix op(B) and C.]
    K is number of rows of op(B) and columns of op(A).

    op(A) is M by K
    op(B) is K by N
    C is M by N
    cblas_sgemm(CblasRowMajor, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    performs C=alpha op ( A )op ( B ) + beta C
  */
  
  if(num_ <= (long long int)0 || num_ > (M * N) || start_ <= (long long int)0 || start_ > (M * N) || num_ - start_ > (M * N) ){
    num_ = M * N;
    start_ = (long long int)0;
    LOG("start_ : "<<start_) ;
    LOG("num_ : "<<num_) ;
  }
  long long int lda = (TransA == false) ? K : M;
  long long int ldb = (TransB == false) ? N : K;

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, num_);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    if(Debug){
      omp_set_lock(&printlock);
      LOG("num_ : "<<num_);
      LOG("(long long int)th_id : "<<(long long int)th_id);
      LOG("start_ + (long long int)th_id : "<<start_ + (long long int)th_id);
      omp_unset_lock(&printlock);
    }
    for(long long int k = (long long int)th_id; k < num_; k += (long long int)nthreads){
      long long int row = (start_ + k) / N;
      long long int col = (start_ + k) % N;
      std::string line = ToString<Dtype>(alpha) + "* ( ";
      Dtype temp = (Dtype)0.0;
      Dtype a = (Dtype)0.0;
      Dtype b = (Dtype)0.0;
      for(long long int i = (long long int)0; i < K; i += (long long int)1){
        a = (TransA == false) ? A[row * lda + i] : A[i * lda + row];
        b = (TransB == false) ? B[i * ldb + col] : B[col * ldb + i];
        if (::isinf(a) || ::isnan(a)){
          omp_set_lock(&printlock);
          LOG("k : "<<k);
          LOG("row : "<<row);
          LOG("col : "<<col);
          LOG("i : "<<i);
          LOG("a : "<<a);
          ABORT_IF_EQ(0, 0, "abort");
          omp_unset_lock(&printlock);
        }
        if (::isinf(b) || ::isnan(b)){
          omp_set_lock(&printlock);
          LOG("k : "<<k);
          LOG("row : "<<row);
          LOG("col : "<<col);
          LOG("i : "<<i);
          LOG("b : "<<b);
          ABORT_IF_EQ(0, 0, "abort");
          omp_unset_lock(&printlock);
        }
        temp += a * b;
        if(Debug){
          if (i > (long long int)0){
            line = ( line + " + " + ToString<Dtype>(a) + " * " + ToString<Dtype>(b) ).c_str();
          }else{
            line = ( line + ToString<Dtype>(a) + " * " + ToString<Dtype>(b) ).c_str();
          }
        }
        //temp += a * b * alpha;
      }
      line = (line + " ) ").c_str();
      if (::isinf(temp) || ::isnan(temp)){
        omp_set_lock(&printlock);
        LOG("k : "<<k);
        LOG("row : "<<row);
        LOG("col : "<<col);
        LOG("temp : "<<temp);
        ABORT_IF_EQ(0, 0, "abort");
        omp_unset_lock(&printlock);
      }
      temp *= alpha;
      if(beta == (Dtype)0.0){
        C[k] = temp;
      }else{
        C[k] *= beta;
        C[k] += temp;         
      }
      if (::isinf(C[k]) || ::isnan(C[k])){
        omp_set_lock(&printlock);
        LOG("C["<<row<<" , "<<col<<"] : "<<C[k]);
        ABORT_IF_EQ(0, 0, "abort");
        omp_unset_lock(&printlock);
      }
      if(Debug && (th_id == 0)) {
        omp_set_lock(&printlock);
        LOG("k : "<<k);
        LOG("C["<<row<<" , "<<col<<"] = "<<C[k]);
        //LOG(line);
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_gemm run time so far : "<<readable_time(program_time));
        omp_unset_lock(&printlock);
      }
    }
  }

  //cblas_dgemm(CblasRowMajor, (TransA == false) ? CblasNoTrans : CblasTrans, (TransB == false) ? CblasNoTrans : CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, N);

  if(Debug){

  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(Debug) LOG("cpu_gemm run time : "<<readable_time(program_time));
}

template void cpu_gemm<float>(const bool TransA, const bool TransB, 
                     const long long int M, const long long int N, const long long int K,
                     const float alpha, const float* A, const float* B, const float beta,
                     float* C, long long int start_, long long int num_);

template void cpu_gemm<double>(const bool TransA, const bool TransB, 
                      const long long int M, const long long int N, const long long int K,
                      const double alpha, const double* A, const double* B, const double beta,
                      double* C, long long int start_, long long int num_);

void cpu_gemm_test() {
  bool TransA = true;
  bool TransB = false;
  const long long int M = (long long int)5;
  const long long int N = (long long int)6;
  const long long int K = (long long int)7;
  const float alpha = (float)(-1.0);
  const float beta = (float)1.0;
  std::string blank = "";

  float* A = NULL;
  float* B = NULL;
  float* C = NULL;
  A = (float *)malloc(K * M *  SIZE_OF(float)); 
  B = (float *)malloc(N * K *  SIZE_OF(float)); 
  C = (float *)malloc(M * N *  SIZE_OF(float));  
  checkErrors(A);
  checkErrors(B);
  checkErrors(C);

  host_rng_uniform<float>(K * M, (float)(-10.0), (float)10.0, A);
  host_rng_uniform<float>(N * K, (float)(-10.0), (float)10.0, B);
  host_rng_uniform<float>(M * N, (float)(-10.0), (float)10.0, C);

  if(1){
    print_host_mtx(A, (TransA == false) ? M : K, (TransA == false) ? K : M, "A", true, strPreamble(blank));
    print_host_mtx(B, (TransB == false) ? K : N, (TransB == false) ? N : K, "B", true, strPreamble(blank));
    print_host_mtx(C, M, N, "C", true, strPreamble(blank));
  }


  cpu_gemm<float>(TransA, TransB, M, N, K, alpha, A, B, beta, C);

  if(1){
    print_host_mtx(A, (TransA == false) ? M : K, (TransA == false) ? K : M, "A", true, strPreamble(blank));
    print_host_mtx(B, (TransB == false) ? K : N, (TransB == false) ? N : K, "B", true, strPreamble(blank));
    print_host_mtx(C, M, N, "C", true, strPreamble(blank));
  }
  free(A);
  free(B); 
  free(C);  
}

template <>
void cpu_swap_ordering<float>(const long long int rows, const long long int cols, float *A, const bool row_major_ordering)
{
  bool Debug = false;
  if(Debug) {
    LOG("cpu_swap_ordering called");
    LOG("rows : "<<rows);
    LOG("cols : "<<cols);
    LOG("row_major_ordering : "<<row_major_ordering);
  }
  const long long int total = rows * cols;
  float* A_copy = NULL;
  A_copy  = (float *)malloc(total * SIZE_OF(float));
  checkErrors(A_copy);
  
  host_copy<float>(total, A, A_copy);
  if(Debug) {
    LOG("Here");
  }
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, total);
    if(Debug) {
      LOG("nthreads : "<<nthreads);
    }
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < total; j += (long long int)nthreads){
    //for(long long int j = (long long int)0; j < total; j += (long long int)1){
      if(!row_major_ordering){
          //starts in colum major ordering
        long long int row = j % rows;
        long long int col = j / rows;
          //i = row + rows * col;
        long long int new_i = cols * row + col;
        if(Debug) {
          omp_set_lock(&printlock);
          LOG("th_id : "<<th_id);
          LOG("j : "<<j);
          LOG("new_i : "<<new_i);

        }
        A[new_i] = A_copy[j];
        if(Debug){
          omp_unset_lock(&printlock);
        }
      }else{
          //starts in row major ordering
        long long int row = j / cols;
        long long int col = j % cols;
          //i = cols * row + col;
        long long int new_i = row + rows * col;
        if(Debug) {
          omp_set_lock(&printlock);
          LOG("th_id : "<<th_id);
          LOG("j : "<<j);
          LOG("new_i : "<<new_i);
        }
        A[new_i] = A_copy[j];
        if(Debug){
          omp_unset_lock(&printlock);
        }
      }
    }
  }
  free(A_copy);
  if(Debug) LOG("cpu_swap_ordering done");
}

template <typename Dtype>
Dtype cpu_min(const long long int n,  const Dtype* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  thrust::pair<const Dtype *, const Dtype *> tuple = thrust::minmax_element(thrust::host, x, x + n);

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

template int cpu_min<int>(const long long int n,  const int* x);


// template <typename Dtype>
// Dtype cpu_abs_max(const long long int n, Dtype* X){
//   Dtype max_ = std::abs(X[0]);
//   for(long long int i = (long long int)1; i < n; i+=(long long int)1){
//     Dtype temp = std::abs(X[i]);
//     if(temp > max_){
//       max_ = temp;
//     }
//   }
//   return max_;
// }

// template float cpu_abs_max<float>(const long long int n, float* X);
// template int cpu_abs_max<int>(const long long int n, int* X);

template <typename Dtype>
Dtype cpu_abs_max(const long long int n, const Dtype* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  thrust::pair<const Dtype *, const Dtype *> tuple = thrust::minmax_element(thrust::host, x, x + n);
  /*
    if int data[6] = {1, 0, 2, 2, 1, 3};
    thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
    result.first is data + 1
    result.second is data + 5
    *result.first is 0
    *result.second is 3
  */

  Dtype max = (Dtype)0.0;

  if(abs(*tuple.first) > abs(*tuple.second)){
    max =  abs(*tuple.first);
  }else{
    max =  abs(*tuple.second);
  };
  /*
    save_device_array_to_file<float>(x, n , "gradient");
    LOG(INFO) << "max : " <<max ;
    LOG(INFO) << "Press Enter to continue." ;
    std::cin.ignore();
  */

  return max;

}

template int cpu_abs_max<int>(const long long int n,  const int* x); 
template float cpu_abs_max<float>(const long long int n,  const float* x); 

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

template <typename Dtype> 
Dtype cpu_expected_value(const long long int n,  const Dtype* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  Dtype y = (Dtype)0.0;
  for(long long int i = (long long int)0; i < n; i += (long long int)1 ) {
    cpu_incremental_average(i + (long long int)1, &y, x[i]);
  };  
  return y;
}

template float cpu_expected_value<float>(const long long int n,  const float* x);

template <>
float cpu_expected_abs_value<float>(const long long int n,  const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  // setup arguments 
  abss<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = (float)0.0; 
  // compute norm 

  float s =  thrust::transform_reduce(thrust::host, x, x+n, unary_op, init, binary_op) ; 


  return (float)(s/(float)n);

}

//============================================================================================
// random utilities
//============================================================================================



// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, SIZE_OF(seed), f) == SIZE_OF(seed)) {
    fclose(f);
    return seed;
  }

  LOG("System entropy source not available, using fallback algorithm to generate seed instead.");
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


template <typename Dtype>
void fillupMatrix(Dtype *A , int lda , int rows, int cols)
{
  int seed = cluster_seedgen();
  int sign;
  for (int j = 0; j < cols; j++)
  {
    for (int i = 0; i < rows; i++)
    {
      sign = rand() % 2;
      A[i + lda*j ] = (Dtype) (((double)(((lda*i+j+seed) % 253)+1))/256.0, ((double)((((cols*i+j) + 123 + seed) % 253)+1))/256.0);
      if(sign==0){
        A[i + lda*j ] = A[i + lda*j ] * (Dtype)(-1);
      };
      /*
        if(i<5 && j==0){
            printf("A[%d + %d * %d] = %f \n", i, lda, j, A[i + lda*j ]);
        };
      */
    }
  }
}
/* Explicit instantiation */
template void  fillupMatrix<float>(float *A , int lda , int rows, int cols);
template void  fillupMatrix<double>(double *A , int lda , int rows, int cols);

template <typename Dtype>
Dtype nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
    b, std::numeric_limits<Dtype>::max());
}
template float nextafter(const float b);
template double nextafter(const double b);


template <typename Dtype>
void host_rng_uniform(const long long int n, const Dtype a, const Dtype b, Dtype* r) {
  //ABORT_IF_NEQ(0, 1, "host_rng_uniform not yet supported");
  //if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  bool Debug = false;
  if(Debug){
    LOG("a : "<<a);
    LOG("b : "<<b);
  }

  ABORT_IF_LESS(n, 1, "host_rng_uniform has n < 0");
  ABORT_IF_EQ(a, b, "host_rng_uniform has a = b");
  ABORT_IF_LESS(b, a, "host_rng_uniform has b < a");

  base_generator_type generator(static_cast<unsigned int>(cluster_seedgen()));

  // Define a uniform random number distribution which produces "double"
  // values between 0 and 1 (0 inclusive, 1 exclusive).
  boost::uniform_real<> uni_dist(static_cast<double>(a), static_cast<double>(b));
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

  std::cout.setf(std::ios::fixed);
  // You can now retrieve random numbers from that distribution by means
  // of a STL Generator interface, i.e. calling the generator as a zero-
  // argument function.
  long long int zero_count = (long long int)0;
  for(long long int i = (long long int)0; i < n; i+=(long long int)1){
    r[i] = static_cast<Dtype>(uni()) ;
    if(r[i] == (Dtype)0.0){
      zero_count+=(long long int)1;
      //LOG("r["<<i<<"] = 0");
    }
  }
  if(zero_count > (long long int)0){
    LOG(zero_count <<" out of "<< n<<" entries are zero in submitted vector.");
    LOG(((float)zero_count) / ((float)n)  <<" of the entries are zero in submitted vector.");
  }
  if(Debug){
    save_host_array_to_file<Dtype>(r, static_cast<int>(n), "random_array");
  }
  // boost::uniform_real<Dtype> random_distribution(a, nextafter<Dtype>(b));
  // boost::variate_generator<rng_t*, boost::uniform_real<Dtype> >
  //     variate_generator(_rng(), random_distribution);
  // for (int i = 0; i < n; ++i) {
  //   r[i] = variate_generator();
  // }
}

template void host_rng_uniform<float>(const long long int n, const float a, const float b, float* r);
template void host_rng_uniform<double>(const long long int n, const double a, const double b, double* r);


template <typename Dtype>
void host_rng_gaussian(const long long int n, const Dtype a, const Dtype b, Dtype* r) {
  //ABORT_IF_NEQ(0, 1, "host_rng_gaussian not yet supported");
  //if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  bool Debug = false;
  if(Debug){
    LOG("a : "<<a);
    LOG("b : "<<b);
  }

  ABORT_IF_LESS(n, 1, "host_rng_gaussian has n < 0");
  ABORT_IF_EQ((Dtype)0.0, b, "host_rng_gaussian has variance 0");
  ABORT_IF_LESS(b, (Dtype)0.0, "host_rng_gaussian has b < 0");

  base_generator_type generator(static_cast<unsigned int>(cluster_seedgen()));

  // Define a uniform random number distribution which produces "double"
  // values between 0 and 1 (0 inclusive, 1 exclusive).
  boost::normal_distribution<> nd(static_cast<double>(a), static_cast<double>(b));
  boost::variate_generator<base_generator_type&, boost::normal_distribution<> > uni(generator, nd);

  std::cout.setf(std::ios::fixed);
  // You can now retrieve random numbers from that distribution by means
  // of a STL Generator interface, i.e. calling the generator as a zero-
  // argument function.
  long long int zero_count = (long long int)0;
  for(long long int i = (long long int)0; i < n; i+=(long long int)1){
    r[i] = static_cast<Dtype>(uni()) ;
    if(r[i] == (Dtype)0.0){
      zero_count+=(long long int)1;
      //LOG("r["<<i<<"] = 0");
    }
  }
  if(zero_count > (long long int)0){
    LOG(zero_count <<" out of "<< n<<" entries are zero in submitted vector.");
    LOG(((float)zero_count) / ((float)n)  <<" of the entries are zero in submitted vector.");
  }
  if(Debug){
    save_host_array_to_file<Dtype>(r, static_cast<int>(n), "random_array");
  }
  // boost::uniform_real<Dtype> random_distribution(a, nextafter<Dtype>(b));
  // boost::variate_generator<rng_t*, boost::uniform_real<Dtype> >
  //     variate_generator(_rng(), random_distribution);
  // for (int i = 0; i < n; ++i) {
  //   r[i] = variate_generator();
  // }
}

template void host_rng_gaussian<float>(const long long int n, const float a, const float b, float* r);
template void host_rng_gaussian<double>(const long long int n, const double a, const double b, double* r);




//============================================================================================
// Old
//============================================================================================

void getRandIntsBetween(int *A , int lower_bd , int upper_bd, int num)
{
  for (int i = 0; i < num; i++)
  {
    /* generate secret number between lower_bd and upper_bd: */
    A[i] = rand() % (upper_bd - lower_bd) + lower_bd;
    /*
      if(i<5){
          printf("A[%d] = %d \n", i, A[i]);
      };
    */
    }
  //printf("\n");

  }

  void secondarySort(int *rows , int *cols, int lower_bd , int upper_bd,  int num)
  {
    int *rows_bottom = rows;
    int *cols_bottom = cols;
    int current=rows_bottom[0];
    int count = 0;
    int next = 0;
    bool repeats = 1;

    while (count <= num)
    {
      while(rows_bottom[next ] == current){
      //if (count<100){printf("row[%d] =  %d, col[%d] =  %d, next = %d \n",  count + next, rows[count + next], count + next,  cols[count + next], next);};
        next+=1;
      };
      if (next>0){
      //if (count<100){printf("need to sort, count = %d , next = %d\n", count, next);};
        thrust::sort(cols_bottom, cols_bottom + next );
      //check for repeats
        while(repeats){
          repeats = 0;
          for (int j = 0; j < next-1; j++)
          {
            if(cols_bottom[j] == cols_bottom[j+1])
            {
              printf("repeat found: [ %d , %d ]\n", lower_bd, upper_bd);
              for (int k = 0; k < next; k++)
              {
                printf("[ %d , %d ]\n", rows_bottom[k], cols_bottom[k]);
              };

              repeats = 1;
              cols_bottom[j+1] = rand() % (upper_bd - lower_bd) + lower_bd;
              thrust::sort(cols_bottom, cols_bottom + next );
            };
          };
        }
      }
      count+=next ;
      rows_bottom = rows_bottom + next ;
      cols_bottom = cols_bottom + next ;
      current=rows_bottom[next ]; 
      next = 0;
    };
  }

  void gatherIndices(int *rows , int *cols, int num_rows , int num_cols)
  {
    int num = num_rows * num_cols;
    for (int i = 0; i < num_rows; i++)
    {
      for (int j = 0; j < num_cols; j++)
      {
        rows[i*num_cols +j]=i;
        cols[i*num_cols +j]=j; 
      };
    };
  }

  bool selectIndices(int *rows , int *cols, int *select_rows , int *select_cols,  int num_rows , int num_cols,  int nnz)
  {
    struct timeval time_start, time_end, while_start, while_end;
    double selectIndices_time, while_time;

    int* indicies = NULL;
    indicies  = (int *)malloc(nnz * SIZE_OF(int));
    checkErrors(indicies);

    gettimeofday(&time_start, NULL);
  //generate random numbers:
    for (int i=0;i<nnz;i++)
    {   
      gettimeofday(&while_start, NULL);
    bool check; //variable to check or number is already used
    int n; //variable to store the number in
    do
    {
      n=rand()%(num_rows * num_cols);
        //check or number is already used:
      check=true;
      for (int j=0;j<i;j++)
      {
        if (n == indicies[j]) //if number is already used
        {
          check=false; //set check to false
          break; //no need to check the other elements of value[]
        }
      }
      gettimeofday(&while_end, NULL);
      while_time = (while_end.tv_sec * 1000 +(while_end.tv_usec/1000.0))-(while_start.tv_sec * 1000 +(while_start.tv_usec/1000.0));
    }while (!check && while_time < (double)30000.0); //loop until new, unique number is found
    gettimeofday(&time_end, NULL);
    selectIndices_time = (time_end.tv_sec * 1000 +(time_end.tv_usec/1000.0))-(time_start.tv_sec * 1000 +(time_start.tv_usec/1000.0));

    if(while_time < (double)30000.0 && selectIndices_time <(double)60000.0){
      indicies[i]=n; //store the generated number in the array
    }else{
      //printf("while_time: %d\n", while_time < (double)30.0);
      //printf("selectIndices_time: %d\n", selectIndices_time <(double)300.0);
      return false;
    };
  };


  thrust::sort(indicies, indicies + nnz );
  for (int j = 0; j < nnz; j++)
  {
    select_rows[j]=rows[indicies[j]];
    select_cols[j]=cols[indicies[j]];
  };
  free(indicies);

  return true;
}








void initializeVal(double * A, int rows, int cols , int ld, double val)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      A[j*ld +i] = val;
    }
  }
}






template <typename Dtype>
void fillupMatrixDebug(Dtype *A , int lda , int rows, int cols)
{
  for (int j = 0; j < cols; j++)
  {
    for (int i = 0; i < rows; i++)
    {
      A[i + lda*j ] = (Dtype) (i + j);
    }
  }
}




//============================================================================================
// Prints and Saves
//============================================================================================

template<typename Dtype>
std::string ToString(const Dtype val)
{
  std::stringstream stream;
  stream << val;
  return stream.str();
}
template std::string ToString<bool>(const bool val);
template std::string ToString<int>(const int val);
//template std::string ToString<long long int>(const long long int val);
template std::string ToString<float>(const float val);
template std::string ToString<double>(const double val);

extern "C" std::string readable_time(double ms){
  if(ms >= (double)0.0){
    int ms_int = (int)ms;
    int hours = ms_int / 3600000;
    ms_int = ms_int % 3600000;
    int minutes = ms_int / 60000;
    ms_int = ms_int % 60000;
    int seconds = ms_int / 1000;
    ms_int = ms_int % 1000;

    if(hours == 7) ABORT_IF_GT(minutes, 45, "Stopping early to prevent saving incomplete data.");

    return (ToString<int>(hours) + ":" +ToString<int>(minutes) + ":" + ToString<int>(seconds) + ":" + ToString<int>(ms_int));
  }else{
    int ms_int = (int)(abs(ms));
    int hours = ms_int / 3600000;
    ms_int = ms_int % 3600000;
    int minutes = ms_int / 60000;
    ms_int = ms_int % 60000;
    int seconds = ms_int / 1000;
    ms_int = ms_int % 1000;

    std::string neg_ = "-";
    return (neg_ + ToString<int>(hours) + ":" + ToString<int>(minutes) + ":" + ToString<int>(seconds) + ":" + ToString<int>(ms_int));
  }
}

template<typename Dtype>
void printPartialMatrices(Dtype * A, Dtype * B, int rows, int cols , int ld)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f , ",  A[j*ld +i]);
    }
    printf("    ");
    for (int j = 0; j < cols; j++)
    {
      printf("%f , ",  B[j*ld +i]);
    }
    printf(";\n");
  }
}

template void printPartialMatrices<float>(float * A, float * B, int rows, int cols , int ld);
template void printPartialMatrices<int>(int * A, int * B, int rows, int cols , int ld);


template<typename Dtype>
void printPartialMtx(Dtype * A, int rows, int cols , int ld)
{
// This function assumes column major ordering, such that ld corresponds to the number of rows
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      if (j<cols-1){ printf("%d , ",  A[j*ld +i]); }
      else{ printf("%d ;\n ",  A[j*ld +i]); }
    }
  }
}
template void printPartialMtx<int>(int * A, int rows, int cols , int ld);
template void printPartialMtx<float>(float * A, int rows, int cols , int ld);
template void printPartialMtx<double>(double * A, int rows, int cols , int ld);


template<typename Dtype>
void print_host_array(const Dtype* host_pointer, int count, std::string title, std::string file_line)
{
  if(file_line != ""){
    LOG2(file_line, title<<" : ");
  }else{
    LOG(title<<" : ");
  }
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
  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }
}

template void print_host_array<int>(const int* host_pointer, int count, std::string title, std::string file_line);
template void print_host_array<float>(const float* host_pointer, int count, std::string title, std::string file_line);

template<typename Dtype>
void print_host_mtx(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line)
{
  //assumes row major order
  std::string line = (title + " : \r\n").c_str();
  if(row_major_order){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        //line = (line + ToString<Dtype>(A_host[i + j * rows])).c_str();
        line = (line + ToString<Dtype>(A_host[i * cols + j])).c_str();
        if(j < cols - 1){
          line = (line + ", ").c_str();
        }
      }
      line = (line + "\r\n").c_str();
      //line = (line + "; ").c_str();
    }
  }else{
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        line = (line + ToString<Dtype>(A_host[i + j * rows])).c_str();
        //line = (line + ToString<Dtype>(A_host[i * cols + j])).c_str();
        if(j < cols - 1){
          line = (line + ", ").c_str();
        }
      }
      line = (line + "\r\n").c_str();
      //line = (line + "; ").c_str();
    }    
  }
  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }
}

template void print_host_mtx<int>(const int* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
//template void print_host_mtx<long long int>(const long long int* A_host, int rows, int cols, std::string title, bool row_major_order, std::string file_line);
template void print_host_mtx<float>(const float* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void print_host_mtx<double>(const double* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);


template<typename Dtype>
void save_host_array_to_file(const Dtype* A_host, int count, std::string title, std::string file_line)
{
  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<ToString(A_host[ i ]);
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
      entries.flush();
    };
  };
  entries.close();
  if(file_line != ""){
    LOG2(file_line, "save_host_array_to_file "<< title << " has "<< count<<" entries");
  }
}

template void save_host_array_to_file<int>(const int* A_host, int count, std::string title, std::string file_line);
template void save_host_array_to_file<float>(const float* A_host, int count, std::string title, std::string file_line);
template void save_host_array_to_file<double>(const double* A_host, int count, std::string title, std::string file_line);

template<typename Dtype>
void get_host_array_from_saved_txt(const Dtype* A_host, int count, std::string title)
{
  ABORT_IF_NEQ(0, 1, "Function not ready.");
  std::ifstream A_host_file ((ToString(title + ".txt")).c_str());

  for (int i = 0; i < count; i++){
    //A_host_file>>A_host[i];
  }

  //LOG(INFO)<<"It_count grabbed from file: "<<It_count;
  A_host_file.close();
}

template void get_host_array_from_saved_txt<int>(const int* A_host, int count, std::string title);
template void get_host_array_from_saved_txt<float>(const float* A_host, int count, std::string title);
template void get_host_array_from_saved_txt<double>(const double* A_host, int count, std::string title);

template<typename Dtype>
void append_host_array_to_file(const Dtype* A_host, int count, std::string title, std::string file_line)
{


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str(), std::ofstream::app);
  //entries<<"; ";
  entries<<"\r\n";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      entries<<", ";
    };
    entries.flush();
  };
  entries.close();
  if(file_line != ""){
    LOG2(file_line, "append_host_array_to_file "<< title << " has "<< count<<" entries");
  }
}

template void append_host_array_to_file<int>(const int* A_host, int count, std::string title, std::string file_line);
template void append_host_array_to_file<float>(const float* A_host, int count, std::string title, std::string file_line);
template void append_host_array_to_file<double>(const double* A_host, int count, std::string title, std::string file_line);

template<typename Dtype>
void save_host_arrays_side_by_side_to_file(const Dtype* A_host, const Dtype* B_host, int count, std::string title, std::string file_line)
{


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
     entries.flush();
  };
  entries.close();
  //LOG("file saved");
  if(file_line != ""){
    LOG2(file_line, "save_host_array_to_file "<< title << " has "<< count<<" entries");
  }

}

template void save_host_arrays_side_by_side_to_file<int>(const int* A_host, const int* B_host, int count, std::string title, std::string file_line);
template void save_host_arrays_side_by_side_to_file<float>(const float* A_host, const float* B_host, int count, std::string title, std::string file_line);
template void save_host_arrays_side_by_side_to_file<double>(const double* A_host, const double* B_host, int count, std::string title, std::string file_line);

template<typename Dtype>
void save_host_array_side_by_side_with_device_array(const Dtype* A_host, const Dtype* B_dev, int count, std::string title, std::string file_line)
{
  Dtype *B_host = NULL;
  B_host=(Dtype *)malloc(count * SIZE_OF(Dtype));
  checkErrors(B_host);
  checkCudaErrors(cudaMemcpy(B_host, B_dev, count * SIZE_OF(Dtype), cudaMemcpyDeviceToHost));

  Dtype mean_abs_nonzero_ = (Dtype)0.0;
  cpu_mean_abs_nonzero<Dtype>((long long int)count, B_host, &mean_abs_nonzero_, true);

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
     entries.flush();
  };
  entries.close();
  //LOG("file saved");
  if(file_line != ""){
    LOG2(file_line, "save_host_array_side_by_side_with_device_array "<< title << " has "<< count<<" entries");
  }
  free(B_host);
}

template void save_host_array_side_by_side_with_device_array<int>(const int* A_host, const int* B_dev, int count, std::string title, std::string file_line);
template void save_host_array_side_by_side_with_device_array<float>(const float* A_host, const float* B_dev, int count, std::string title, std::string file_line);
template void save_host_array_side_by_side_with_device_array<double>(const double* A_host, const double* B_dev, int count, std::string title, std::string file_line);



void save_host_arrays_side_by_side_to_file_(const int* A_host, const int* B_host, 
 const float* C_host, int count, std::string title)
{


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
  //LOG("file saved");
  entries.close();

}



template<typename Dtype>
void save_host_mtx_to_file(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line)
{
  if(file_line != ""){
    LOG2(file_line, "save_host_mtx_to_file "<< title<< " running.");
  }
  //assumes row major order
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());

  std::streamsize ss = std::cout.precision();
  std::cout << "Initial precision = " << ss << '\n';

  //entries<<"[ ";
  if(row_major_order){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        //entries<<A_host[i + j * rows];
        entries<<std::setprecision(8)<<A_host[((long long int)i) * ((long long int)cols) + ((long long int)j)];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries.flush();
      if(i < rows - 1){
        entries<<"\r\n";
        //entries<<"; ";
      }
    }
  }else{
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        entries<<std::setprecision(8)<<A_host[((long long int)i) + ((long long int)rows) * ((long long int)j)];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries.flush();
      if(i < rows - 1){
        entries<<"\r\n";
        //entries<<"; ";
      }
    }    
  }
  entries.close();
  if(file_line != ""){
    LOG2(file_line, "save_host_mtx_to_file "<< title<< " finished.");
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(1) LOG("save_host_mtx_to_file run time : "<<readable_time(program_time));
}

template void save_host_mtx_to_file<int>(const int* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
//template void save_host_mtx_to_file<long long int>(const long long int* A_host, int rows, int cols, std::string title, bool row_major_order, std::string file_line);
template void save_host_mtx_to_file<float>(const float* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void save_host_mtx_to_file<double>(const double* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);


template<typename Dtype>
void append_host_mtx_to_file(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line)
{


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str(), std::ofstream::app);
  //entries<<"[ ";
  if(row_major_order){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        //entries<<A_host[i + j * rows];
        entries<<A_host[((long long int)i) * ((long long int)cols) + ((long long int)j)];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries.flush();
      //entries<<"\r\n";
      if(i < rows - 1){
        entries<<"; ";
      }
    }
  }else{
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        entries<<A_host[((long long int)i) + ((long long int)rows) * ((long long int)j)];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries.flush();
      //entries<<"\r\n";
      if(i < rows - 1){
        entries<<"; ";
      }
    }    
  }
  entries.close();
  if(file_line != ""){
    LOG2(file_line, "save_host_mtx_to_file "<< title);
  }
}

template void append_host_mtx_to_file<int>(const int* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void append_host_mtx_to_file<float>(const float* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void append_host_mtx_to_file<double>(const double* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);





void save_map(std::map<int, int>* items_dictionary, std::string title){

  std::map<int, int>::iterator it = items_dictionary -> begin();

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());

  // Iterate over the map using Iterator till end.
  while (it != items_dictionary->end())
  {
    entries<<it->first;
    entries<<", ";
    entries<<it->second;
    entries<<"\r\n";
    it++;
  }
}




//============================================================================================
// Me Made
//============================================================================================

void cpu_fill_training_mtx(const long long int ratings_rows_training, const long long int ratings_cols_training,  
 const long long int num_entries_CU, const bool row_major_ordering,
 const int* csr_format_ratingsMtx_userID_host_training,
 const int* coo_format_ratingsMtx_itemID_host_training,
 const float* coo_format_ratingsMtx_rating_host_training,
 float* full_training_ratings_mtx)
{
  LOG("cpu_fill_training_mtx called...");
  bool Debug = false;
  LOG("ratings_rows_training : "<< ratings_rows_training);
  LOG("ratings_cols_training : "<< ratings_cols_training);
  //row major ordering
  for(long long int row = (long long int)0; row < ratings_rows_training; row +=(long long int)1){
    for(long long int i = (long long int)(csr_format_ratingsMtx_userID_host_training[row]); i < (long long int)(csr_format_ratingsMtx_userID_host_training[row + 1]); i+=(long long int)1){
      long long int col = (long long int)(coo_format_ratingsMtx_itemID_host_training[i]);
      float val = coo_format_ratingsMtx_rating_host_training[i]; 
      if(row_major_ordering){
        if(Debug) {
          LOG("num_entries_CU : "<< num_entries_CU);
          LOG("ratings_rows_training : "<< ratings_rows_training);
          LOG("ratings_cols_training : "<< ratings_cols_training);
          LOG("full_training_ratings_mtx["<<ratings_cols_training<<" * "<< row<<" + "<<col<<"] = "<<val) ;
          LOG("full_training_ratings_mtx["<<ratings_cols_training *  row + col<<"] = "<<val) ;
          LOG(val<<" = coo_format_ratingsMtx_rating_host_training["<<i<<"]") ;
          LOG(val<<" = "<<coo_format_ratingsMtx_rating_host_training[i]) ;
        }
        full_training_ratings_mtx[ratings_cols_training * row + col] = val;
      }else{
        full_training_ratings_mtx[row + ratings_rows_training * col] = val;
      };
    }
  }
  LOG("cpu_fill_training_mtx finished...");
}

/*
  template < typename Dtype>
  void cpu_shuffle_array(const long long int n,  Dtype* x)
  {
    bool Debug = false;
    if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    double* order = (double *)malloc(n * SIZE_OF(double));
    checkErrors(order);
    host_rng_uniform<double>(n, (double)0.0, (double)1.0, order);
    if(Debug){
      save_host_array_to_file<Dtype>(x, static_cast<int>(n), "before_shuffle");
    }
    thrust::sort_by_key(thrust::host, order, order + n, x);
    if(Debug){
      save_host_array_to_file<Dtype>(x, static_cast<int>(n), "after_shuffle");
    }
    free(order);

  }

*/

template < typename Dtype>
void cpu_shuffle_array(const long long int n,  Dtype* x)
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  float* order = NULL;
  order  = (float *)malloc(n * SIZE_OF(float));
  checkErrors(order);

  host_rng_uniform<float>(n, (float)0.0, (float)1.0, order);

  quickSort_by_key<float,Dtype>(order, 0, n - 1, x);


  free(order);

}

template void cpu_shuffle_array<float>(const long long int n,  float* x);
template void cpu_shuffle_array<int>(const long long int n,  int* x);




void cpu_shuffle_map_second(const long long int M, std::map<int, int>* items_dictionary )
{

  if(too_big(M) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  
  bool Debug = false;
  int * indicies_host = NULL;
  int * indicies_dev;

  CUDA_CHECK(cudaMalloc((void**)&indicies_dev, M * SIZE_OF(int)));
  indicies_host = (int *)malloc(M * SIZE_OF(int));
  checkErrors(indicies_host);

  gpu_set_as_index(indicies_dev, M);
  CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, M * SIZE_OF(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(indicies_dev));

  cpu_shuffle_array<int>(M, indicies_host);

  std::map<int, int>::iterator it = items_dictionary -> begin();
  int i = 0; 
  // Iterate over the map using Iterator till end.
  while (it != items_dictionary->end())
  {
    it->second = indicies_host[i];
    i++;
    it++;
  }

  free(indicies_host);

} 



template<typename Dtype>
void cpu_set_all(Dtype* x, const long long int N, Dtype alpha)
{
  for(long long int i=(long long int)0; i < N; i+=(long long int)1) {
    //origin+x1+rows*y1
    x[i]=alpha;
  };
}

template void cpu_set_all<int>(int* x, const long long int N, int alpha);
template void cpu_set_all<float>(float* x, const long long int N, float alpha);


template <>
void cpu_set_as_index<int>(int* x, const long long int rows, const long long int cols) 
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, rows * cols);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < rows * cols; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < rows * cols; i+=(long long int)1) {
      int row = (int)(i % rows);
      //int col = (int)(i / rows);
      x[i] = row;
    }
  }
  if(0) LOG("finished call to cpu_set_as_index") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_set_as_index run time : "<<readable_time(program_time));
}

void cpu_shuffle_mtx_rows_or_cols(const long long int M, const long long int N, bool row_major_ordering, float* x, bool shuffle_rows)
{
  bool Debug = true;
  int * indicies_host = NULL;

  // A in row major ordering is equivalent to A^T in column major ordering
  //  A in column major ordering is equivalent to A^T in row major ordering
  
  if(Debug) LOG("cpu_shuffle_mtx_rows_or_cols called") ;
  if(too_big(M) ) {
    ABORT_IF_NEQ(0, 1,"Long long long int too big");
  }
  if(too_big(N) ) {
    ABORT_IF_NEQ(0, 1,"Long long long int too big");
  }

  if(shuffle_rows){
    if(Debug) LOG("shuffle_rows is true") ;
    
    indicies_host = (int *)malloc(M * SIZE_OF(int));
    checkErrors(indicies_host);
    cpu_set_as_index(indicies_host, M, (long long int)1);
    cpu_shuffle_array<int>(M, indicies_host);

    if(row_major_ordering){
      if(Debug) LOG("row_major_ordering is true") ;
      cpu_permute<float>(x, indicies_host, N, M, false); 
    }else{
      if(Debug) LOG("col major ordering") ;
      cpu_permute<float>(x, indicies_host, M, N, true); 
    };
  }else{// shuffle columns
    if(Debug) LOG("shuffle columns") ;

    indicies_host = (int *)malloc(N * SIZE_OF(int));
    checkErrors(indicies_host);
    cpu_set_as_index(indicies_host, N, (long long int)1);
    cpu_shuffle_array<int>( N, indicies_host);

    if(row_major_ordering){
      if(Debug) LOG("row_major_ordering is true") ;
      cpu_permute<float>(x, indicies_host, N, M, true); 
    }else{
      if(Debug) LOG("col major ordering") ;
      cpu_permute<float>(x, indicies_host, M, N, false); 
    };
  }
  free(indicies_host);
} 


void cpu_get_cosine_similarity(const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_host,
  const int* coo_format_ratingsMtx_itemID_host,
  const float* coo_format_ratingsMtx_rating_host,
  float* cosine_similarity) 
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) LOG("called cpu_get_cosine_similarity") ;

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, ratings_rows * ratings_rows);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int entry = (long long int)th_id; entry < ratings_rows * ratings_rows; entry += (long long int)nthreads){
    //for( long long int entry = (long long int)0; entry < ratings_rows * ratings_rows; entry+=(long long int)1){
      long long int whole_index = from_below_diag_to_whole_faster(entry, ratings_rows);
      int user_i = (int)(whole_index % ratings_rows);
      int user_j = (int)(whole_index / ratings_rows);
      if( user_i == user_j){
        LOG("Thread "<<th_id<<" is bad : " <<user_i) ;
        cosine_similarity[entry] = (float)1.0;
      }else{
        int   count   = 0;
        float num     = (float)0.0;
        float denom_i = (float)0.0;
        float denom_j = (float)0.0;
        int start_j = csr_format_ratingsMtx_userID_host[user_j];
        for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
          int user_i_itemID = coo_format_ratingsMtx_itemID_host[i];
          int user_j_itemID = 0;
          denom_i += pow(coo_format_ratingsMtx_rating_host[i], (float)2.0) ;
          for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
            user_j_itemID = coo_format_ratingsMtx_itemID_host[j];
            denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
            if( user_i_itemID == user_j_itemID){
              count   += 1;
              num     += coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j] ;
              start_j = j + 1;
              break;
            }else if(user_i_itemID < user_j_itemID){
              start_j = j;
              denom_j -= pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ;
              break;
            }
          }
        }
        if(count > 0){
            //float temp = num / sqrt(denom_i * denom_j);
          float temp_i = (float)csr_format_ratingsMtx_userID_host[user_i + 1] - (float)csr_format_ratingsMtx_userID_host[user_i];
          float temp_j = (float)csr_format_ratingsMtx_userID_host[user_j + 1] - (float)csr_format_ratingsMtx_userID_host[user_j];
          float temp = ((float)count) / sqrtf(temp_i * temp_j);
          cosine_similarity[entry] = temp;
          if (::isinf(temp) || ::isnan(temp)){
            LOG("Thread "<<th_id<<" is bad : " <<temp) ;
          };
        }else{
         cosine_similarity[entry] = (float)0.0;
        }
      }
    }
  }

  if(0) LOG("finished call to cpu_get_cosine_similarity") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_get_cosine_similarity run time : "<<readable_time(program_time));
}


long long int cpu_compute_hidden_values (const long long int ratings_rows, 
  const long long int ratings_cols, const int Top_N, const long long int num_entries,
  const int* csr_format_ratingsMtx_userID_host,
  const int* coo_format_ratingsMtx_itemID_host,
  const float* coo_format_ratingsMtx_rating_host,
  const std::vector<std::vector<int> >* top_N_most_sim_itemIDs_host,
  const std::vector<std::vector<float> >* top_N_most_sim_item_similarity_host,
  int**   coo_format_ratingsMtx_userID_host_new,
  int**   coo_format_ratingsMtx_itemID_host_new,
  float** coo_format_ratingsMtx_rating_host_new)
{
  bool print = true;
  bool Global_Debug = false;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) {
    LOG("called cpu_compute_hidden_values") ;
  }

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    //if(!Global_Debug) 
      nthreads = (int)std::min((long long int)nProcessors, ratings_rows);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock, worklock;
    omp_init_lock(&printlock);
    omp_init_lock(&worklock);
  #endif
  if(0) {
    LOG("ratings_rows : "<< ratings_rows) ;
    LOG("ratings_cols : "<< ratings_cols) ;
    LOG("num_entries : "<< num_entries) ;
    LOG("Top_N : "<< Top_N) ;
    LOG("nthreads : "<< nthreads) ;
  }
  int new_num_ratings[ratings_rows];

  int max_num_new_ratings = 0;
  int user_with_max_num_new_ratings = 0;
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int user = (long long int)th_id; user < ratings_rows; user += (long long int)nthreads)
    {
      if(Global_Debug && 0) {
        //omp_set_lock(&printlock);
        LOG("user : "<<user);
        //omp_unset_lock(&printlock);
      }
      std::set<int> really_rated_itemIDS;
      std::set<int> hidden_rated_itemIDS;
      for(int i = csr_format_ratingsMtx_userID_host[user]; i < csr_format_ratingsMtx_userID_host[user + 1]; i++){
        int user_itemID = coo_format_ratingsMtx_itemID_host[i];
        really_rated_itemIDS.insert(user_itemID);
      }
      if(Global_Debug && 0) {
        //omp_set_lock(&printlock);
        LOG("really_rated_itemIDS.size() : "<<really_rated_itemIDS.size());
        //omp_unset_lock(&printlock);
      }
      for(int i = csr_format_ratingsMtx_userID_host[user]; i < csr_format_ratingsMtx_userID_host[user + 1]; i++){
        int user_itemID = coo_format_ratingsMtx_itemID_host[i];
        //which items have item user_itemID listed as a similar item? 
        int vec_length = (*top_N_most_sim_itemIDs_host)[user_itemID].size();
        if(Global_Debug && 0) {
          //omp_set_lock(&printlock);
          LOG("   user_itemID : "<<user_itemID);
          LOG("   vec_length : "<<vec_length);
          //omp_unset_lock(&printlock);
        }
        for(long long int j = (long long int)0; j < (long long int)vec_length; j+=(long long int)1){
          int other_similar_itemID = (*top_N_most_sim_itemIDs_host)[user_itemID][j];
            if(Global_Debug && 0) {
              //omp_set_lock(&printlock);
              LOG("      j : "<<j);
              LOG("      other_similar_itemID : "<<other_similar_itemID);
              //omp_unset_lock(&printlock);
            }
            // make sure the user didn't actually rate the item
            if(really_rated_itemIDS.find(other_similar_itemID) == really_rated_itemIDS.end()){
              if(hidden_rated_itemIDS.find(other_similar_itemID) == hidden_rated_itemIDS.end()){
                hidden_rated_itemIDS.insert(other_similar_itemID);
                if(Global_Debug && 0) {
                  //omp_set_lock(&printlock);
                  //LOG("th_id : "<<th_id);
                  LOG("            new! adding..");
                  //omp_unset_lock(&printlock);
                }
              }else{
                if(Global_Debug && 0) {
                  //omp_set_lock(&printlock);
                  //LOG("th_id : "<<th_id);
                  LOG("            already added to hidden ratings");
                  //omp_unset_lock(&printlock);
                }              
              }
            }else{
              if(Global_Debug && 0) {
                //omp_set_lock(&printlock);
                //LOG("th_id : "<<th_id);
                LOG("         already rated");
                //omp_unset_lock(&printlock);
              }
            }
          }
        
        if(Global_Debug && 0) {
          omp_set_lock(&printlock);
          LOG("th_id : "<<th_id);
          LOG("user : "<<user);
          LOG("user_itemID : "<<user_itemID);
          LOG("really_rated_itemIDS["<<user<<"].size() : "<<really_rated_itemIDS.size());
          LOG("hidden_rated_itemIDS["<<user<<"].size() : "<<hidden_rated_itemIDS.size());
          LOG("new_num_ratings["<<user<<"] : "<<new_num_ratings[user]);

          gettimeofday(&program_end, NULL);
          program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
          LOG("cpu_compute_hidden_values run time so far: "<<readable_time(program_time));
          omp_unset_lock(&printlock);
        }
      }
      new_num_ratings[user] = static_cast<int>(really_rated_itemIDS.size()) + static_cast<int>(hidden_rated_itemIDS.size());

      omp_set_lock(&worklock);
      int temp = max_num_new_ratings;
      max_num_new_ratings = std::max(static_cast<int>(hidden_rated_itemIDS.size()),max_num_new_ratings);
      if(temp != max_num_new_ratings)
        user_with_max_num_new_ratings = (int)user;
      omp_unset_lock(&worklock);

      really_rated_itemIDS.erase ( really_rated_itemIDS.begin(), really_rated_itemIDS.end() );
      hidden_rated_itemIDS.erase ( hidden_rated_itemIDS.begin(), hidden_rated_itemIDS.end() );
    }// end for user

  }// end parallel
  
  //now build new crs arrays
  int new_coo_count = cpu_asum<int>(ratings_rows, new_num_ratings);
  if(Global_Debug){
    save_host_array_to_file<int>(new_num_ratings, ratings_rows, "new_num_ratings");
  }
  if(print){
    LOG("old coo count : "<<num_entries);
    LOG("new_coo_count : "<<new_coo_count);
    LOG("maximum hidden ratings added for a given user : "<<max_num_new_ratings);
    LOG("user with maximum hidden ratings added : "<<user_with_max_num_new_ratings);
  }
  cum_asum<int>(ratings_rows, new_num_ratings);
  if(Global_Debug){
    save_host_array_to_file<int>(new_num_ratings, ratings_rows, "cummulative_new_num_ratings");
  }
  (*coo_format_ratingsMtx_userID_host_new) = (int *)  malloc(new_coo_count *  SIZE_OF(int)); 
  (*coo_format_ratingsMtx_itemID_host_new) = (int *)  malloc(new_coo_count *  SIZE_OF(int)); 
  (*coo_format_ratingsMtx_rating_host_new) = (float *)malloc(new_coo_count *  SIZE_OF(float)); 
  checkErrors((*coo_format_ratingsMtx_userID_host_new));
  checkErrors((*coo_format_ratingsMtx_itemID_host_new));
  checkErrors((*coo_format_ratingsMtx_rating_host_new));

  //gettimeofday(&program_start, NULL);
  //Global_Debug = true;
  // #ifdef _OPENMP
  //   nthreads = 1;
  //   omp_set_num_threads(nthreads);
  // #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int user = (long long int)th_id; user < ratings_rows; user += (long long int)nthreads){
      //for(long long int user = (long long int)0; user < ratings_rows; user += (long long int)1){
      bool thread_debug = false;
      if(Global_Debug) {
        //omp_set_lock(&printlock);
        LOG("th_id : "<<th_id);
        LOG("user : "<<user);
        thread_debug = true;
        //omp_unset_lock(&printlock);
      }
      long long int first_place = (long long int)0;
      if(user != (long long int)0){
        first_place = new_num_ratings[user - 1];
      }
      long long int place_ = first_place;

      std::set<int> really_rated_itemIDS;
      std::map<int, std::pair <float,float> > hidden_rated_itemIDS;
      // std::set<int> hidden_rated_itemIDS;
      for(int i = csr_format_ratingsMtx_userID_host[user]; i < csr_format_ratingsMtx_userID_host[user + 1]; i++){
        int user_itemID = coo_format_ratingsMtx_itemID_host[i];
        really_rated_itemIDS.insert(user_itemID);
        (*coo_format_ratingsMtx_userID_host_new)[place_] = user;
        (*coo_format_ratingsMtx_itemID_host_new)[place_] = user_itemID;
        (*coo_format_ratingsMtx_rating_host_new)[place_] = coo_format_ratingsMtx_rating_host[i];
        place_ +=(long long int)1;  
      }
      int vec_length = 0;
      for(int i = csr_format_ratingsMtx_userID_host[user]; i < csr_format_ratingsMtx_userID_host[user + 1]; i++){
        int user_itemID = coo_format_ratingsMtx_itemID_host[i];
        //which items have item user_itemID listed as a similar item? 
        vec_length = (*top_N_most_sim_itemIDs_host)[user_itemID].size();
        if(Global_Debug || thread_debug) {
          omp_set_lock(&printlock);
          LOG("   user_itemID : "<<user_itemID);
          LOG("   vec_length : "<<vec_length);
          omp_unset_lock(&printlock);
        }
        for(long long int j = (long long int)0; j < (long long int)vec_length; j+=(long long int)1){
          int other_similar_itemID = (*top_N_most_sim_itemIDs_host)[user_itemID][j];
          // make sure the user didn't actually rate the item
          if(really_rated_itemIDS.find(other_similar_itemID) == really_rated_itemIDS.end()){
            float add_to_num = coo_format_ratingsMtx_rating_host[i] * (*top_N_most_sim_item_similarity_host)[user_itemID][j];
            float add_to_denom = (*top_N_most_sim_item_similarity_host)[user_itemID][j]; 
            //check if you have already added this hidden rating
            std::map<int, std::pair <float,float> >::iterator it = hidden_rated_itemIDS.find(other_similar_itemID);
            if(it == hidden_rated_itemIDS.end()){
              std::pair <float,float> product1; 
              product1 = std::make_pair(add_to_num, add_to_denom);
              hidden_rated_itemIDS[other_similar_itemID] = product1;
              if((Global_Debug || thread_debug) && 0) {
                //omp_set_lock(&printlock);
                //LOG("th_id : "<<th_id);
                LOG("   user_itemID : "<<user_itemID);
                LOG("   other_similar_itemID : "<<other_similar_itemID);
                LOG("   num : "<<add_to_num);
                LOG("   denom : "<<add_to_denom);
                //omp_unset_lock(&printlock);
              }
            }else{
              // std::pair <float,float> num_denum = it->second;
              // num_denum.first += add_to_num;
              // num_denum.second += add_to_denom; 
              // hidden_rated_itemIDS[other_similar_itemID] = num_denum;
              it->second.first += add_to_num;
              it->second.second += add_to_denom;
              if((Global_Debug || thread_debug) && 0) {
                //omp_set_lock(&printlock);
                //LOG("th_id : "<<th_id);
                LOG("   user_itemID : "<<user_itemID);
                LOG("   other_similar_itemID : "<<other_similar_itemID);
                LOG("   num : "<<it->second.first);
                LOG("   denom : "<<it->second.second);
                //omp_unset_lock(&printlock);
              }
            }
          } //if not really rated
        } //for j
      } // for i item
      really_rated_itemIDS.erase ( really_rated_itemIDS.begin(), really_rated_itemIDS.end() );
      for (std::map<int, std::pair <float,float> >::iterator it = hidden_rated_itemIDS.begin(); it != hidden_rated_itemIDS.end(); it++ )
      {
        (*coo_format_ratingsMtx_userID_host_new)[place_] = user;

        int user_itemID = it->first;
        if(user_itemID < 0 || user_itemID >= (int)ratings_cols){
          ABORT_IF_EQ(0, 0, "Item ID is out of bounds");
        }
        (*coo_format_ratingsMtx_itemID_host_new)[place_] = user_itemID;

        float new_rating = (float)0.0;
        if(std::abs(it->second.second) > 0.0001){
         new_rating = (it->second.first) / (it->second.second);
        }
        if (::isinf(new_rating) || ::isnan(new_rating)){
          ABORT_IF_EQ(0, 0, "new rating is bad");
        }
        (*coo_format_ratingsMtx_rating_host_new)[place_] = new_rating;
        place_ +=(long long int)1;            
      }
      hidden_rated_itemIDS.erase ( hidden_rated_itemIDS.begin(), hidden_rated_itemIDS.end() );
      // sort
      // thrust::sort_by_key(thrust::host, 
      //   (*coo_format_ratingsMtx_itemID_host_new) + first_place, 
      //   (*coo_format_ratingsMtx_itemID_host_new) + place_ , 
      //   (*coo_format_ratingsMtx_rating_host_new) + first_place);

      // quickSort_by_key<int,float>((*coo_format_ratingsMtx_itemID_host_new) + first_place, 0, place_ - 1, 
      //                  (*coo_format_ratingsMtx_rating_host_new) + first_place);

      if(Global_Debug || thread_debug ) {
        omp_set_lock(&printlock);
        //LOG("th_id : "<<th_id);
        LOG("user : "<<user);
        // LOG("first_place : "<<first_place);
        // LOG("last_place : "<<place_ - 1);
        LOG("num to sort : "<<place_ - first_place);

        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_compute_hidden_values run time so far: "<<readable_time(program_time));
        omp_unset_lock(&printlock);
      }
    }// end for user
  }// end parallel
  if(Global_Debug){
      int ran_ind = 0;
      //getRandIntsBetween(&ran_ind, 0, (int)ratings_rows - 2, 1);
      int first_place = ((*coo_format_ratingsMtx_userID_host_new)[ran_ind]);
      int last_place = ((*coo_format_ratingsMtx_userID_host_new)[ran_ind + 2]);
      LOG("random row index : "<<ran_ind);
      LOG("first coo index : "<<first_place);
      LOG("last coo index : "<<last_place - 1);
      LOG("number of entries to print : "<<last_place - first_place);
      save_host_array_to_file<int>((*coo_format_ratingsMtx_userID_host_new) + ran_ind, 3, "coo_format_ratingsMtx_userID_host_new");
      save_host_array_to_file<int>((*coo_format_ratingsMtx_itemID_host_new) + first_place, last_place - first_place, "coo_format_ratingsMtx_itemID_host_new");
      save_host_array_to_file<float>((*coo_format_ratingsMtx_rating_host_new) + first_place, last_place - first_place,  "coo_format_ratingsMtx_rating_host_new");
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_compute_hidden_values run time : "<<readable_time(program_time));
  return (long long int)new_coo_count;
}


/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
template<typename Dtype, typename Itype>
int partition (Dtype* x, int low_index, int high_index, Itype* indicies)
{
  bool debug = false;
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

template int partition<int, int>(int* x, int low_index, int high_index, int* indicies);
template int partition<float, int>(float* x, int low_index, int high_index, int* indicies);
template int partition<int, float>(int* x, int low_index, int high_index, float* indicies);


/* low  --> Starting index,  high  --> Ending index */
template<typename Dtype, typename Itype>
void quickSort_by_key(Dtype* x, int low_index, int high_index, Itype* indicies)
{
  bool debug = false;
  //ABORT_IF_NEQ(0, 1, "function not ready");
  if (low_index < high_index)
  {
    /* pi is partitioning index, arr[pi] is now
       at right place */
    int pi = partition<Dtype,Itype>(x, low_index, high_index, indicies);
    if(debug){
      LOG("pi : "<<pi);
      if(low_index < 0 || high_index > 1316){
        LOG("low_index : "<<low_index);
        LOG("high_index : "<<high_index);
        ABORT_IF_EQ(0, 0, "uh oh!");
      }
    }
    quickSort_by_key<Dtype,Itype>(x, low_index, pi - 1, indicies);  // Before pi
    quickSort_by_key<Dtype,Itype>(x, pi + 1, high_index, indicies); // After pi
  }
}

template void quickSort_by_key<int, int>(int* x, int low_index, int high_index, int* indicies);
template void quickSort_by_key<float, int>(float* x, int low_index, int high_index, int* indicies);
template void quickSort_by_key<int, float>(int* x, int low_index, int high_index, float* indicies);

template<typename Dtype>
void cpu_sort_csr_colums_kernel(long long int start, int num, 
                                const int *csr_format_ratingsMtx_userID_dev,
                                int* coo_format_ratingsMtx_itemID_dev,
                                Dtype* coo_format_ratingsMtx_rating_dev) 
{
  bool debug = false;

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = std::min(nProcessors, num);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(int i = th_id; i < num; i += nthreads){

      long long int j = (long long int)i + start;
      int first_place = (csr_format_ratingsMtx_userID_dev[j]);
      int last_place = (csr_format_ratingsMtx_userID_dev[j + (long long int)1] - 1);

      quickSort_by_key<int,Dtype>(coo_format_ratingsMtx_itemID_dev, first_place, last_place, coo_format_ratingsMtx_rating_dev);

      if(debug && th_id == 0){
        omp_set_lock(&printlock);
        LOG("i : "<<i);
        LOG("first_place : "<<first_place);
        LOG("last_place : "<<last_place);
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("run time so far : "<<readable_time(program_time)<<std::endl);  
        omp_unset_lock(&printlock);
      }    
    }
  }
}

template <>
void cpu_sort_csr_colums<float>(const long long int ratings_rows, 
                                const int *csr_format_ratingsMtx_userID_host,
                                int* coo_format_ratingsMtx_itemID_host,
                                float* coo_format_ratingsMtx_rating_host, 
                                long long int num_entries_,
                                std::string preprocessing_path)
{
  if(1) LOG("called cpu_sort_csr_colums");
  bool debug = true;

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

  long long int num_cpu_blocks = (ratings_rows + CUDA_NUM_THREADS_TEMP - (long long int)1) / CUDA_NUM_THREADS_TEMP;

  if(debug){
    LOG("CUDA_NUM_BLOCKS_TEMP : "<<CUDA_NUM_BLOCKS_TEMP);
    LOG("CUDA_NUM_THREADS_TEMP : "<<CUDA_NUM_THREADS_TEMP);
  }

  if ( 0 /*num_cpu_blocks > CUDA_NUM_BLOCKS_TEMP*/){
    long long int num_loops = (long long int)0;
    long long int num_entries = (long long int)(CUDA_NUM_BLOCKS_TEMP * CUDA_NUM_THREADS_TEMP);
    long long int spot = (long long int)0;
    if(too_big(num_entries) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

    while (num_cpu_blocks > CUDA_NUM_BLOCKS_TEMP){
      if(debug){
        LOG("num_cpu_blocks : "<<num_cpu_blocks);
        LOG("num_loops : "<<num_loops);
        LOG("spot : "<<spot);
        LOG("num_entries : "<<num_entries);

        int first_place = (csr_format_ratingsMtx_userID_host[spot]);
        int last_place = (csr_format_ratingsMtx_userID_host[spot + num_entries]); 
        LOG("first_place : "<<first_place);
        LOG("last_place : "<<last_place - 1);
        LOG("num_entries : "<<last_place - first_place);
        save_host_array_to_file<int>(csr_format_ratingsMtx_userID_host + spot, (int)num_entries + 1, preprocessing_path + "csr_format_ratingsMtx_userID_host");
        save_host_array_to_file<int>(coo_format_ratingsMtx_itemID_host + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_host");
        save_host_array_to_file<float>(coo_format_ratingsMtx_rating_host + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_host");
      }
      cpu_sort_csr_colums_kernel<float>(spot, (int)num_entries,
                                                                                        csr_format_ratingsMtx_userID_host,
                                                                                        coo_format_ratingsMtx_itemID_host,
                                                                                        coo_format_ratingsMtx_rating_host);
      
      num_cpu_blocks = num_cpu_blocks - (long long int)CUDA_NUM_BLOCKS_TEMP;
      num_loops += (long long int)1;
      spot = num_loops * num_entries;
      if(debug){
        int first_place = (csr_format_ratingsMtx_userID_host[spot]);
        int last_place = (csr_format_ratingsMtx_userID_host[spot + num_entries]);
        save_host_array_to_file<int>(csr_format_ratingsMtx_userID_host + spot, (int)num_entries + 1, preprocessing_path + "csr_format_ratingsMtx_userID_host");
        save_host_array_to_file<int>(coo_format_ratingsMtx_itemID_host + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_host");
        save_host_array_to_file<float>(coo_format_ratingsMtx_rating_host + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_host");

        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_sort_csr_colums run time loop "<<num_loops<<" : "<<readable_time(program_time)<<std::endl);              
      }
    }
    // spot is the number of entries done so far
    // total - (done) = left to go 
    cpu_sort_csr_colums_kernel<float>(spot, (int)(ratings_rows - spot),
                                                                            csr_format_ratingsMtx_userID_host,
                                                                            coo_format_ratingsMtx_itemID_host,
                                                                            coo_format_ratingsMtx_rating_host);

  }else{
    if(too_big(ratings_rows) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
    if(debug){
      LOG("num_cpu_blocks : "<<num_cpu_blocks);
    }
    cpu_sort_csr_colums_kernel<float>((long long int)0, (int)ratings_rows,
                                      csr_format_ratingsMtx_userID_host,
                                      coo_format_ratingsMtx_itemID_host,
                                      coo_format_ratingsMtx_rating_host);
  }
  if(1) LOG("finished call to cpu_sort_csr_colums") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(1) LOG("cpu_sort_csr_colums run time : "<<readable_time(program_time)<<std::endl);
}

void cpu_sort_csr_colums_test()
{

  int num_rows = 1;
  int num_entries = 1317;

  int csr_format_ratingsMtx_userID_host[num_rows + 1] = {0, num_entries};
  int coo_format_ratingsMtx_itemID_host[num_entries] = {0, 8, 18, 35, 55, 59, 86, 94, 103, 109, 110, 125, 149, 152, 157, 159, 160, 162, 164, 167, 172, 179, 184, 207, 229, 230, 234, 248, 257, 265, 279, 281, 287, 315, 316, 343, 348, 352, 355, 376, 379, 392, 404, 433, 441, 456, 470, 473, 474, 479, 493, 499, 526, 545, 550, 552, 554, 557, 585, 586, 589, 591, 598, 607, 609, 672, 677, 732, 735, 740, 744, 749, 761, 777, 779, 783, 785, 787, 809, 831, 840, 857, 860, 884, 898, 902, 903, 907, 909, 911, 912, 919, 921, 922, 939, 952, 967, 968, 990, 1006, 1009, 1010, 1012, 1013, 1015, 1017, 1018, 1026, 1027, 1029, 1030, 1031, 1034, 1072, 1083, 1085, 1087, 1100, 1125, 1126, 1134, 1147, 1174, 1177, 1196, 1198, 1202, 1203, 1205, 1206, 1208, 1209, 1211, 1212, 1217, 1219, 1220, 1222, 1223, 1224, 1227, 1232, 1233, 1236, 1240, 1241, 1244, 1246, 1247, 1249, 1251, 1252, 1253, 1254, 1255, 1256, 1259, 1262, 1264, 1269, 1271, 1273, 1274, 1275, 1281, 1282, 1283, 1284, 1287, 1289, 1296, 1300, 1302, 1306, 1319, 1338, 1339, 1344, 1346, 1355, 1358, 1370, 1371, 1372, 1374, 1375, 1379, 1384, 1387, 1393, 1395, 1406, 1407, 1428, 1484, 1494, 1498, 1516, 1526, 1537, 1543, 1550, 1551, 1555, 1561, 1572, 1582, 1586, 1590, 1591, 1605, 1607, 1609, 1616, 1624, 1638, 1644, 1652, 1675, 1680, 1681, 1701, 1703, 1706, 1720, 1730, 1731, 1747, 1783, 1821, 1830, 1857, 1861, 1866, 1880, 1881, 1910, 1916, 1917, 1918, 1920, 1922, 1951, 1952, 1953, 1960, 1964, 1967, 1981, 1999, 2000, 2001, 2002, 2004, 2009, 2010, 2011, 2013, 2015, 2016, 2018, 2027, 2032, 2033, 2037, 2042, 2049, 2050, 2052, 2053, 2057, 2075, 2077, 2087, 2090, 2091, 2094, 2104, 2113, 2114, 2115, 2133, 2136, 2138, 2143, 2152, 2159, 2160, 2161, 2185, 2211, 2244, 2267, 2272, 2299, 2310, 2316, 2323, 2328, 2337, 2353, 2354, 2365, 2372, 2380, 2381, 2401, 2403, 2405, 2406, 2411, 2412, 2413, 2419, 2420, 2421, 2428, 2448, 2449, 2454, 2466, 2469, 2470, 2487, 2501, 2527, 2528, 2548, 2550, 2565, 2570, 2615, 2632, 2639, 2641, 2642, 2653, 2659, 2661, 2693, 2698, 2700, 2705, 2709, 2716, 2719, 2725, 2727, 2734, 2787, 2788, 2790, 2794, 2796, 2797, 2806, 2807, 2809, 2857, 2866, 2870, 2875, 2878, 2879, 2904, 2914, 2915, 2923, 2947, 2948, 2950, 2952, 2984, 2985, 2986, 2990, 2992, 2996, 3017, 3021, 3032, 3038, 3043, 3051, 3061, 3069, 3073, 3074, 3086, 3090, 3104, 3107, 3113, 3133, 3146, 3189, 3195, 3197, 3199, 3242, 3252, 3253, 3256, 3272, 3274, 3299, 3347, 3362, 3363, 3364, 3395, 3396, 3399, 3408, 3420, 3434, 3438, 3439, 3447, 3470, 3480, 3507, 3526, 3549, 3550, 3577, 3592, 3622, 3623, 3634, 3637, 3638, 3653, 3670, 3675, 3680, 3685, 3698, 3702, 3703, 3704, 3705, 3726, 3735, 3739, 3741, 3744, 3762, 3770, 3784, 3792, 3801, 3806, 3825, 3827, 3831, 3835, 3876, 3878, 3916, 3929, 3945, 3947, 3948, 3955, 3958, 3971, 3976, 3980, 3983, 4014, 4021, 4033, 4039, 4080, 4084, 4103, 4123, 4131, 4197, 4209, 4213, 4214, 4222, 4261, 4269, 4274, 4309, 4326, 4342, 4343, 4366, 4368, 4382, 4387, 4395, 4396, 4404, 4436, 4437, 4439, 4443, 4530, 4532, 4541, 4543, 4545, 4551, 4552, 4557, 4579, 4586, 4590, 4620, 4635, 4637, 4642, 4657, 4672, 4677, 4680, 4700, 4717, 4733, 4734, 4811, 4826, 4847, 4854, 4859, 4864, 4875, 4885, 4886, 4901, 4908, 4928, 4962, 4965, 4967, 4968, 4972, 4973, 4978, 4994, 5026, 5037, 5040, 5042, 5049, 5059, 5061, 5071, 5085, 5088, 5092, 5093, 5099, 5104, 5155, 5180, 5181, 5192, 5217, 5218, 5245, 5246, 5253, 5280, 5290, 5293, 5307, 5308, 5312, 5348, 5377, 5393, 5410, 5417, 5418, 5426, 5432, 5437, 5440, 5444, 5451, 5458, 5462, 5480, 5488, 5497, 5501, 5506, 5555, 5567, 5608, 5617, 5629, 5648, 5689, 5704, 5711, 5780, 5781, 5783, 5809, 5832, 5852, 5949, 5961, 5963, 5970, 5973, 5994, 6015, 6053, 6061, 6077, 6098, 6103, 6137, 6139, 6141, 6156, 6173, 6228, 6249, 6263, 6272, 6273, 6282, 6300, 6322, 6349, 6364, 6376, 6382, 6439, 6502, 6529, 6533, 6536, 6540, 6563, 6600, 6663, 6668, 6702, 6720, 6726, 6730, 6733, 6747, 6750, 6765, 6784, 6799, 6856, 6873, 6906, 6933, 6951, 6966, 6986, 6995, 7003, 7012, 7021, 7062, 7089, 7098, 7114, 7115, 7122, 7146, 7190, 7230, 7253, 7256, 7307, 7309, 7312, 7321, 7359, 7360, 7361, 7396, 7418, 7447, 7457, 7563, 7568, 7586, 7697, 7702, 7757, 7765, 7791, 7801, 7819, 7837, 7882, 7886, 7921, 7923, 7924, 7925, 7981, 8015, 8018, 8041, 8124, 8238, 8268, 8359, 8360, 8370, 8386, 8490, 8520, 8530, 8591, 8639, 8643, 8665, 8669, 8672, 8692, 8741, 8750, 8762, 8765, 8805, 8809, 8814, 8816, 8830, 8860, 8873, 8884, 8888, 8893, 8956, 8971, 8975, 8982, 8983, 8984, 8987, 25748, 25749, 25759, 25793, 25797, 25804, 25824, 25889, 25941, 26073, 26121, 26171, 26286, 26337, 26429, 26506, 26584, 26661, 26709, 26766, 26775, 26834, 26864, 26945, 27092, 27104, 27191, 27316, 27433, 27659, 27667, 27727, 27771, 27800, 27838, 30792, 30809, 30893, 31037, 31269, 31426, 31430, 31657, 31749, 31792, 31877, 31949, 32010, 32229, 32360, 32550, 32586, 32934, 33492, 33678, 33793, 33833, 33939, 34047, 34149, 34318, 34658, 35720, 37728, 37948, 40814, 41563, 41565, 41568, 41819, 41879, 42542, 42737, 43674, 43918, 44154, 44902, 44971, 45080, 45446, 45498, 45721, 48393, 48515, 48773, 49081, 49662, 49751, 49768, 49816, 50357, 50797, 50871, 51076, 52107, 52282, 52547, 52580, 52703, 52721, 52999, 53372, 53463, 54000, 54009, 54048, 54825, 54832, 55342, 55468, 56547, 56873, 58558, 58609, 58769, 58880, 59314, 59614, 60068, 60283, 60355, 61239, 61933, 63780, 65467, 65681, 66050, 67297, 68156, 68532, 68589, 68953, 69301, 69608, 69752, 69843, 70750, 71026, 71279, 71932, 71985, 72275, 72303, 72335, 72652, 72924, 72935, 73161, 73358, 73474, 74160, 74316, 74477, 74856, 75976, 75978, 76021, 76694, 76828, 77537, 77775, 77807, 77943, 78024, 78412, 78695, 78859, 79105, 79423, 79635, 79766, 80205, 80423, 80567, 80679, 80718, 80824, 80949, 81392, 81833, 82122, 82303, 82752, 83050, 83670, 83772, 84831, 84988, 86203, 86307, 86398, 86714, 86755, 87030, 87050, 87357, 88098, 88124, 88165, 88328, 89013, 89515, 89548, 89550, 89669, 89766, 89796, 89832, 89871, 90085, 90379, 90534, 90650, 90774, 90776, 91066, 91153, 91418, 91424, 91557, 91559, 91609, 91691, 91708, 91767, 91895, 92084, 92470, 92675, 93036, 93195, 93329, 93392, 93483, 93655, 93784, 94432, 94834, 95587, 95694, 95764, 96293, 96761, 96844, 97767, 97818, 97909, 97947, 97970, 98594, 98765, 98804, 99011, 99053, 99084, 99269, 99272, 99924, 100069, 100495, 100945, 101225, 101286, 101445, 101733, 101824, 101943, 101951, 102011, 102089, 102398, 102424, 102589, 102597, 103102, 103379, 103436, 103518, 103562, 103636, 103662, 103670, 103744, 103812, 104090, 104098, 104639, 104812, 105307, 105480, 105812, 106140, 106396, 106526, 106701, 106867, 107182, 107294, 107381, 107483, 108011, 108047, 108075, 108523, 109031, 109054, 109061, 109105, 109152, 109324, 109572, 109770, 110045, 110115, 110176, 110178, 110228, 110319, 110351, 110534, 110556, 110817, 110896, 111232, 111289, 111311, 112061, 112331, 112394, 112484, 112600, 112930, 112958, 113015, 113219, 113231, 113357, 113605, 113848, 113905, 114279, 114281, 114419, 114576, 115163, 115290, 115378, 115621, 115928, 116856, 116926, 116932, 116988, 117361, 117569, 117581, 117929, 118176, 118197, 118707, 118773, 118853, 118859, 119146, 119311, 119423, 119431, 119795, 120431, 120854, 120933, 121321, 121323, 122287, 123406, 124301, 124536, 124561, 125530, 127629, 128168, 128172, 128444, 128519, 128633, 128861, 129067, 130070, 130348, 130473, 130957, 130983, 131010, 1, 28, 31, 46, 49, 111, 150, 222, 252, 259, 292, 295, 317, 336, 366, 540, 588, 592, 652, 918, 923, 1008, 1035, 1078, 1079, 1088, 1089, 1096, 1135, 1192, 1195, 1197, 1199, 1200, 1207, 1213, 1214, 1216, 1218, 1221, 1239, 1242, 1245, 1248, 1257, 1258, 1260, 1261, 1265, 1277, 1290, 1303, 1320, 1332, 1347, 1349, 1357, 1369, 1373, 1386, 1524, 1583, 1749, 1847, 1919, 1966, 1993, 1996, 2020, 2099, 2117, 2137, 2139, 2142, 2172, 2173, 2192, 2193, 2252, 2287, 2290, 2541, 2627, 2643, 2647, 2663, 2682, 2691, 2715, 2760, 2761, 2803, 2871, 2917, 2943, 2946, 2958, 2967, 2999, 3029, 3036, 3080, 3152, 3264, 3437, 3475, 3478, 3488, 3498, 3888, 3931, 3995, 3996, 4010, 4026, 4104, 4127, 4132, 4225, 4305, 4445, 4466, 4570, 4719, 4753, 4877, 4895, 4910, 4914, 4940, 4979, 4992, 5025, 5038, 5039, 5145, 5170, 5539, 5678, 5796, 5815, 5897, 5951, 5998, 6092, 6241, 6332, 6501, 6538, 6753, 6754, 6773, 6806, 6833, 6887, 7000, 7044, 7045, 7152, 7163, 7246, 7386, 7388, 7437, 7448, 7453, 7481, 7756, 8367, 8481, 8506, 8635, 8689, 8960, 31695};
  float coo_format_ratingsMtx_rating_host[num_entries] = {0.177706, -0.637103, -0.637103, -0.637103, -1.17237, -0.637103, -0.637103, -1.146, -0.637103, -0.137473, -0.326808, -1.11048, -0.155476, -0.440858, 0.674579, -1.94879, 0.674579, -0.637103, -1.94879, -1.94879, -1.94879, 0.674579, -0.637103, -0.478419, 0.674579, -0.637103, -0.637103, 0.674579, -1.16705, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.462651, -0.0602304, -0.861401, -1.94879, -0.637103, 0.674579, -0.637103, -1.94879, 0.0303579, 0.674579, 0.674579, -0.274926, -0.486803, -1.94879, -0.637103, 0.0359569, -1.00097, 0.674579, 0.674579, 0.674579, -0.637103, -0.469645, 0.674579, 0.674579, 0.67458, -0.637103, -0.0297566, 0.674579, -0.637103, 0.674579, -1.94879, -0.528783, -0.637103, -1.28711, -0.293823, -0.637103, -0.125956, -0.644179, -0.637103, -1.19001, -0.670619, -0.637103, 0.674579, 3.29794, 0.0192334, -0.637103, -0.637103, -0.637103, 0.0426224, -0.251982, -0.274276, -0.0199693, -0.437627, -0.177032, -0.637103, -0.557999, -0.160624, 0.674579, -0.278596, -0.381082, -1.07516, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, -0.637103, -0.164661, 0.674579, -0.831809, -0.710015, -1.3044, 0.431595, -0.302686, -0.637103, -0.910546, -0.316508, -0.467197, -1.94879, 1.21624, -1.0206, -0.119223, -0.398339, -0.637103, -0.0200866, -1.94879, -1.94879, -0.193671, -0.247501, 0.674579, -0.117064, -0.637103, -0.4006, 0.0333618, -1.19973, -0.616105, -0.958377, -0.524093, -0.256905, -0.637103, -0.864431, -0.637103, 0.219958, 0.674579, -0.404372, -0.0602766, -0.272447, 0.473058, -0.673156, -0.426598, 0.674579, -0.844806, 0.674579, -1.94879, -0.637103, -0.623995, -0.280731, 0.674579, 0.67458, -0.215643, -0.637103, -0.0329604, 0.674579, 0.674579, -0.82628, -0.242388, -0.0664514, 0.362175, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, -0.637103, -0.0829305, -0.0604934, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.67458, -1.94879, -0.428624, -0.637103, -1.94879, -0.637103, -0.415069, -1.94879, -1.01571, -0.435386, -0.637103, -0.637103, -1.94879, -1.94879, -0.120456, 0.00949191, -0.637103, 0.674579, 0.674579, -0.148218, 0.674579, -0.637103, 0.170818, -0.637103, 0.182678, -1.94879, 0.674579, -0.637103, 0.191381, 0.153274, -0.637103, -0.637103, -0.57337, -0.637103, 0.674579, -0.637103, -1.06701, -0.328279, -0.637103, -0.535967, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, 0.307902, 0.674579, -0.102854, -0.254629, -0.0436821, -1.12417, -1.94879, 0.255385, 0.674579, -0.830422, -0.521523, -1.94879, -0.637103, -0.637103, -0.637103, -0.778003, 0.791021, 0.674579, 0.195782, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.406215, -1.94879, 0.143165, 0.674579, -0.0690882, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.366569, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.0551844, 0.67458, -1.27046, 0.129329, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, 0.0426347, -1.94879, -0.637103, -1.94879, -0.529162, -1.23139, -1.94879, -1.94879, -1.94879, -1.94879, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.754389, -0.637103, -0.637103, -0.701621, 0.374468, 0.674579, 0.674579, -0.637103, 2.04119, -0.0594538, 0.674579, 0.372803, -0.0742494, 0.674579, -0.637103, 0.651898, -0.244879, -0.637103, 0.674579, -0.916603, -0.862378, -0.917509, -1.08289, -0.172581, -0.637103, 0.674579, -0.315244, -0.637103, -0.637103, -0.0161588, -0.637103, -0.637103, 0.674579, 0.674579, -0.301266, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, 0.0358773, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -1.37229, 0.674579, -1.94879, -0.637103, -0.637103, -0.454708, -0.806231, -0.345035, 0.674579, -0.651668, -0.0316486, -0.637103, -0.637103, -0.253871, -0.637103, -1.16859, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -1.23996, -0.637103, 0.674579, 0.148997, -1.94879, 0.277109, 0.674579, -0.103048, -0.637103, -0.637103, -0.637103, 0.202765, -0.272621, -0.0360244, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.67458, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.939542, -1.2072, -0.637103, 0.251636, 0.674579, 0.395388, 0.674579, 0.674579, 0.674579, 2.1697, 0.205222, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.69779, 0.674579, 0.674579, 0.321634, 0.674579, 0.67458, -0.637103, 0.674579, -1.94879, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -1.52034, 0.946077, -1.94879, 0.674579, -0.222471, 0.674579, 0.674579, 0.674579, -1.41221, 0.674579, -0.637103, -1.94879, 0.674579, -0.637103, -0.00356942, 0.674579, 0.674579, -0.637103, -0.272464, 0.223686, -0.637103, -1.41276, -0.637103, 0.674579, -0.637103, -1.94879, 0.674579, -0.637103, -1.94879, -1.29488, -0.637103, 0.674579, 0.674579, -1.94879, 0.674579, -1.94879, 0.20477, -1.12233, 0.195866, -0.383621, 0.674579, 0.674579, 0.178412, -0.637103, 0.674579, -1.6561, -0.786677, -1.94879, -0.637103, 0.338086, -1.94879, -1.94879, -1.94879, -1.94879, -0.637103, -1.94879, -1.50875, 0.00240074, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, -1.94879, -0.189671, 0.34467, 0.674579, -1.31835, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.201652, -1.94879, -0.637103, -0.637103, -0.800606, -1.94879, 0.674579, 0.674579, 0.543248, 0.674579, -0.637103, -0.637103, -1.94879, 0.674579, -1.94879, -0.637103, -1.94879, -0.0160151, -1.94879, 0.674579, -0.182755, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.75268, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.318759, -0.637103, 0.674579, 0.67458, -0.637103, -1.36439, -0.0144304, 0.674579, -0.637103, -0.225503, 1.15468, 0.503029, -0.637103, -0.637103, 2.51199, -1.94879, -0.637103, -1.94879, -1.24546, -0.637103, 0.782777, -0.637103, -0.265776, 0.674579, -0.637103, -0.0997543, -1.94879, -0.637103, 0.345317, -0.637103, -0.637103, -0.637103, 0.448523, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.67458, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.210605, 0.674579, -0.637103, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.137635, -0.637103, 0.674579, -0.637103, 0.674579, 3.29794, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 1.39234, -1.94879, -0.637103, -0.198552, 0.674579, -0.547232, 0.674579, 0.330088, -0.451396, -0.637103, -1.94879, -1.39169, -0.637103, -1.94879, 0.67458, -0.348339, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -1.94879, -1.11002, 0.393678, 0.674579, 0.418379, -0.637103, -0.637103, 1.03817, -0.768724, 0.674579, 2.35354, -0.637103, 0.67458, -0.0504983, -0.637103, 0.0784256, 3.29794, 0.67458, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -1.94879, -0.637103, 0.674579, -0.637103, -0.0754003, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 3.29794, -1.05553, 0.674579, -0.637103, -0.637103, 0.674579, -0.460676, -0.637103, -0.637103, 3.29794, -0.637103, 3.29795, -0.637103, 0.674579, -1.94879, -1.43924, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 3.29794, 3.29794, -0.637103, 0.938161, 0.674579, 0.674579, -1.94879, -1.94879, -0.163584, -1.94879, -0.637103, 0.674579, 0.674579, 0.0920067, 3.29794, 0.674579, 0.674579, -0.637103, 3.29794, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.616435, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, 3.29794, -0.637103, -1.94879, -1.94879, 3.29794, 3.29794, -0.637103, 0.674579, 0.674579, 3.29794, 3.29794, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.17764, -0.637103, -0.637103, 0.674579, -0.637103, 0.074912, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.316765, -0.637103, 0.979924, 0.674579, 1.88818, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.764716, 0.674579, 0.674579, 1.98626, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, -1.94879, 0.674579, 0.674579, 0.674579, 0.674579, -0.271676, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 1.98626, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, -1.94879, 2.54423, -0.637103, 0.674579, -0.637103, 1.02233, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, -1.94879, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.67458, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.67458, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.67458, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, 1.98626, 1.98626, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, -1.94879, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, -1.94879, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, -0.637103, -0.637103, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -0.637103, -1.94879, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, 0.674579, 0.674579, -1.94879, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -1.94879, -0.637103, -0.637103, 3.29794, 0.674579, 0.674579, -1.94879, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 3.29794, -0.637103, 0.674579, -0.637103, 0.674579, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -0.637103, -0.637103, -1.94879, -0.637103, -0.637103, 0.674579, 3.29794, -0.637103, -0.637103, -0.637103, 0.674579, 0.674579, -0.637103, 0.674579, -1.94879, 0.674579, 0.674579, -0.637103, 3.29794, 1.98626, -0.637103, 0.674579, 0.674579};

  print_host_array(csr_format_ratingsMtx_userID_host, num_rows + 1, "csr_format_ratingsMtx_userID_host");
  print_host_array(coo_format_ratingsMtx_itemID_host, num_entries, "coo_format_ratingsMtx_itemID_host");
  print_host_array(coo_format_ratingsMtx_rating_host, num_entries, "coo_format_ratingsMtx_rating_host");

  cpu_sort_csr_colums<float>(num_rows, csr_format_ratingsMtx_userID_host, coo_format_ratingsMtx_itemID_host, coo_format_ratingsMtx_rating_host);

  print_host_array(csr_format_ratingsMtx_userID_host, num_rows + 1, "csr_format_ratingsMtx_userID_host");
  print_host_array(coo_format_ratingsMtx_itemID_host, num_entries, "coo_format_ratingsMtx_itemID_host");
  print_host_array(coo_format_ratingsMtx_rating_host, num_entries, "coo_format_ratingsMtx_rating_host");
}


/* low  --> Starting index,  high  --> Ending index */
template<typename Dtype, typename Itype>
void naiveSort_by_key(Dtype* x, int total, int num_sorted, Itype* indicies, Itype* indicies_sorted, Dtype* x_sorted)
{
  bool Debug = false;
  if (total < num_sorted){
    ABORT_IF_NEQ(0, 1, "total < num_sorted");
  }

    // pivot (Element to be placed at right position)
    Dtype temp = 0.0;
    Itype temp_ = 0;
    int count = 0;  // Index of smaller element

    for (int j = total - 1; j >= total - num_sorted; j--)
    {
      if(Debug){
        LOG("j : "<<j);
      }
      bool first_max_found = false;
      Dtype max_so_far = (Dtype)0.0;
      int max_so_far_index = 0;
      for (int i = 0; i < total - count; i++){
          if (!first_max_found){
            max_so_far = x[i];
            max_so_far_index = i;
            first_max_found = true;
          }else{
            if(x[i] > max_so_far){
              max_so_far = x[i];
              max_so_far_index = i;
            }
          }  
      }
      temp = x[j];
      if(x_sorted){
        x_sorted[num_sorted - 1 - count] = max_so_far;
      }else{
        x[j] = max_so_far;
      }
      x[max_so_far_index] = temp;

      temp_ = indicies[j];
      if(indicies_sorted){
        indicies_sorted[num_sorted - 1 - count] = indicies[max_so_far_index];
      }else{
        indicies[j] = indicies[max_so_far_index];
      }
      indicies[max_so_far_index] = temp_;

      count++;
    }
    if(Debug) LOG("count : "<<count)
}

template void naiveSort_by_key<int, int>(int* x, int total, int num_sorted, int* indicies, int* indicies_sorted, int* x_sorted);
template void naiveSort_by_key<float, int>(float* x, int total, int num_sorted, int* indicies, int* indicies_sorted, float* x_sorted);
template void naiveSort_by_key<int, float>(int* x, int total, int num_sorted, float* indicies, float* indicies_sorted, int* x_sorted);

template<typename Dtype, typename Itype>
void cpu_sort_index_by_max(const long long int rows, const long long int cols,  Dtype* x, Itype* indicies)
{
  // col major ordering 
  // sort each col
  bool print = true;
  struct timeval program_start, program_end;
  if(print) LOG("called cpu_sort_index_by_max") ;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, cols);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < cols; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < rows; i+=(long long int)1){
      //thrust::sort_by_key sorts indicies by x smallest to x largest
      //thrust::sort_by_key(thrust::host, x + i * rows, x + (i + 1) * rows , indicies + i * rows);
      quickSort_by_key<Dtype, Itype>(x + i * rows, 0, (int)rows - 1, indicies + i * rows);
    }
  }

  if(print) LOG("finished call to cpu_sort_index_by_max") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG( "cpu_sort_index_by_max run time : "  << readable_time(program_time) );
}


template void cpu_sort_index_by_max<int, int>(const long long int rows, const long long int cols,  int* x, int* indicies);
template void cpu_sort_index_by_max<float, int>(const long long int rows, const long long int cols,  float* x, int* indicies);
template void cpu_sort_index_by_max<int, float>(const long long int rows, const long long int cols,  int* x, float* indicies);

template<typename Dtype>
void cpu_sort_index_by_max(const long long int dimension,  Dtype* x, int* indicies, int top_N, Dtype* x_sorted)
{
  /*
    "sort" the rows (not including the diagonal) of a symetric 
    matrix when storing only the entries below the diagonal 
    in column major order. 

    The indicies vector stores the first 
    top_N indicies of the entries in x for a given row for each row. 
    Thus indicies is top_N * dimension in column major order.
  */
  bool print = true;
  bool debug = false;
  double avg_time = 0.0;
  struct timeval program_start, program_end;
  if(print) LOG("called cpu_sort_index_by_max") ;

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    if(!debug)
      nthreads = std::min(nProcessors, (int)dimension);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
    omp_lock_t thrustlock;
    omp_init_lock(&thrustlock);
    if(debug){
      LOG("max threads: "<<nProcessors)
      LOG("number of threads: "<<nthreads);
    }
  #endif

  Dtype* temp_x  = (Dtype *)malloc((dimension - (long long int)1) * ((long long int)nthreads) * SIZE_OF(Dtype));
  int* temp_indicies  = (int *)malloc((dimension - (long long int)1) * ((long long int)nthreads) * SIZE_OF(int));
  checkErrors(temp_x);
  checkErrors(temp_indicies);
  //save_host_mtx_to_file(temp_x, (dimension - 1), nthreads, "temp_x");

  double program_time;
  gettimeofday(&program_start, NULL);
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < dimension; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < dimension; i+=(long long int)1){

      long long int num_below_diag = (long long int)0;
      long long int left_off = (long long int)0;
      long long int num_in_col = dimension - (long long int)1;

      for(long long int j = (long long int)0; j < i; j+=(long long int)1){
        left_off = num_below_diag + i - (dimension - num_in_col);

        temp_x[((long long int)th_id) * (dimension - (long long int)1) + j] = x[left_off];
        temp_indicies[((long long int)th_id) * (dimension - (long long int)1) + j] = (int)j;

        num_below_diag += num_in_col;
        num_in_col -= (long long int)(1);
      }
      left_off = num_below_diag + (i + (long long int)(1)) - (dimension - num_in_col);
      for(long long int j = i + (long long int)1; j < dimension; j+=(long long int)1){
        temp_x[((long long int)th_id) * (dimension - (long long int)1) + j - 1] = x[left_off];
        temp_indicies[((long long int)th_id) * (dimension - (long long int)1) + j - 1] = (int)j;
        left_off += (long long int)(1);

      }

      if(debug && th_id==0) {
        omp_set_lock(&printlock);
        LOG("th_id : "<<th_id);
        LOG("i : "<<i);
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_sort_index_by_max average loop run time so far : "<<readable_time(program_time * (double)nthreads/ (double)i));  
        //print_host_array((temp_x + th_id * (dimension - 1)), ((int)dimension - 1), ("temp_x"));
        //print_host_array((temp_indicies + th_id * (dimension - 1)), ((int)dimension - 1), ("temp_indicies"));
        save_host_array_to_file<Dtype>(temp_x + th_id * (dimension - 1), dimension - 1, "temp_x_before");
        save_host_array_to_file<int>(temp_indicies + th_id * (dimension - 1), dimension - 1, "temp_indicies_before");
        omp_unset_lock(&printlock);
      }
      //LOG("Hello from thread "<<th_id) ;
      //thrust::sort_by_key sorts temp_indicies by temp_x smallest to temp_x largest
      // omp_set_lock(&thrustlock);
      // thrust::sort_by_key(thrust::host, 
      //   temp_x + ((long long int)th_id * (dimension - (long long int)1)), 
      //   temp_x + ((long long int)(th_id + 1) * (dimension - (long long int)1)) , 
      //   temp_indicies + ((long long int)th_id * (dimension - (long long int)1)));
      // omp_unset_lock(&thrustlock);
      // quickSort_by_key<Dtype, int>(temp_x + ((long long int)th_id * (dimension - (long long int)1)), 0, (int)dimension - 2, 
      //                              temp_indicies + ((long long int)th_id * (dimension - (long long int)1)) );
      if(x_sorted){
        naiveSort_by_key<Dtype, int>(temp_x + ((long long int)th_id * (dimension - (long long int)1)), (int)dimension - 1, top_N, 
                                     temp_indicies + ((long long int)th_id * (dimension - (long long int)1)),
                                     indicies + (i * (long long int)top_N), x_sorted + (i * (long long int)top_N));
      }else{
        naiveSort_by_key<Dtype, int>(temp_x + ((long long int)th_id * (dimension - (long long int)1)), (int)dimension - 1, top_N, 
                                     temp_indicies + ((long long int)th_id * (dimension - (long long int)1)),
                                     indicies + (i * (long long int)top_N));        
      }

      // host_copy(top_N, 
      //   temp_indicies + ((long long int)(th_id + 1) * (dimension - (long long int)1) - (long long int)top_N), 
      //   indicies + (i * (long long int)top_N));
      // if(x_sorted){
      //   host_copy(top_N, 
      //   temp_x + ((long long int)(th_id + 1) * (dimension - (long long int)1) - (long long int)top_N), 
      //   x_sorted + (i * (long long int)top_N));
      // }
      if(debug && th_id==0) {
        omp_set_lock(&printlock);
        //LOG("th_id : "<<th_id);
        LOG("i : "<<i);
        // print_host_array((temp_x + (long long int)th_id * (dimension - (long long int)1)), ((int)dimension - 1), ("temp_x"));
        // print_host_array((temp_indicies + (long long int)th_id * (dimension - (long long int)1)), ((int)dimension - 1), ("temp_indicies"));
        // print_host_array((indicies + i * (long long int)top_N), (top_N), ("indicies"));

        //LOG("th_id * (dimension - 1) : "<<th_id * (dimension - 1));
        save_host_array_to_file<Dtype>(temp_x + th_id * (dimension - 1), dimension - 1, "temp_x_after");
        save_host_array_to_file<int>(temp_indicies + th_id * (dimension - 1), dimension - 1, "temp_indicies_after");
        if(x_sorted){
          save_host_array_to_file<Dtype>((x_sorted + i * (long long int)top_N), top_N, "x_sorted");
        }
        save_host_array_to_file<int>((indicies + i * (long long int)top_N), top_N, "indicies");

        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_sort_index_by_max average loop run time so far : "<<readable_time(program_time * (double)nthreads/ (double)i));  
        ABORT_IF_NEQ(0,1,"returning");  
        omp_unset_lock(&printlock);  
      }
    }
  }

  if(debug) LOG("Hello") ;
  free(temp_x);
  free(temp_indicies);

  if(0) LOG("finished call to cpu_sort_index_by_max") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_sort_index_by_max run time : "<<readable_time(program_time));
}


template void cpu_sort_index_by_max<int>(const long long int dimension,  int* x, int* indicies, int top_N,  int* x_sorted);
template void cpu_sort_index_by_max<float>(const long long int dimension,  float* x, int* indicies, int top_N,  float* x_sorted);



void cpu_count_appearances(const int top_N, const long long int dimension,
  int* count, const int* indicies)
{
  /*
    Count the appearances of each of the dimension indicies 
    in the columns of the vector called indicies which is 
    top_N * dimension in memory in column major ordering
  */
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, dimension);
    omp_set_num_threads(nthreads);
    omp_lock_t *locks = (omp_lock_t *)malloc(dimension * SIZE_OF(omp_lock_t));
    checkErrors(locks);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      omp_init_lock(locks + j);
      omp_unset_lock(locks + j);
    }
  }
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      for(long long int i = (long long int)0; i < (long long int)top_N; i+=(long long int)1){
        int temp = indicies[i + (long long int)top_N * j];
        omp_set_lock(locks + temp);
        count[temp] += 1;
        omp_unset_lock(locks + temp);
      }
    }
  }

  #ifdef _OPENMP
    free(locks);
  #endif

  if(0) LOG("finished call to cpu_count_appearances") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_count_appearances run time : "<<readable_time(program_time));
}




void cpu_rank_appearances(const int top_N, const long long int dimension,
  float* rank, const int* indicies)
{
  /*
    Count the appearances of each of the dimension indicies 
    in the columns of the vector called indicies which is 
    top_N * dimension in memory in column major ordering
  */
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, dimension);
    omp_set_num_threads(nthreads);
    omp_lock_t *locks = (omp_lock_t *)malloc(dimension * SIZE_OF(omp_lock_t));
    checkErrors(locks);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      omp_init_lock(locks + j);
      omp_unset_lock(locks + j);
    }
  }
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      for(long long int i = (long long int)0; i < (long long int)top_N; i+=(long long int)1){
        int temp = indicies[i + (long long int)top_N * j];
        omp_set_lock(locks + temp);
        rank[temp] += (float)(1.0 / ((float)(top_N - i)));
        omp_unset_lock(locks + temp);
      }
    }
  }

  #ifdef _OPENMP
    free(locks);
  #endif

  if(0) LOG("finished call to cpu_rank_appearances") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_rank_appearances run time : "<<readable_time(program_time));
}




void cpu_mark_CU_users(const int ratings_rows_CU, const int ratings_rows, const int* x, int* y )
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, ratings_rows_CU);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < ratings_rows_CU; j += (long long int)nthreads){
    //for(long long int j = (long long int)(ratings_rows - 1); j > (long long int)(ratings_rows - 1 - ratings_rows_CU); j-=(long long int)1){
      y[x[ratings_rows - 1 - j]] = 0;
    }
  }

  if(0) LOG("finished call to gpu_orthogonal_decomp") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_mark_CU_users run time : "<<readable_time(program_time));

}

long long int from_below_diag_to_whole(long long int below_diag_index, long long int dimension){
  bool debug = false;
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) return (long long int)(-1);

  long long int num_so_far = (long long int)0;
  int col = 0;
  long long int num_in_col = (long long int)(dimension - 1);
  while(num_so_far < below_diag_index + (long long int)(1)){
    if(debug){
      LOG("num_so_far : "<<num_so_far);
      LOG("num_in_col : "<<num_in_col);
      LOG("col : "<<col);
      LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
    }
    
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

long long int num_below_diag_(long long int dimension){
  return ( ( dimension * (dimension - (long long int)1) ) / ( (long long int)2 ) );
}

long long int from_below_diag_to_whole_faster(long long int below_diag_index, long long int dimension)
{
  bool debug = false;
  const long long int num_below_diag = num_below_diag_(dimension);
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)){
    if(debug){
      LOG("num_below_diag : "<<num_below_diag);
      LOG("below_diag_index : "<<below_diag_index);
    }
    return (long long int)(-1);
  }

  long long int inverted_number = num_below_diag - below_diag_index;
  float n_float = ((float)1.0 + sqrt((float)1.0 + (float)8.0 * (float)inverted_number)) / (float)2.0;
  long long int n = (long long int)round(n_float);
  long long int row = (long long int)(-1);
  long long int col = (long long int)(0);

  long long int one_less = num_below_diag_(n - (long long int)1); // triangle has n - 2 columns
  long long int on_it = num_below_diag_(n);                       // triangle has n - 1 columns
  long long int one_more = num_below_diag_(n + (long long int)1); // triangle has n columns
  if(debug){
    LOG("below_diag_index : "<<below_diag_index);
    LOG("dimension : "<<dimension);
    LOG("inverted_number : "<<inverted_number);
    LOG("n_float : "<<n_float);
    LOG("n : "<<n);
    LOG("((n - 1) * (n - 2)) / 2 : "<<one_less);
    LOG("(n * (n - 1)) / 2 : "<<on_it);
    LOG("(n * (n + 1)) / 2 : "<<one_more);
  }


  if(one_less < inverted_number && inverted_number < on_it){
    if(debug) LOG("one_less < inverted_number && inverted_number < on_it");
    col = dimension - n;
    row = dimension -  (inverted_number - one_less);
  }else if(inverted_number == on_it){
    if(debug) LOG("inverted_number == on_it");
    col = dimension - n;
    row = col + (long long int)1;
  }else if(on_it < inverted_number && inverted_number < one_more){
    if(debug) LOG("on_it < inverted_number && inverted_number < one_more");
    col = dimension - n - (long long int)1;
    row = dimension - (inverted_number - on_it);
  }else if(inverted_number == one_more){
    if(debug) LOG("inverted_number == one_more");
    //return (long long int)(-7);
    col = dimension - n - (long long int)1;
    row = col + (long long int)1;
  }else if(inverted_number == one_less){
    if(debug) LOG("inverted_number == one_less");
    //return (long long int)(-8);
    col = dimension - n + (long long int)1;
    row = col + (long long int)1;
  } else {
    return (long long int)(-2);
    // col = dimension - n + (long long int)1;
    // row = dimension - (inverted_number - (n * (n - (long long int)1)) / (long long int)2);
  }
  
  if( row == col){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-3); 
  }
  if( col < (long long int)0 || col >= dimension - (long long int)1){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-4);
  }
  if( row < (long long int)1 || row >= dimension){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-5);
  }
  long long int return_val = row + dimension * col;
  if( return_val < (long long int)0 || return_val >= (dimension * dimension)){
    return (long long int)(-6);
  }
  return return_val;
}

void from_below_diag_to_whole_test(){
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  LOG("from_below_diag_to_whole_test called.");

  const long long int dimension = (long long int)131262; //138493
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, num_below_diag);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
  #endif
  LOG("nthreads : "<<nthreads);
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < num_below_diag; i += (long long int)nthreads){
      
      long long int whole_faster = from_below_diag_to_whole_faster(i, dimension);
      if(whole_faster < (long long int)0){
        omp_set_lock(&printlock);
        LOG("th_id : "<<th_id);
        LOG("below_diag_indicies "<<i<<" maps to -> whole_indicies "   << whole_faster );
        omp_unset_lock(&printlock);        
      }
      // long long int whole_slow_way = from_below_diag_to_whole(i, dimension);
      // if(whole_slow_way != whole_faster){
      //   omp_set_lock(&printlock);
      //   LOG("th_id : "<<th_id);
      //   LOG("below_diag_indicies "<<i<<" maps to -> whole_indicies "   << whole_slow_way);
      //   LOG("below_diag_indicies "<<i<<" maps to -> whole_indicies "   << whole_faster );
      //   omp_unset_lock(&printlock);        
      // }
      // long long int below_after = from_whole_to_below_diag(whole_faster, dimension);
      // if(below_after != i){
      //   omp_set_lock(&printlock);
      //   LOG("th_id : "<<th_id);
      //   LOG("whole_indicies "<<whole_faster<<" maps to -> below_diag_indicies "   <<below_after );
      //   omp_unset_lock(&printlock); 
      // }
      // below_after = from_whole_to_below_diag(whole_slow_way, dimension);
      // if(below_after != i){
      //   omp_set_lock(&printlock);
      //   LOG("th_id : "<<th_id);
      //   LOG("whole_indicies "<<whole_slow_way<<" maps to -> below_diag_indicies "   <<below_after );
      //   omp_unset_lock(&printlock); 
      // }
    }
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  LOG("from_below_diag_to_whole_test run time : "<<readable_time(program_time)<<std::endl);
}


/*
  long long int from_below_diag_to_whole_faster(long long int below_diag_index, int dimension){
    bool debug = false;
    const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
    if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) return (long long int)(-1);

    if (below_diag_index < num_below_diag / 2){
      long long int num_so_far = (long long int)0; // number to the left
      int col = 0;
      long long int num_in_col = (long long int)(dimension - 1);
      while(num_so_far < below_diag_index + (long long int)(1)){
        if(debug){
          LOG("num_so_far : "<<num_so_far);
          LOG("num_in_col : "<<num_in_col);
          LOG("col : "<<col);
          LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
        }
        
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
    }else{
      long long int num_so_far = num_below_diag; // number to the left
      int col = dimension - 2;
      long long int num_in_col = (long long int)1;
      while(num_so_far >= below_diag_index + (long long int)(1)){
        if(false){
          LOG("num_so_far : "<<num_so_far - num_in_col);
          LOG("num_in_col : "<<num_in_col);
          LOG("col : "<<col);
          LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far- num_in_col<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - (num_so_far-num_in_col));
        }
        
        if(num_so_far - num_in_col == below_diag_index + (long long int)(1)){
          return (long long int)(col) * dimension - (long long int)(1);
        }
        if(num_so_far - num_in_col < below_diag_index + (long long int)(1)){ 
          num_so_far -= num_in_col;
          break;
        }else{
          num_so_far -= num_in_col;
          num_in_col += (long long int)(1);
          col -= 1;
        }
      }
      LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
      return (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;    
    }
  }
*/

long long int from_whole_to_below_diag(long long int whole_index, long long int dimension){
  bool debug = false;
  if(debug){
    LOG("whole_index : "<<whole_index);
    LOG("dimension : "<<dimension);
  }
  long long int row = (long long int)(whole_index % (long long int)dimension);
  long long int col = (long long int)(whole_index / (long long int)dimension);
  if(row == col) return (long long int)(-1);
  if(row < (long long int)0 || row > dimension - (long long int)1) return (long long int)(-1);
  if(col < (long long int)0 || col > dimension - (long long int)1) return (long long int)(-1);

  long long int temp = row;
  row = std::max(row, col);
  col = std::min(temp,col);
  // now row is larger than col
  if(debug){
    LOG("row : "<<row);
    LOG("col : "<<col);
  }

  long long int num_below_diag = (long long int)0;
  long long int count = (long long int)0;
  long long int num_in_col = (long long int)(dimension - 1);
  while(count < col){
    if(debug){
      LOG("count : "<<count);
      LOG("num_in_col : "<<num_in_col);
      LOG("num_below_diag : "<<num_below_diag);
      LOG("num_below_diag + row - (dimension - num_in_col) = "<<num_below_diag<<" + "<<row<<" - ("<<dimension<<" - "<<num_in_col<<") = "<<num_below_diag + row - (dimension - num_in_col));
    }
    num_below_diag += num_in_col;
    num_in_col -= (long long int)(1);
    count += (long long int)1;
  }
  return num_below_diag + row - (dimension - num_in_col);
}

template <>
void cpu_get_num_latent_factors<float>(const long long int m, float* S_host, 
  long long int* num_latent_factors, const float percent) 
{
  bool Debug = false;
  float sum = cpu_asum<float>( m, S_host);


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

}

template <>
void cpu_get_latent_factor_mass<float>(const long long int m, float* S_host, 
  const long long int num_latent_factors, float *percent) 
{
  bool Debug = false;
  float sum = cpu_asum<float>( m, S_host);

  float sum_so_far;
  for(int j = 0; j < (int)num_latent_factors; j++){
    sum_so_far += S_host[j];
  }
  percent[0] = sum_so_far / sum;
}

template<typename Dtype>
void cpu_div_US_in_SVD(const long long int m, const long long int num_latent_factors, Dtype* U, const Dtype* S, 
  const bool right_divide_by_S) 
{
  //U is m by num_latent_factors in row major ordering
  for(long long int l = (long long int)0; l < m * num_latent_factors; l+=(long long int)1){
    long long int k;
    if(right_divide_by_S){
        k = l % num_latent_factors; //get column
      }else{
        k = l / num_latent_factors; //get row
      }

      //U[l] = (int)k;
      //U[l] = S[k];
      U[l] = U[l] / S[k];
  }

}
template void cpu_div_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, float* U, const float* S, 
  const bool right_divide_by_S);
 

template<typename Dtype>
void cpu_mult_US_in_SVD(const long long int m, const long long int num_latent_factors, Dtype* U, const Dtype* S, 
  const bool right_multiply_by_S)  
{
  //U is m by num_latent_factors in row major ordering
  for(long long int l = (long long int)0; l < m * num_latent_factors; l+=(long long int)1){
    long long int k;
    if(right_multiply_by_S){
      k = l % num_latent_factors; //get column
    }else{
      k = l / num_latent_factors; //get row
    }

    U[l] = U[l] * S[k];
    //U[l] = S[k];
  }
}

template void cpu_mult_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, float* U, const float* S, 
  const bool right_multiply_by_S);


template<>
void cpu_orthogonal_decomp<float>(const long long int m, const long long int n, const bool row_major_ordering,
                                  long long int* num_latent_factors, const float percent,
                                  float* A, float* U, float* V, bool SV_with_U, float* S)
{
  ABORT_IF_EQ(0, 0, "Function requires Eigen Library");
  /*
  bool Debug = false;
  LOG("cpu_orthogonal_decomp called");
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  long long int lda, sda;
  float *d_U  = NULL;
  float *d_VT = NULL;

  
  // A in row major ordering is equivalent to A^T in column major ordering
  // A in column major ordering is equivalent to A^T in row major ordering
  

  if(n > m){
    //we have to solve the transpose problem
    if(Debug) LOG("Solving the tranpose problem...");
    d_U  = V;
    d_VT = U;
    lda  = n;
    sda  = m;
    if(!row_major_ordering){
      if(Debug) LOG("A in row major ordering is equivalent to A^T in column major ordering...");
      MatrixInplaceTranspose<float>(A, m, n, row_major_ordering);
    }
  }else{
    d_U  = U;
    d_VT = V;
    lda  = m;
    sda  = n;
    if(row_major_ordering){
      if(Debug) LOG("swap ordering of A");
      cpu_swap_ordering<float>(m, n, A, row_major_ordering);
    }
  };

  long long int smaller_dim = (int)std::min(m, n);
  Eigen::MatrixXf eigen_A = Eigen::Map<Eigen::MatrixXf>( A, lda, sda);

  if(Debug) {
    LOG("eigen_A.rows() : "<<eigen_A.rows()) ;
    LOG("eigen_A.cols() : "<<eigen_A.cols()) ;
    LOG("eigen_A.size() : "<<eigen_A.size()) ;
    //std::cout << "Here is the matrix A:" << std::endl << eigen_A << std::endl;
  }

  // float* A_copy = NULL;
  // A_copy = (float *)malloc(m*n *  SIZE_OF(float)); 
  // checkErrors(A_copy);  
  // cpu_set_all<float>(A_copy, m*n, (float)0.0);
  // Eigen::Map<Eigen::MatrixXf>( A_copy, sda, lda ) = eigen_A;
  // print_host_mtx<float>(A_copy, sda, lda, "A_copy", 1, strPreamble(blank));
  // free(A_copy);

  if(Debug) LOG("Compute svd...") ;
  if(smaller_dim <= (long long int)16){
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if(Debug){
      std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
      std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
      std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    }
    eigen_A.resize(0,0);
    Eigen::Map<Eigen::MatrixXf>( d_VT, sda, smaller_dim) = svd.matrixV();
    Eigen::Map<Eigen::MatrixXf>( d_U, lda, smaller_dim ) =  svd.matrixU();
    if(S != NULL){
      if(Debug) LOG("saving singular values.");
      Eigen::Map<Eigen::VectorXf>( S, smaller_dim ) = svd.singularValues();
    }

  }else{
    //Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeFullV);
    Eigen::BDCSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if(Debug){
      std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
      std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
      std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    }
    eigen_A.resize(0,0);
    Eigen::Map<Eigen::MatrixXf>( d_VT, sda, smaller_dim) = svd.matrixV();
    Eigen::Map<Eigen::MatrixXf>( d_U, lda, smaller_dim ) =  svd.matrixU();
    
    if(S != NULL){
      if(Debug) LOG("saving singular values.");
      Eigen::Map<Eigen::VectorXf>( S, smaller_dim ) = svd.singularValues();
    }
  }
  //MatrixXf eigen_V(smaller_dim, n) = svd.matrixV();
  if(Debug) LOG("svd computed...") ;
  

  // M, N, K
  //M number of rows of matrix op(A) and C.
  //N is number of columns of matrix op(B) and C.]
  //K is number of rows of op(B) and columns of op(A).

  // op(A) is M by K
  // op(B) is K by N
  // C is M by N
  // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C

  if(n > m){
    if(!row_major_ordering){
      if(Debug) LOG("tranpose back...");
      MatrixInplaceTranspose<float>(A, n, m);
    }

    // if(Debug) LOG("solving for V now...") ;
    // C←αAB + βC
    // cpu_gemm<float>(true, false, smaller_dim, n, m,
    //                 (float)1.0, U, A, (float)0.0, V);
    //cpu_mult_US_in_SVD<float>(m, smaller_dim, U, S, true);
    //cpu_div_US_in_SVD<float>(n, smaller_dim, V, S, true);
  }else{
    
    if(row_major_ordering){
      if(Debug) LOG("swap ordering of A back...");
      cpu_swap_ordering<float>(m, n, A, !row_major_ordering);
    }
    //if(Debug) LOG("solving for U now...") ;
    //C←αAB + βC
    // cpu_gemm<float>(false, false, m, smaller_dim, n,
    //                 (float)1.0, A, V, (float)0.0, U);
  };


  if(row_major_ordering){
    //if(Debug) LOG("A in row major ordering is equivalent to A^T in column major ordering...");
    //cpu_swap_ordering<float>(n, smaller_dim, V, !row_major_ordering);
    cpu_swap_ordering<float>(m, smaller_dim, U, !row_major_ordering);
    if(SV_with_U){
      cpu_mult_US_in_SVD<float>(m, smaller_dim, U, S, true);
    }else{
      cpu_mult_US_in_SVD<float>(smaller_dim, n, V, S, false);
    }
  }else{
    
    // A in row major ordering is equivalent to A^T in column major ordering
    // A in column major ordering is equivalent to A^T in row major ordering
    
    if(SV_with_U){
      cpu_mult_US_in_SVD<float>(smaller_dim, m, U, S, false);
    }else{
      cpu_mult_US_in_SVD<float>(smaller_dim, n, V, S, false);
    }
  }
  cpu_get_num_latent_factors<float>(smaller_dim, S, num_latent_factors, percent);

  if(Debug && 0){

    LOG("num_latent_factors : "<<num_latent_factors[0]) ;
    save_host_mtx_to_file<float>(U, m, m, "U_3");
    save_host_mtx_to_file<float>(V, n, m, "V_3");
    save_host_mtx_to_file<float>(A, m, n, "A");


    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }    
  save_host_array_to_file<float>(S, std::min(m,n), "singular_values");

  if(0){
    float* R  = NULL;
    R = (float *)malloc(m * n *  SIZE_OF(float)); 
    checkErrors(R);

    
    // A is m by n stored in row-maj ordering where m<<n
    // V is n by m stored in row-maj ordering
    // (V^T is m by n)
    // U is m by m stored in row-maj ordering

    // M, N, K
    //M number of rows of matrix op(A) and C.
    //N is number of columns of matrix op(B) and C.]
    //K is number of rows of op(B) and columns of op(A).

    // op(A) is M by K
    // op(B) is K by N
    // C is M by N
    // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C

    cpu_gemm<float>(false, true, m, m, smaller_dim, // <- num_latent_factors[0]
     (float)1.0, U, U, (float)0.0, R);

    save_host_mtx_to_file<float>(R, m, m, "UUT");

    cpu_gemm<float>(true, false, smaller_dim, smaller_dim, n , // <- num_latent_factors[0]
     (float)1.0, V, V, (float)0.0, R);

    save_host_mtx_to_file<float>(R, smaller_dim, smaller_dim, "VTV");



    cpu_gemm<float>(false, true, m, n, smaller_dim , // <- num_latent_factors[0]
     (float)1.0, U, V, (float)0.0,
     R);

    cpu_axpby<float>(m * n, (float)1.0, A,
      (float)(-1.0), R);

    save_host_mtx_to_file<float>(R, m, n, "svd_error");

    //float range_A    = gpu_range<float>(m * n,  A);
    float error      = cpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
    free(R);
    // LOG("A mtx range of values = "<<range_A) ;
    LOG("SVD max error = "<<error) ;
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
    LOG("Press Enter to continue.") ;
    std::cin.ignore();

  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_orthogonal_decomp run time : "<<readable_time(program_time)<<std::endl);

  */

}


void cpu_orthogonal_decomp_test() {

  // MatrixXf M = MatrixXf::Random(3,2);
  // std::cout << "Here is the matrix M:" << std::endl << M << std::endl;
  // JacobiSVD<MatrixXf> svd(M, ComputeThinU | ComputeThinV);
  // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
  // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
  // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;

  const int m = 19;
  const int n = 17;
  const int min_dim = std::min(m,n);
  const int max_dim = std::max(m,n);
  std::string blank = "";

  float* A = NULL;
  float* U = NULL;
  float* V = NULL;
  float* S = NULL;
  A = (float *)malloc(m*n *  SIZE_OF(float)); 
  U = (float *)malloc(m*min_dim *  SIZE_OF(float)); 
  V = (float *)malloc(n*min_dim *  SIZE_OF(float)); 
  S = (float *)malloc(min_dim *  SIZE_OF(float)); 
  checkErrors(A);
  checkErrors(U);
  checkErrors(V);
  checkErrors(S);

  cpu_set_all<float>(U, m*min_dim, (float)0.0);
  cpu_set_all<float>(V, min_dim*n, (float)0.0);
  cpu_set_all<float>(S, min_dim, (float)0.0);

  host_rng_uniform<float>(m*n, (float)(-10.0), (float)10.0, A);

  //save_host_mtx_to_file<float>(A, m, n, "A");

  long long int num_latent_factors;
  const float percent = (float)0.95;
  bool row_major_ordering = false;
  if(1){
    print_host_mtx<float>(A, m, n, "A", row_major_ordering, strPreamble(blank));
  }
  cpu_orthogonal_decomp<float>(m, n, row_major_ordering, &num_latent_factors, percent,
  A, U, V, false, S);


  if(1){
    print_host_mtx<float>(A, m, n, "A", row_major_ordering, strPreamble(blank));
    print_host_mtx<float>(U, m, min_dim, "U", row_major_ordering, strPreamble(blank));
    print_host_mtx<float>(V, n, min_dim, "V", false, strPreamble(blank));
    print_host_array<float>(S, min_dim, "S", strPreamble(blank));
  }

  if(row_major_ordering){
    float* R  = NULL;
    R = (float *)malloc(max_dim * max_dim *  SIZE_OF(float)); 
    checkErrors(R);

    /*
        A is m by n stored in row-maj ordering where m<<n
        V is n by m stored in row-maj ordering
        (V^T is m by n)
        U is m by m stored in row-maj ordering
    */
    // M, N, K
    //M number of rows of matrix op(A) and C.
    //N is number of columns of matrix op(B) and C.]
    //K is number of rows of op(B) and columns of op(A).

    // op(A) is M by K
    // op(B) is K by N
    // C is M by N
    // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C


    cpu_gemm<float>(true, false, min_dim, min_dim, m /*num_latent_factors[0]*/,
     (float)1.0, U, U, (float)0.0, R);
    print_host_mtx<float>(R, min_dim, min_dim, "UTU", 1, strPreamble(blank));

    if(min_dim == m){
      cpu_gemm<float>(false, true, m, m, min_dim /*num_latent_factors[0]*/,
       (float)1.0, U, U, (float)0.0, R);
      print_host_mtx<float>(R, m, m, "UUT", 1, strPreamble(blank));
    }

    cpu_gemm<float>(true, false, min_dim, min_dim, n /*num_latent_factors[0]*/,
     (float)1.0, V, V, (float)0.0, R);
    print_host_mtx<float>(R, min_dim, min_dim, "VTV", 1, strPreamble(blank));

    if(min_dim == n){
      cpu_gemm<float>(false, true, n, n, min_dim /*num_latent_factors[0]*/,
       (float)1.0, V, V, (float)0.0, R);
      print_host_mtx<float>(R, n, n, "VVT", 1, strPreamble(blank));
    }



    cpu_gemm<float>(false, true, m, n, min_dim /*num_latent_factors[0]*/,
     (float)1.0, U, V, (float)0.0,
     R);

    cpu_axpby<float>(m * n, (float)1.0, A,
      (float)(-1.0), R);

    print_host_mtx<float>(R, m, n, "svd_error", 1, strPreamble(blank));

    //float range_A    = gpu_range<float>(m * n,  A);
    float error      = cpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
    free(R);
    // LOG("A mtx range of values = "<<range_A) ;
    LOG("SVD max error = "<<error) ;
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }

  free(A);
  free(U);
  free(V);
  free(S);  

}



void cpu_center_rows(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var) 
{
  LOG("cpu_center_rows called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, rows);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row = (long long int)th_id; row < rows; row += (long long int)nthreads){
    //for(long long int row = (long long int)0; row < rows; row+=(long long int)1){
      float mean = (float)0.0;
      float std_dev = (float)0.0;
      for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
        mean += X[row * cols + i];
        std_dev += pow(X[row * cols + i], (float)2.0);
      }
      mean /= (float)cols;
      std_dev /= (float)cols;
      std_dev = std_dev - pow(mean, (float)2.0);
      user_var[row] = std_dev;
      std_dev = sqrt(std_dev);
      user_means[row] = mean;

      if(std_dev == (float)0.0 ){
        for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
          X[row * cols + i] = val_when_var_is_zero;
        } 
      }else{
        for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
          X[row * cols + i] = (X[row * cols + i] - mean) / std_dev;
          if (::isinf(X[row * cols + i]) || ::isnan(X[row * cols + i])){
            ABORT_IF_NEQ(0, 1, "isBad");
          };
        } 
      }       
    }
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));  
  if(1) LOG("cpu_center_rows run time : "<<readable_time(program_time)<<std::endl);
}

template<typename Dtype>
__global__ void gpu_logarithmic_histogram_abs_val_kernel(const long long int n, Dtype* error, Dtype* probability, 
                                                          int min_pow, int max_pow, int non_zero_count)
{

}

template <typename Dtype>
void cpu_logarithmic_histogram_abs_val(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
                                       const int rows_B, const int num_sparse_entries, 
                                       const int* csr_rows_B, const int* coo_cols_B,
                                       const Dtype* coo_entries_B, int* selection, 
                                       int min_pow, int max_pow, Dtype* probability)
{
  cpu_set_all(probability, (long long int)(max_pow - min_pow + 1), (Dtype)0.0);

  const int row_skip = csr_rows_B[0];
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, rows_B);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row_B = (long long int)th_id; row_B < (long long int)rows_B; row_B += (long long int)nthreads){
      for(long long int coo_index = (long long int)(csr_rows_B[row_B]); coo_index < (long long int)(csr_rows_B[row_B + 1]); coo_index+=(long long int)1){
        int col = coo_cols_B[coo_index - row_skip];
        Dtype error_temp = std::abs(dense_mtx_A[(long long int)(selection[row_B])  * (long long int)cols + (long long int)col] - coo_entries_B[coo_index - row_skip]);
        if( (Dtype)0.0 <= error_temp && error_temp < (Dtype)pow((Dtype)10.0, (Dtype)min_pow) ) {
          probability[0] += (Dtype)1.0 / (Dtype)num_sparse_entries;
        }else{
          int count = 1;
          for(int j = min_pow + 1; j <= max_pow; j++){
            if( (Dtype)pow((Dtype)10.0, (Dtype)(j - 1))<= error_temp && error_temp < (Dtype)pow((Dtype)10.0, (Dtype)j) ) {
              probability[count] += (Dtype)1.0 / (Dtype)num_sparse_entries;
              break;
            }else{
              count++;
            }
          }
        }
      }
    }
  }
  // for(int i = 0; i < (max_pow - min_pow + 1), i++){

  // }
}

template void cpu_logarithmic_histogram_abs_val<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
                                       const int rows_B, const int num_sparse_entries, 
                                       const int* csr_rows_B, const int* coo_cols_B,
                                       const float* coo_entries_B, int* selection, 
                                       int min_pow, int max_pow, float* probability);


template <typename Dtype>
void cpu_sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection, Dtype* error)
{
  bool Debug = false;
  LOG("cpu_sparse_nearest_row called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in row major ordering
  */
  const int row_skip = csr_rows_B[0];
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, rows_B);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row_B = (long long int)th_id; row_B < (long long int)rows_B; row_B += (long long int)nthreads){
    //CUDA_KERNEL_LOOP(row_B,rows_B) {
      Dtype closest_A_row_dist = (Dtype)10000.0;
      int   closest_A_row      = 0;
      for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
        Dtype temp = (Dtype)0.0;
        long long int count = (long long int)0;
        for(long long int coo_index = (long long int)(csr_rows_B[row_B]); coo_index < (long long int)(csr_rows_B[row_B + 1]); coo_index+=(long long int)1){
          int col = coo_cols_B[coo_index - row_skip];
          count += (long long int)1;
          //cpu_incremental_average<Dtype>(count, &temp, (Dtype)pow(dense_mtx_A[row_A  * (long long int)cols + (long long int)col] - coo_entries_B[coo_index - row_skip], (Dtype)2.0));
          cpu_incremental_average<Dtype>(count, &temp, std::abs(dense_mtx_A[row_A  * (long long int)cols + (long long int)col] - coo_entries_B[coo_index - row_skip]));
          //temp += pow(dense_mtx_A[row_A  * (long long int)cols + (long long int)col] - coo_entries_B[coo_index - row_skip], (Dtype)2.0);
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
        ABORT_IF_NEQ(0, 1, "isBad");
      };
    }
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_sparse_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}

template void cpu_sparse_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const float* coo_entries_B, int* selection, float* error);


template<typename Dtype>
void cpu_dense_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype* error)
{
  bool Debug = false;
  LOG("cpu_dense_nearest_row called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, rows_B);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row_B = (long long int)th_id; row_B < (long long int)rows_B; row_B += (long long int)nthreads){
      //CUDA_KERNEL_LOOP(row_B,rows_B) {
      Dtype closest_A_row_dist = (Dtype)1000000.0;
      int   closest_A_row      = 0;
      for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
        Dtype temp = (Dtype)0.0;
        for(long long int col = (long long int)0; col < (long long int)cols; col+=(long long int)1){
          temp += pow(dense_mtx_A[row_A * (long long int)cols + col] - dense_mtx_B[row_B * (long long int)cols+ col], (Dtype)2.0);
        }
        if(temp < closest_A_row_dist || row_A == (long long int)0){
          closest_A_row_dist = temp;
          closest_A_row      = (int)row_A;
        }
      }
      selection[row_B] = closest_A_row;
      error[row_B] = closest_A_row_dist;

      if (::isinf(error[row_B]) || ::isnan(error[row_B])){
        ABORT_IF_EQ(0, 0, "isBad");
      };
    };
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_dense_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}


template void cpu_dense_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const float* dense_mtx_B, int* selection, float* error);




template<typename Dtype>
void cpu_calculate_KM_error_and_update(const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                                    const int rows_B, const Dtype* dense_mtx_B, int* selection,  
                                                    Dtype alpha, Dtype lambda, Dtype* checking)
{
  bool Debug = false;
  LOG("cpu_calculate_KM_error_and_update called");
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  Dtype* per_thread = NULL;
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, (long long int)rows_A * (long long int)cols);
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
    if(Debug){
      per_thread = (Dtype *)malloc(nthreads * SIZE_OF(Dtype));
      checkErrors(per_thread);
    }
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int index = (long long int)th_id; index < (long long int)rows_A * (long long int)cols; index += (long long int)nthreads){
      //CUDA_KERNEL_LOOP(index, num) {
      long long int row_A = index / ((long long int) cols);
      long long int col = index % ((long long int) cols);
      Dtype temp = (Dtype)0.0;
      int count = 0;
      if(Debug){
        per_thread[th_id] = (Dtype)0.0;
      }
      for(long long int row_B = (long long int)0; row_B < (long long int)rows_B; row_B +=(long long int)1){
        if(selection[row_B] == (int)row_A){
          temp += dense_mtx_B[row_B * (long long int)cols + col];
          count++;
        }
      }
      if(count > 0){
        Dtype old_val = dense_mtx_A[row_A  * (long long int)cols + col];
        dense_mtx_A[row_A * (long long int)cols + col] = ((Dtype)1.0 - alpha * lambda) * old_val + alpha * (temp / ((Dtype)count));
        if(Debug){
          // omp_set_lock(&printlock);
          // LOG("th_id : "<<th_id);
          // LOG("index : "<<index);
          // LOG("row_A : "<<row_A);
          // LOG("col : "<<col);
          // LOG("count : "<<count);
          // LOG("update term : "<<temp / ((float)count));
          // LOG("before term : "<<old_val);
          // LOG("new term : "<<dense_mtx_A[row_A  * (long long int)cols + col]);
          // omp_unset_lock(&printlock);
          per_thread[th_id] = std::max(per_thread[th_id], std::abs(dense_mtx_A[row_A  * (long long int)cols + col] - old_val));
        }
      }
      if (::isinf(dense_mtx_A[row_A  * (long long int)cols + col]) || ::isnan(dense_mtx_A[row_A  * (long long int)cols + col])){
        omp_set_lock(&printlock);
        LOG("th_id : "<<th_id);
        ABORT_IF_EQ(0, 0, "isBad");
        omp_unset_lock(&printlock);
      }
    }
  }
  if(Debug){
    print_host_array(per_thread, nthreads, "max per_thread", strPreamble(blank));
    save_host_array_to_file(per_thread, nthreads, "max per_thread", strPreamble(blank));
    if(per_thread) free(per_thread);
  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_calculate_KM_error_and_update run time : "<<readable_time(program_time)<<std::endl);
}

template void cpu_calculate_KM_error_and_update<float>(const int rows_A, const int cols, float* dense_mtx_A, 
    const int rows_B, const float* dense_mtx_B, int* selection, float alpha, float lambda, float* checking);


template<>
void cpu_supplement_training_mtx_with_item_sim<float>(const long long int ratings_rows_training, 
  const long long int ratings_cols_training, 
  const bool row_major_ordering,
  const int* csr_format_ratingsMtx_userID_training,
  const int* coo_format_ratingsMtx_itemID_training,
  const float* coo_format_ratingsMtx_rating_training,
  float* full_training_ratings_mtx,
  const int* top_N_most_sim_itemIDs,
  const float* top_N_most_sim_item_similarity, const int top_N){

  long long int total = ratings_rows_training * ratings_cols_training;
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, total);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
  int th_id = 0;
  #ifdef _OPENMP
    th_id = omp_get_thread_num();
  #endif
    for(long long int ind_ = (long long int)th_id; ind_ < total; ind_+=(long long int)nthreads){
      if(full_training_ratings_mtx[ind_] == (float)0.0){
        //user have not rated the item, has user rated similar items?
        //walk through the items user HAS rated and calculate a weighted average of ratings for similar items
        int item; 
        int user;
        if(row_major_ordering){
          item = (int)(ind_ / ratings_cols_training);
          user = (int)(ind_ % ratings_cols_training);
        }else{
          item = (int)(ind_ % ratings_rows_training);
          user = (int)(ind_ / ratings_rows_training);
        }
        int   count_micro   = 0;
        float num_micro    = (float)0.0;
        float denom_micro = (float)0.0;
        //float user_rating = (float)0.0;
        int start = 0;
        for(int i = csr_format_ratingsMtx_userID_training[user]; i < csr_format_ratingsMtx_userID_training[user + 1]; i++){
          int user_itemID_other = coo_format_ratingsMtx_itemID_training[i];
          for(int k = start; k < top_N; k++){
            int _other_similar_item_index = top_N_most_sim_itemIDs[(long long int)k + (long long int)item * (long long int)top_N];
            if( user_itemID_other == _other_similar_item_index){
              count_micro += 1;
              num_micro += coo_format_ratingsMtx_rating_training[i] * top_N_most_sim_item_similarity[(long long int)k + (long long int)item * (long long int)top_N];
              denom_micro += top_N_most_sim_item_similarity[(long long int)k + (long long int)item * (long long int)top_N] ; 
              start = k + 1;
              break;
            }else if(user_itemID_other < _other_similar_item_index){
              start = k;
              break;
            }
          }
        }
        if(count_micro > 0){
          float user_rating = num_micro / denom_micro;
          full_training_ratings_mtx[ind_] = user_rating;
          if (::isinf(user_rating) || ::isnan(user_rating)){
            ABORT_IF_EQ(1,1,"bad");
          }
        }
      }
    }
  }
}




