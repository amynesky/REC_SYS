#ifndef UTIL_H_
#define UTIL_H_

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
 
#include "util_gpu.cuh"

#define ABORT_IF_EQ(a, b, output_statement) \
  if(a==b){ \
      std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
      if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_NEQ(a, b, output_statement) \
	if(a!=b){ \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
    	if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_LESS(a, b, output_statement) \
	if(a < b){ \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
    	if (abort) exit(EXIT_FAILURE);\
    }

#define LOG(output_statement) \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;


#define checkErrors(ans) { Hassert((ans), __FILE__, __LINE__); }
template <typename Dtype>
inline void Hassert(const Dtype *ptr, const char *file, int line, bool abort=true)
{
   if (ptr == NULL) 
   {
      fprintf(stderr,"assert: %s %d\n",  file, line);
      if (abort) {ABORT_IF_NEQ(0, 1, "Fatal: failed to allocate");};
   }
}
//============================================================================================
// manipulate memory
//============================================================================================


template <typename Dtype>
void host_copy(const int N, const Dtype* X, Dtype* Y);
//============================================================================================
// math functions
//============================================================================================
int gcd(int a, int b);

template < typename Dtype>
void cpu_permute(Dtype* a, const int* pvt, const long long int rows, const long long int cols, bool direction);

template < typename Dtype>
void MatrixInplaceTranspose(Dtype *A, int r, int c);

//============================================================================================
// random utilities
//============================================================================================


template < typename Dtype>
void  fillupMatrix(Dtype *A , int lda , int rows, int cols, int seed = 0);

template <typename Dtype>
void host_rng_uniform(const long long int n, const Dtype a, const Dtype b, Dtype* r);
//============================================================================================
// prints
//============================================================================================

template <typename T>
std::string ToString(T val);

template < typename Dtype>
void printPartialMatrices(Dtype * A, Dtype * B, int rows, int cols , int ld);

template < typename Dtype>
void printPartialMtx(Dtype * A, int rows, int cols , int ld);


template <typename Dtype>
void save_host_array_to_file(const Dtype* A_host, int count, std::string title);

template <typename Dtype>
void append_host_array_to_file(const Dtype* A_host, int count, std::string title);


void save_host_arrays_side_by_side_to_file(const int* A_host, const int* B_host, 
                                           const float* C_host, int count, std::string title);

template <typename Dtype>
void save_host_mtx_to_file(const Dtype* A_host, const long long int lda, const long long int  sda, std::string title);


//============================================================================================
// Me Made
//============================================================================================

void cpu_fill_training_mtx(const long long int ratings_rows_training, const long long int ratings_cols_training,  
                           const long long int num_entries_GA, const bool row_major_ordering,
                            const int* csr_format_ratingsMtx_userID_dev_training,
                            const int* coo_format_ratingsMtx_itemID_dev_training,
                            const float* coo_format_ratingsMtx_rating_dev_training,
                            float* full_training_ratings_mtx);


void cpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const int M, const int N, bool row_major_ordering, float* x, bool shuffle_rows);

template <typename Dtype>
void cpu_set_all(Dtype* x, const int N, Dtype alpha);







#ifndef CPU_ONLY  // GPU

#endif  // !CPU_ONLY


#endif //UTIL_H_