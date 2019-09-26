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
    	std::cout<<__FILE__<<" line "<<__LINE__<<" at "<<currentDateTime()<<" : "<<output_statement<<std::endl;


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

const std::string currentDateTime();
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

template <typename Dtype>
std::string ToString(const Dtype val);

extern "C" std::string readable_time(double ms);

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
void save_host_mtx_to_file(const Dtype* A_host, const int lda, const int  sda, std::string title);


//============================================================================================
// Me Made
//============================================================================================

void cpu_fill_training_mtx(const long long int ratings_rows_training, const long long int ratings_cols_training,  
                           const long long int num_entries_CU, const bool row_major_ordering,
                            const int* csr_format_ratingsMtx_userID_host_training,
                            const int* coo_format_ratingsMtx_itemID_host_training,
                            const float* coo_format_ratingsMtx_rating_host_training,
                            float* full_training_ratings_mtx);


void cpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const long long int M, const long long int N, 
  bool row_major_ordering, float* x, bool shuffle_rows);

template <typename Dtype>
void cpu_set_all(Dtype* x, const int N, Dtype alpha);

template <typename Dtype>
void cpu_set_as_index(Dtype* x, const long long int rows, const long long int cols);

void cpu_get_cosine_similarity(const long long int ratings_rows, const int num_entries,
                              const int* csr_format_ratingsMtx_userID_host,
                              const int* coo_format_ratingsMtx_itemID_host,
                              const float* coo_format_ratingsMtx_rating_host,
                              float* cosine_similarity);

template <typename Dtype>
void cpu_sort_index_by_max(const long long int rows, const long long int cols,  Dtype* x, int* indicies);

template<typename Dtype>
void cpu_sort_index_by_max(const long long int dimension,  Dtype* x, int* indicies, int top_N);


void cpu_count_appearances(const int top_N, const long long int dimension,  int* count, const int* indicies );

void cpu_mark_CU_users(const int ratings_rows_CU, const int ratings_rows, const int* x, int* y );



long long int from_below_diag_to_whole(long long int below_diag_index, int dimension);

long long int from_whole_to_below_diag(long long int whole_index, int dimension);






#ifndef CPU_ONLY  // GPU

#endif  // !CPU_ONLY


#endif //UTIL_H_