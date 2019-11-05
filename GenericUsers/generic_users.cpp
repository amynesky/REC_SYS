/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to get better performance by
 * batching CUBLAS calls with the use of using streams
 */

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
#include <cuda_runtime.h>
#include "/home/nesky/REC_SYS/helper_files/helper_cuda.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
//#include "helper_cusolver.h"
#include "cusolverDn.h"

// Utilities and system includes
#include "/home/nesky/REC_SYS/helper_files/CSVReader.h"
#include "/home/nesky/REC_SYS/helper_files/util.h"
#include "/home/nesky/REC_SYS/helper_files/util_gpu.cuh"
#include "generic_users.h"

const char *sSDKname = "Generic Users Recommender Systems";

const bool Debug = 1;

bool Content_Based = 0;
bool random_initialization = 1;
bool Conserve_GPU_Mem = 0;

#define update_Mem(new_mem) \
    allocatedMem += static_cast<long long int>(new_mem); \
    memLeft = static_cast<long long int>(devMem) - allocatedMem; \
    ABORT_IF_LESS(memLeft, 0, "Out of Memory"); 
    // if(Debug) LOG(allocatedMem<<" allocated bytes on the device");
    //if(Debug) LOG(memLeft<<" available bytes left on the device");
    //
    //ABORT_IF_LESS(allocatedMem, (long long int)((double)devMem * (double)0.75), "Out of Memory"); \
    // if(Debug) LOG((int)devMem <<" total bytes on the device");\
    // if(Debug) LOG(new_mem<<" change in memory on the device");\






// #define allocate_V() \
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     checkCudaErrors(cudaMalloc((void**)&U_GU,       batch_size_GU       * batch_size_GU                       * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * ratings_cols                        * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training * std::max(min_, batch_size_training) * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training * ratings_cols                        * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_, batch_size_testing)  * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * sizeof(float)));
//     update_Mem(Training_bytes);











int main(int argc, char *argv[])
{
    struct timeval program_start, program_end, training_start, training_end;
    double program_time;
    double training_time;
    gettimeofday(&program_start, NULL);

    long long int allocatedMem = (long long int)0; 


    cublasStatus_t cublas_status     = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;


    printf("%s Starting...\n\n", sSDKname);
    std::cout << "Current Date and Time :" << currentDateTime() << std::endl<< std::endl;
    LOG("Debug = "<<Debug);

    /* initialize random seed: */
    srand (cluster_seedgen());

    int dev = findCudaDevice(argc, (const char **) argv);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printf("Major revision number:         %d\n",  devProp.major);
    //printf("Minor revision number:         %d\n",  devProp.minor);
    int major = devProp.major;
    int minor = devProp.minor;
    float compute_capability = (float)major + (float)minor / (float)10;
    //printf("\nDevice Compute Capability: %f \n\n", compute_capability);

    long long int devMem = std::abs(static_cast<long long int>(getDeviceMemory()));
    LOG("devMem : "<<devMem);
    long long int memLeft = devMem;

    if (dev == -1)
    {
        return CUBLASTEST_FAILED;
    }

    //============================================================================================
    // Dense matrix handles and descriptors
    //============================================================================================
    cublasHandle_t dn_handle;
    if (cublasCreate(&dn_handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stdout, "CUBLAS initialization failed!\n");
        exit(EXIT_FAILURE);
    };

    cusolverDnHandle_t dn_solver_handle;
    if (cusolverDnCreate(&dn_solver_handle) != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stdout, "DN CUSOLVER initialization failed!\n");
        exit(EXIT_FAILURE);
    };


    //============================================================================================
    // Sparse matrix handles and descriptors
    //============================================================================================
    cusparseHandle_t sp_handle;

    if (cusparseCreate(&sp_handle) != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "CUSPARSE initialization failed!\n");
        exit(EXIT_FAILURE);
    };

    /* create and setup matrix descriptor */ 
    cusparseMatDescr_t sp_descr;
    if (cusparseCreateMatDescr(&sp_descr) != CUSPARSE_STATUS_SUCCESS) {  
        fprintf(stdout, "SP Matrix descriptor initialization failed\n");
        exit(EXIT_FAILURE);
    } 
    cusparseSetMatType(sp_descr,CUSPARSE_MATRIX_TYPE_GENERAL);



    cusolverSpHandle_t sp_solver_handle;
    if (cusolverSpCreate(&sp_solver_handle) != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stdout, "SP CUSOLVER initialization failed!\n");
        exit(EXIT_FAILURE);
    };




    //============================================================================================
    // Get the ratings data from CSV File
    //============================================================================================


    // Creating an object of CSVWriter

    std::string Dataset_Name;

    
    std::string csv_Ratings_Path;
    std::string csv_keyWords_path;

    long long int temp_num_entries;

    std::map<int, int> items_dictionary;

    int case_ = 1;

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            Dataset_Name = "MovieLens 20 million";

            csv_Ratings_Path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/ratings_copy.csv";
            csv_keyWords_path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/movies_copy.csv";
            //temp_num_entries = csv_Ratings.num_rows() - 1; // the first row is a title row
            temp_num_entries = 20000264 - 1;   // MovieLens 20 million
            break;
        }case 2:{ // code to be executed if n = 2;
            Dataset_Name = "Rent The Runaway";
            csv_Ratings_Path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data_copy.json";
            //csv_keyWords_path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data.json";
            Content_Based = 0;
            temp_num_entries = 192544;           // use for Rent The Runaway dataset
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }

    if(random_initialization) Content_Based = 0;

    LOG("Training using the "<< Dataset_Name <<" dataset");
    LOG("csv_Ratings_Path : "<< csv_Ratings_Path);
    LOG("csv_keyWords_path : "<< csv_keyWords_path <<" dataset");
    LOG("random_initialization : "<< random_initialization<<std::endl);
    LOG("Content_Based : "<< Content_Based<<std::endl);



    CSVReader csv_Ratings(csv_Ratings_Path);
    

    const long long int num_entries = temp_num_entries;
    //const int num_entries = 100000; //for debuging code
    
    int*   coo_format_ratingsMtx_userID_host  = NULL;
    int*   coo_format_ratingsMtx_itemID_host  = NULL;
    float* coo_format_ratingsMtx_rating_host  = NULL;
    coo_format_ratingsMtx_userID_host = (int *)  malloc(num_entries *  sizeof(int)); 
    coo_format_ratingsMtx_itemID_host = (int *)  malloc(num_entries *  sizeof(int)); 
    coo_format_ratingsMtx_rating_host = (float *)malloc(num_entries *  sizeof(float)); 
    checkErrors(coo_format_ratingsMtx_userID_host);
    checkErrors(coo_format_ratingsMtx_itemID_host);
    checkErrors(coo_format_ratingsMtx_rating_host);

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            //Dataset_Name = "MovieLens 20 million";

            csv_Ratings.getData(coo_format_ratingsMtx_userID_host,
                                coo_format_ratingsMtx_itemID_host,
                                coo_format_ratingsMtx_rating_host, 
                                num_entries, Content_Based == 1, &items_dictionary);
            break;
        }case 2:{ // code to be executed if n = 2;
            //Dataset_Name = "Rent The Runaway";
            int missing_ = 5;
            LOG("value used to fill in missing ratings : "<< missing_) ;
            csv_Ratings.getDataJSON(coo_format_ratingsMtx_userID_host,
                                    coo_format_ratingsMtx_itemID_host,
                                    coo_format_ratingsMtx_rating_host, 
                                    num_entries, missing_);
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }


    if(Debug && 0){
        save_host_arrays_side_by_side_to_file(coo_format_ratingsMtx_userID_host, coo_format_ratingsMtx_itemID_host, 
                                              coo_format_ratingsMtx_rating_host, num_entries, "rows_cols_rating");
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();          
    }
 


    int*   coo_format_ratingsMtx_userID_dev;
    int*   coo_format_ratingsMtx_itemID_dev;
    float* coo_format_ratingsMtx_rating_dev;
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_userID_dev,  num_entries * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev,  num_entries * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev,  num_entries * sizeof(float)));
    update_Mem(2 * num_entries * sizeof(int) + num_entries * sizeof(float));

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_dev,  coo_format_ratingsMtx_userID_host,  num_entries * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev,  coo_format_ratingsMtx_itemID_host,  num_entries * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev,  coo_format_ratingsMtx_rating_host,  num_entries * sizeof(float), cudaMemcpyHostToDevice));
    

    free(coo_format_ratingsMtx_userID_host);
    free(coo_format_ratingsMtx_itemID_host);
    free(coo_format_ratingsMtx_rating_host);

    if(Debug){
        // Print the content of row by row on screen
        // for(int row = 0; row <2; row++)
        // {
        //     for(int col = 0; col <dims; col++)
        //     {
        //         std::cout<<ratings_host[row + col * num_entries] << " , ";
        //         if (col < dims-1 ) std::cout<<" , ";
        //     }
        //     std::cout<<std::endl;
        // }
    }

    const long long int ratings_rows = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_userID_dev) + 1); 
    const long long int ratings_cols = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_itemID_dev) + 1); 

    LOG(std::endl<<"The sparse data matrix has "<<ratings_rows<<" users and "<<ratings_cols<<" items with "<<num_entries<<" specified entries.");
    LOG("The sparse data matrix has "<<(float)(ratings_rows * ratings_cols - num_entries) / (float)(ratings_rows * ratings_cols)<<" empty entries.");
    
    int*   csr_format_ratingsMtx_userID_dev;
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev, (ratings_rows + 1) * sizeof(int)));
    update_Mem( (ratings_rows + 1) * sizeof(int) );

    cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_ratingsMtx_userID_dev, num_entries, 
                                       ratings_rows, csr_format_ratingsMtx_userID_dev, CUSPARSE_INDEX_BASE_ZERO); 
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "Conversion from COO to CSR format failed\n");
        return 1; 
    } 

    

    if(Debug){
        // save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev, ratings_rows + 1, "csr_format_ratingsMtx_userID_dev");
        // LOG("csr_format_ratingsMtx_userID_dev : ");
        // print_gpu_array_entries<int>(csr_format_ratingsMtx_userID_dev, ratings_rows + 1, 1 , ratings_rows + 1);
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }

    long long int num_entries_keyWord_mtx_temp = (long long int)0;
    long long int num_keyWords_temp            = (long long int)0;

    
    int*   coo_format_keyWordMtx_itemID_host  = NULL;
    int*   coo_format_keyWordMtx_keyWord_host = NULL;
    int*   coo_format_keyWordMtx_itemID_dev   = NULL;
    int*   coo_format_keyWordMtx_keyWord_dev  = NULL;
    if(Content_Based){
        CSVReader csv_keyWords(csv_keyWords_path);
        LOG("Here!") ;
        //num_entries_keyWord_mtx_temp = csv_keyWords.num_entries();
        switch (case_)
        {
            case 1:{ // code to be executed if n = 1;
                //Dataset_Name = "MovieLens 20 million";
                num_entries_keyWord_mtx_temp = ratings_cols;
                break;
            }case 2:{ // code to be executed if n = 2;
                //Dataset_Name = "Rent The Runaway";
                //num_entries_keyWord_mtx_temp = 48087;
                break;
            }default: 
                ABORT_IF_EQ(0, 1, "no valid dataset selected");
        }
        //num_entries_keyWord_mtx = 48087;
        LOG("num_entries_keyWord_mtx : "<<num_entries_keyWord_mtx_temp);

        if(num_entries_keyWord_mtx_temp <= 0 ){
            return 0;
        }

        coo_format_keyWordMtx_itemID_host  = (int *)malloc(num_entries_keyWord_mtx_temp *  sizeof(int)); 
        coo_format_keyWordMtx_keyWord_host = (int *)malloc(num_entries_keyWord_mtx_temp *  sizeof(int)); 
        checkErrors(coo_format_keyWordMtx_itemID_host);
        checkErrors(coo_format_keyWordMtx_keyWord_host);

        LOG("Attempting to collect Content Based Information.");
        num_keyWords_temp  = csv_keyWords.makeContentBasedcooKeyWordMtx(coo_format_keyWordMtx_itemID_host,
                                                                        coo_format_keyWordMtx_keyWord_host,
                                                                        num_entries_keyWord_mtx_temp);
        LOG("num_keyWords : "<<num_keyWords_temp);
        if(num_keyWords_temp <= 0 ){
            return 0;
        }

        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_itemID_dev,   num_entries_keyWord_mtx_temp * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_keyWord_dev,  num_entries_keyWord_mtx_temp * sizeof(int)));
        update_Mem(2 * num_entries_keyWord_mtx_temp * sizeof(int) );

        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_itemID_dev,   coo_format_keyWordMtx_itemID_host,   num_entries_keyWord_mtx_temp * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_keyWord_dev,  coo_format_keyWordMtx_keyWord_host,  num_entries_keyWord_mtx_temp * sizeof(int), cudaMemcpyHostToDevice));
        free(coo_format_keyWordMtx_itemID_host);
        free(coo_format_keyWordMtx_keyWord_host);
    }
    const long long int num_entries_keyWord_mtx = num_entries_keyWord_mtx_temp;
    const long long int num_keyWords            = num_keyWords_temp;




    int*   csr_format_keyWordMtx_itemID_dev;
    if(Content_Based){
        checkCudaErrors(cudaMalloc((void**)&csr_format_keyWordMtx_itemID_dev, (ratings_cols + 1) * sizeof(int)));
        update_Mem( (ratings_cols + 1) * sizeof(int) );

        cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_keyWordMtx_itemID_dev, num_entries_keyWord_mtx, 
                                           ratings_cols, csr_format_keyWordMtx_itemID_dev, CUSPARSE_INDEX_BASE_ZERO); 
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
            fprintf(stdout, "Conversion from COO to CSR format failed\n");
            return 1; 
        } 
        cudaFree(coo_format_keyWordMtx_itemID_dev);           update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
    }














    //============================================================================================
    // split the data into testing data and training data
    //============================================================================================

    










    const float probability_GU       = (float)10.0/(float)100.0;
    const float p                    = (float)1.0/(float)10.0;
    const float probability_testing  = random_initialization ?       p        :        p        * ((float)1.0 - probability_GU);
    const float probability_training = random_initialization ? (float)1.0 - p :((float)1.0 - p) * ((float)1.0 - probability_GU);
    LOG("percentage of users for testing: " <<probability_testing);
    LOG("percentage of users for training: "<<probability_training);
    if(!random_initialization) {
        LOG("percentage of users for GA: "      <<(float)1.0 - probability_training - probability_testing<<std::endl);
    }

    long long int ratings_rows_GU_temp       = (long long int)(probability_GU * (float)ratings_rows);

    ABORT_IF_LE(probability_GU, (float)0.0, "probability_GU <= 0");
    ABORT_IF_LE(probability_testing, (float)0.0, "probability_testing <= 0");
    ABORT_IF_LE(probability_training, (float)0.0, "probability_training <= 0");



    const int num_groups =  random_initialization ? 2 : 3;
    float  probability_of_groups_host [num_groups];
    probability_of_groups_host[0] = probability_testing;
    probability_of_groups_host[1] = probability_training;
    if(!random_initialization){
        probability_of_groups_host[2] = (float)1.0 - probability_training - probability_testing;
    }

    float* probability_of_groups_dev;
    checkCudaErrors(cudaMalloc((void**)&probability_of_groups_dev, num_groups * sizeof(float)));
    update_Mem( num_groups * sizeof(float) );
    checkCudaErrors(cudaMemcpy(probability_of_groups_dev, probability_of_groups_host, num_groups * sizeof(float), cudaMemcpyHostToDevice));

    int *group_indicies;
    checkCudaErrors(cudaMalloc((void**)&group_indicies, ratings_rows * sizeof(int)));
    update_Mem( ratings_rows * sizeof(int) );

    gpu_get_rand_groups(ratings_rows,  group_indicies, probability_of_groups_dev, 3);

    int* group_sizes = NULL;
    group_sizes = (int *)malloc(num_groups * sizeof(int)); 
    checkErrors(group_sizes);

    count_each_group_from_coo(num_groups, group_indicies, num_entries, coo_format_ratingsMtx_userID_dev, group_sizes);
    const long long int num_entries_testing   = group_sizes[0];
    const long long int num_entries_training  = group_sizes[1];
    const long long int num_entries_GU        = (long long int)0 ? ratings_rows_GU_temp : group_sizes[2];

    count_each_group(ratings_rows, group_indicies, group_sizes, num_groups);
    const long long int ratings_rows_testing  = group_sizes[0];
    const long long int ratings_rows_training = group_sizes[1];
    const long long int ratings_rows_GU       = random_initialization ? ratings_rows_GU_temp : group_sizes[2];
    
    LOG("num testing users : "   <<ratings_rows_testing);
    LOG("num training users : "  <<ratings_rows_training);
    LOG("num GU users : "        <<ratings_rows_GU);
    LOG("num testing entries : " <<num_entries_testing);
    LOG("num training entries : "<<num_entries_training);
    LOG("num GU entries : "      <<num_entries_GU<<std::endl);
    
    if(!random_initialization){
        ABORT_IF_NEQ(ratings_rows_testing + ratings_rows_training + ratings_rows_GU, ratings_rows, "The number of rows does not add up correctly.");
        ABORT_IF_NEQ(num_entries_testing  + num_entries_training  + num_entries_GU,  num_entries, "The number of entries does not add up correctly.");
    }else{
        ABORT_IF_NEQ(ratings_rows_testing + ratings_rows_training, ratings_rows, "The number of rows does not add up correctly.");
        ABORT_IF_NEQ(num_entries_testing  + num_entries_training ,  num_entries, "The number of entries does not add up correctly.");        
    }
    ABORT_IF_LE(ratings_rows_GU, (long long int)0, "ratings_rows_GU <= 0");
    ABORT_IF_LE(ratings_rows_testing, (long long int)0, "ratings_rows_testing <= 0");
    ABORT_IF_LE(ratings_rows_training, (long long int)0, "ratings_rows_training <= 0");    

    if(Debug && 0){
        // LOG("coo_format_ratingsMtx_userID_dev : ");
        // print_gpu_array_entries<int>(coo_format_ratingsMtx_userID_dev, 100, 1 , num_entries);
        LOG("group_indicies :");
        print_gpu_mtx_entries<int>(group_indicies, ratings_rows, 1 );
        //save_device_array_to_file<int>(group_indicies, ratings_rows, "testing_bools");
        LOG("Press Enter to continue.") ;
        std::cin.ignore();
    }
    
    cudaFree(coo_format_ratingsMtx_userID_dev);
    update_Mem( num_entries * sizeof(int) * (-1) );

    int*   csr_format_ratingsMtx_userID_dev_testing;
    int*   coo_format_ratingsMtx_itemID_dev_testing;
    float* coo_format_ratingsMtx_rating_dev_testing;

    int*   csr_format_ratingsMtx_userID_dev_training;
    int*   coo_format_ratingsMtx_itemID_dev_training;
    float* coo_format_ratingsMtx_rating_dev_training;

    int*   csr_format_ratingsMtx_userID_dev_GU;
    int*   coo_format_ratingsMtx_itemID_dev_GU;
    float* coo_format_ratingsMtx_rating_dev_GU;

    
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing        * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing        * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_training,  (ratings_rows_training + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_training,  num_entries_training        * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_training,  num_entries_training        * sizeof(float)));

    if(!random_initialization){
        checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_GU,  (ratings_rows_GU + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_GU,  num_entries_GU        * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_GU,  num_entries_GU        * sizeof(float)));
    }
    update_Mem(  (ratings_rows_testing + 1)  * sizeof(int) + num_entries_testing  * sizeof(int) + num_entries_testing  * sizeof(float)
               + (ratings_rows_training + 1) * sizeof(int) + num_entries_training * sizeof(int) + num_entries_training * sizeof(float)
               + (ratings_rows_GU + 1)       * sizeof(int) + num_entries_GU       * sizeof(int) + num_entries_GU       * sizeof(float)  );
    
    int*   csr_format_ratingsMtx_userID_dev_by_group_host  [num_groups];
    csr_format_ratingsMtx_userID_dev_by_group_host[0] = csr_format_ratingsMtx_userID_dev_testing;
    csr_format_ratingsMtx_userID_dev_by_group_host[1] = csr_format_ratingsMtx_userID_dev_training;
    if(!random_initialization){
        csr_format_ratingsMtx_userID_dev_by_group_host[2] = csr_format_ratingsMtx_userID_dev_GU;
    }
    int*   coo_format_ratingsMtx_itemID_dev_by_group_host  [num_groups];
    coo_format_ratingsMtx_itemID_dev_by_group_host[0] = coo_format_ratingsMtx_itemID_dev_testing;
    coo_format_ratingsMtx_itemID_dev_by_group_host[1] = coo_format_ratingsMtx_itemID_dev_training;
    if(!random_initialization){
        coo_format_ratingsMtx_itemID_dev_by_group_host[2] = coo_format_ratingsMtx_itemID_dev_GU;
    }
    float*   coo_format_ratingsMtx_rating_dev_by_group_host  [num_groups];
    coo_format_ratingsMtx_rating_dev_by_group_host[0] = coo_format_ratingsMtx_rating_dev_testing;
    coo_format_ratingsMtx_rating_dev_by_group_host[1] = coo_format_ratingsMtx_rating_dev_training;
    if(!random_initialization){
        coo_format_ratingsMtx_rating_dev_by_group_host[2] = coo_format_ratingsMtx_rating_dev_GU;
    }
    int ratings_rows_by_group_host[num_groups];
    ratings_rows_by_group_host[0] = ratings_rows_testing;
    ratings_rows_by_group_host[1] = ratings_rows_training;
    if(!random_initialization){
        ratings_rows_by_group_host[2] = ratings_rows_GU;
    }
    int**   csr_format_ratingsMtx_userID_dev_by_group;
    int**   coo_format_ratingsMtx_itemID_dev_by_group;
    float** coo_format_ratingsMtx_rating_dev_by_group;
    int*    ratings_rows_by_group;

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_by_group,  num_groups*sizeof(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_by_group,  num_groups*sizeof(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_by_group,  num_groups*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void**)&ratings_rows_by_group,                      num_groups*sizeof(int)));
    update_Mem( num_groups * sizeof(int*) * 2 + num_groups * sizeof(float*) + num_groups * sizeof(int) );

    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_by_group,  csr_format_ratingsMtx_userID_dev_by_group_host,  num_groups * sizeof(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_by_group,  coo_format_ratingsMtx_itemID_dev_by_group_host,  num_groups * sizeof(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_by_group,  coo_format_ratingsMtx_rating_dev_by_group_host,  num_groups * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ratings_rows_by_group,                      ratings_rows_by_group_host,                      num_groups * sizeof(int),    cudaMemcpyHostToDevice));
    
    gpu_split_data(csr_format_ratingsMtx_userID_dev,
                   coo_format_ratingsMtx_itemID_dev,
                   coo_format_ratingsMtx_rating_dev, 
                   ratings_rows, group_indicies,
                   csr_format_ratingsMtx_userID_dev_by_group,
                   coo_format_ratingsMtx_itemID_dev_by_group,
                   coo_format_ratingsMtx_rating_dev_by_group,
                   ratings_rows_by_group); 
    if(Debug && 0){
        if(!random_initialization){
            save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_GU,        ratings_rows_GU + 1,       "csr_format_ratingsMtx_userID_dev_GU");
            save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_GU,        num_entries_GU,            "coo_format_ratingsMtx_itemID_dev_GU");
            save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_GU,        num_entries_GU,            "coo_format_ratingsMtx_rating_dev_GU");
        }
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_dev_training");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_training,  num_entries_training,      "coo_format_ratingsMtx_itemID_dev_training");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training");
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,   ratings_rows_testing + 1,  "csr_format_ratingsMtx_userID_dev_testing");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing,   num_entries_testing,       "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,   num_entries_testing,       "coo_format_ratingsMtx_rating_dev_testing");
        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_GU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_GU, ratings_rows_GU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }


    
    cudaFree(csr_format_ratingsMtx_userID_dev);            update_Mem((ratings_rows + 1) * sizeof(int) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev);            update_Mem(num_entries * sizeof(int) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev);            update_Mem(num_entries * sizeof(float) * (-1));
    cudaFree(csr_format_ratingsMtx_userID_dev_by_group);   update_Mem(num_groups * sizeof(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev_by_group);   update_Mem(num_groups * sizeof(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev_by_group);   update_Mem(num_groups * sizeof(float*) * (-1));
    cudaFree(ratings_rows_by_group);                       update_Mem(num_groups * sizeof(int) * (-1));
    cudaFree(probability_of_groups_dev);                   update_Mem(num_groups * sizeof(float) * (-1));
    cudaFree(group_indicies);                              update_Mem(num_groups * sizeof(int)* (-1));
    
    free(group_sizes);






    






    //============================================================================================
    // collect User Means and Variances
    //============================================================================================










    LOG("collect User Means and Variance... ");

    float* user_means_training;
    float* user_means_testing;
    float* user_means_GU;
    checkCudaErrors(cudaMalloc((void**)&user_means_training, ratings_rows_training * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&user_means_testing,  ratings_rows_testing  * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&user_means_GU,       ratings_rows_GU       * sizeof(float)));
    update_Mem((ratings_rows_training + ratings_rows_testing + ratings_rows_GU)* sizeof(float));

    float* user_var_training;
    float* user_var_testing;
    float* user_var_GU;
    checkCudaErrors(cudaMalloc((void**)&user_var_training, ratings_rows_training * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&user_var_testing,  ratings_rows_testing  * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&user_var_GU,       ratings_rows_GU       * sizeof(float)));
    update_Mem((ratings_rows_training + ratings_rows_testing + ratings_rows_GU)* sizeof(float));

    LOG("Here!") ;
    collect_user_means(user_means_training, user_var_training,  (long long int)ratings_rows_training,
                       csr_format_ratingsMtx_userID_dev_training,
                       coo_format_ratingsMtx_rating_dev_training,
                       user_means_testing, user_var_testing,    (long long int)ratings_rows_testing,
                       csr_format_ratingsMtx_userID_dev_testing,
                       coo_format_ratingsMtx_rating_dev_testing,
                       user_means_GU, user_var_GU,              random_initialization? (long long int)0 : (long long int)ratings_rows_GU,
                       csr_format_ratingsMtx_userID_dev_GU,
                       coo_format_ratingsMtx_rating_dev_GU);
    LOG("Here!") ;
    if(Debug && 0){
        save_device_array_to_file<float>(user_means_testing,  ratings_rows_testing,  "user_means_testing");
        save_device_array_to_file<float>(user_var_testing,    ratings_rows_testing,  "user_var_testing");
        save_device_array_to_file<float>(user_means_training, ratings_rows_training, "user_means_training");
        save_device_array_to_file<float>(user_var_training,   ratings_rows_training, "user_var_training");
        if(!random_initialization){
            save_device_array_to_file<float>(user_means_GU,       ratings_rows_GU,       "user_means_GU");
            save_device_array_to_file<float>(user_var_GU,         ratings_rows_GU,       "user_var_GU");
        }
        // LOG("user_means_training : ");
        // print_gpu_array_entries(user_means_GU, ratings_rows_GU);
        // LOG("user_means_training : ");
        // print_gpu_array_entries(user_means_GU, ratings_rows_GU);
        // LOG("user_means_GU : ");
        // print_gpu_array_entries(user_means_GU, ratings_rows_GU);
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }





    





    //============================================================================================
    // Center and Fill Training Data
    //============================================================================================










    LOG("Center Data and fill GA matrix... ");


    float * coo_format_ratingsMtx_row_centered_rating_dev_GU;
    float * coo_format_ratingsMtx_row_centered_rating_dev_testing;
    float * coo_format_ratingsMtx_row_centered_rating_dev_training;
    
    if(!random_initialization){
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_GU,       num_entries_GU       * sizeof(float)));
    }
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_testing,  num_entries_testing  * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_training, num_entries_training * sizeof(float)));
    update_Mem( ( num_entries_GU + num_entries_testing + num_entries_training) * sizeof(float) );




    //const float val_when_var_is_zero = (float)3.5774;        // use for MovieLens
    const float val_when_var_is_zero = (float)0.5;        // use for Rent The Runway
    LOG("rating used when the variance of the user's ratings is zero : "<< val_when_var_is_zero);

    if(!random_initialization){
        center_ratings(user_means_GU, user_var_GU, 
                       ratings_rows_GU, num_entries_GU,
                        csr_format_ratingsMtx_userID_dev_GU,
                        coo_format_ratingsMtx_rating_dev_GU,
                        coo_format_ratingsMtx_row_centered_rating_dev_GU, val_when_var_is_zero);

        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_GU,       coo_format_ratingsMtx_row_centered_rating_dev_GU,       num_entries_GU       *  sizeof(float), cudaMemcpyDeviceToDevice));
        cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_GU);             update_Mem((ratings_rows_GU + 1) * sizeof(int) * (-1));
        float range_GU       = gpu_range<float>(num_entries_GU,        coo_format_ratingsMtx_rating_dev_GU);
    }

    center_ratings(user_means_testing, user_var_testing, 
                   ratings_rows_testing, num_entries_testing,
                    csr_format_ratingsMtx_userID_dev_testing,
                    coo_format_ratingsMtx_rating_dev_testing,
                    coo_format_ratingsMtx_row_centered_rating_dev_testing, val_when_var_is_zero);

    center_ratings(user_means_training, user_var_training, 
                   ratings_rows_training, num_entries_training,
                    csr_format_ratingsMtx_userID_dev_training,
                    coo_format_ratingsMtx_rating_dev_training,
                    coo_format_ratingsMtx_row_centered_rating_dev_training, val_when_var_is_zero);

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_testing,  coo_format_ratingsMtx_row_centered_rating_dev_testing,  num_entries_testing  *  sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_training, coo_format_ratingsMtx_row_centered_rating_dev_training, num_entries_training *  sizeof(float), cudaMemcpyDeviceToDevice));
    
    cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_testing);        update_Mem(num_entries_GU        * sizeof(int) * (-1));
    cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_training);       update_Mem(num_entries_GU        * sizeof(float) * (-1));

    
    float range_testing  = gpu_range<float>(num_entries_testing,   coo_format_ratingsMtx_rating_dev_testing);
    float range_training = gpu_range<float>         (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float min_training   = gpu_min<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float max_training   = gpu_max<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float abs_max_training = std::max(max_training, std::abs(min_training));
    const float expt_training  = gpu_expected_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float expt_abs_training  = gpu_expected_abs_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);

    range_training = (float)2.0 * expt_abs_training;

    //LOG(std::endl<<"range_GU = "         <<range_GU) ;
    //LOG("range_testing = "    <<range_testing) ;
    //LOG("range_training = "   <<range_training) ;
    LOG("max_training = "     <<max_training) ;
    LOG("min_training = "     <<min_training) ;
    LOG("abs_max_training = " <<abs_max_training) ;
    LOG("expt_training = "    <<expt_training) ;
    LOG("expt_abs_training = "<<expt_abs_training) ;
   
    if( Debug && 0){
        if(!random_initialization){
            save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_GU,        ratings_rows_GU + 1,       "csr_format_ratingsMtx_userID_dev_GU");
            save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_GU,        num_entries_GU,            "coo_format_ratingsMtx_itemID_dev_GU");
            save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_GU,        num_entries_GU,            "coo_format_ratingsMtx_rating_dev_GU");
        }

    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training_post_centering");
    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_GU,  num_entries_GU,      "coo_format_ratingsMtx_rating_dev_GU_post_centering");
    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing_post_centering");

        // save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_dev_testing_post_centering");
        // save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing,      "coo_format_ratingsMtx_itemID_dev_testing_post_centering");
        // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing_post_centering");
        
        // save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_training, coo_format_ratingsMtx_rating_dev_training, 
        //                                         num_entries_training, "training_users_cols_rating_centered");
        // save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_dev_training_post_centering" );
        // save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_training, num_entries_training,      "coo_format_ratingsMtx_itemID_dev_training_post_centering");
        // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training_post_centering" );

        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_GU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_GU, ratings_rows_GU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }





    bool row_major_ordering = true;

    float * full_ratingsMtx_dev_GU;
    float * full_ratingsMtx_host_GU = NULL;

    const long long int GU_mtx_size = (long long int)ratings_rows_GU * (long long int)ratings_cols;
    const long long int GU_mtx_size_bytes = (long long int)ratings_rows_GU * (long long int)ratings_cols * (long long int)sizeof(float);
    LOG(std::endl<<"Will need "<<GU_mtx_size<< " floats for the GU mtx.") ;
    LOG("Will need "<<GU_mtx_size_bytes<< " bytes for the GU mtx.") ;
    if(allocatedMem + GU_mtx_size_bytes > (long long int)((double)devMem * (double)0.75)){
        LOG("Conserving Memory Now");
        Conserve_GPU_Mem = 1;
    }
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();


    int*   csr_format_ratingsMtx_userID_host_GU = NULL;
    int*   coo_format_ratingsMtx_itemID_host_GU = NULL;
    float* coo_format_ratingsMtx_rating_host_GU = NULL;

    if(Conserve_GPU_Mem){


        full_ratingsMtx_host_GU = (float *)malloc(GU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_GU);

        //cpu_set_all<float>(full_ratingsMtx_host_GU, GU_mtx_size, (float)0.0);
        host_rng_uniform(ratings_rows_GU * ratings_cols, min_training, max_training, full_ratingsMtx_dev_GU);

        if(!random_initialization){
            csr_format_ratingsMtx_userID_host_GU  = (int *)  malloc((ratings_rows_GU + 1) * sizeof(int)  );
            coo_format_ratingsMtx_itemID_host_GU  = (int *)  malloc(num_entries_GU        * sizeof(int)  );
            coo_format_ratingsMtx_rating_host_GU  = (float *)malloc(num_entries_GU        * sizeof(float));


            checkErrors(csr_format_ratingsMtx_userID_host_GU);
            checkErrors(coo_format_ratingsMtx_itemID_host_GU);
            checkErrors(coo_format_ratingsMtx_rating_host_GU);
            checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_GU,  csr_format_ratingsMtx_userID_dev_GU,  (ratings_rows_GU + 1) * sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_GU,  coo_format_ratingsMtx_itemID_dev_GU,  num_entries_GU        * sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_GU,  coo_format_ratingsMtx_rating_dev_GU,  num_entries_GU        * sizeof(float), cudaMemcpyDeviceToHost));
            
            cpu_fill_training_mtx((long long int)ratings_rows_GU, (long long int)ratings_cols, (long long int)num_entries_GU, 
                                  row_major_ordering,  
                                  csr_format_ratingsMtx_userID_host_GU,
                                  coo_format_ratingsMtx_itemID_host_GU,
                                  coo_format_ratingsMtx_rating_host_GU,
                                  full_ratingsMtx_host_GU);

            cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols, 
                                     row_major_ordering, full_ratingsMtx_host_GU, 1);

            free(csr_format_ratingsMtx_userID_host_GU);
            free(coo_format_ratingsMtx_itemID_host_GU);
            free(coo_format_ratingsMtx_rating_host_GU); 
        }

      

        


        LOG("full_ratingsMtx_host_GU filled and shuffled") ;
    }else{
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_GU, ratings_rows_GU * ratings_cols * sizeof(float)));
        update_Mem(ratings_rows_GU * ratings_cols * sizeof(float));


        //gpu_set_all<float>(full_ratingsMtx_dev_GU, ratings_rows_GU * ratings_cols, (float)0.0);
        //gpu_rng_uniform<float>(dn_handle, ratings_rows_GU * ratings_cols, min_training, max_training, full_ratingsMtx_dev_GU);
        gpu_rng_gaussian<float>(ratings_rows_GU * ratings_cols, (float)0.0, (float)1.0, full_ratingsMtx_dev_GU);

        if(!random_initialization){
            gpu_fill_training_mtx(ratings_rows_GU, ratings_cols, row_major_ordering,
                                  csr_format_ratingsMtx_userID_dev_GU,
                                  coo_format_ratingsMtx_itemID_dev_GU,
                                  coo_format_ratingsMtx_rating_dev_GU,
                                  full_ratingsMtx_dev_GU);

            if(Content_Based){
                gpu_supplement_training_mtx_with_content_based(ratings_rows_GU, 
                                                                ratings_cols, 
                                                                row_major_ordering,
                                                                csr_format_ratingsMtx_userID_dev_GU,
                                                                coo_format_ratingsMtx_itemID_dev_GU,
                                                                coo_format_ratingsMtx_rating_dev_GU,
                                                                full_ratingsMtx_dev_GU,
                                                                csr_format_keyWordMtx_itemID_dev,
                                                                coo_format_keyWordMtx_keyWord_dev);

                const float max_training_supplement   = gpu_max<float>(ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
                if (max_training_supplement > max_training) {
                    LOG("max_training_supplement : "<<max_training_supplement);
                }
                const float min_training_supplement   = gpu_min<float>(ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
                if (min_training_supplement < min_training) {
                    LOG("min_training_supplement : "<<min_training_supplement);
                }
            }

            if(Debug && 0){
                save_device_mtx_to_file(full_ratingsMtx_dev_GU, ratings_cols, ratings_rows_GU, "full_ratingsMtx_dev_GU_pre_shuffle", true);
            }

            cudaDeviceSynchronize();
            //shuffle GA rows
            gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols,  
                                         row_major_ordering, full_ratingsMtx_dev_GU, 1);
        }
        if(Debug && 0){
            save_device_mtx_to_file(full_ratingsMtx_dev_GU, ratings_cols, ratings_rows_GU, "full_ratingsMtx_dev_GU", true);
            // LOG("Press Enter to continue.") ;
            // std::cin.ignore();
        }
        LOG("full_ratingsMtx_dev_GU filled and shuffled"<<std::endl) ;

        const float min_GU       = gpu_min<float>               (ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
        const float max_GU       = gpu_max<float>               (ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
        const float abs_max_GU   = std::max(max_GU, std::abs(min_GU));
        const float expt_GU      = gpu_expected_value<float>    (ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
        const float expt_abs_GU  = gpu_expected_abs_value<float>(ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);


        LOG("max_GU = "     <<max_GU) ;
        LOG("min_GU = "     <<min_GU) ;
        LOG("abs_max_GU = " <<abs_max_GU) ;
        LOG("expt_GU = "    <<expt_GU) ;
        LOG("expt_abs_GU = "<<expt_abs_GU) ;

    }
    if(!random_initialization){
        cudaFree(csr_format_ratingsMtx_userID_dev_GU);         update_Mem((ratings_rows_GU + 1) * sizeof(int)   * (-1));
        cudaFree(coo_format_ratingsMtx_itemID_dev_GU);         update_Mem(num_entries_GU        * sizeof(int)   * (-1));
        cudaFree(coo_format_ratingsMtx_rating_dev_GU);         update_Mem(num_entries_GU        * sizeof(float) * (-1));
    }

    if(Debug && 0){
        save_device_array_to_file<int>(  csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_dev_testing" );
        save_device_array_to_file<int>(  coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing" );
        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_GU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_GU, ratings_rows_GU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }




    float* user_means_testing_host = NULL;
    float* user_var_testing_host   = NULL;
    user_means_testing_host        = (float *)malloc(ratings_rows_testing *  sizeof(float)); 
    user_var_testing_host          = (float *)malloc(ratings_rows_testing *  sizeof(float)); 
    checkErrors(user_means_testing_host);
    checkErrors(user_var_testing_host);
    checkCudaErrors(cudaMemcpy(user_means_testing_host, user_means_testing, ratings_rows_testing *  sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(user_var_testing_host,   user_var_testing,   ratings_rows_testing *  sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(user_means_testing);                        update_Mem(ratings_rows_testing * sizeof(float) * (-1));
    cudaFree(user_var_testing);                          update_Mem(ratings_rows_testing * sizeof(float) * (-1));

    //the stored means are useless now
    // cudaFree(user_means_GU);                             update_Mem(ratings_rows_GU * sizeof(float) * (-1));
    // cudaFree(user_var_GU);                               update_Mem(ratings_rows_GU * sizeof(float) * (-1));
    cudaFree(user_means_training);                       update_Mem(ratings_rows_training * sizeof(float) * (-1));
    cudaFree(user_var_training);                         update_Mem(ratings_rows_training * sizeof(float) * (-1));


    //============================================================================================
    // Conserve Memory
    //============================================================================================

    // int *   csr_format_ratingsMtx_userID_host_training  = NULL;
    // int *   coo_format_ratingsMtx_itemID_host_training = NULL;
    // float * coo_format_ratingsMtx_rating_host_training  = NULL;

    // csr_format_ratingsMtx_userID_host_training  = (int *)  malloc(ratings_rows_training *  sizeof(int)); 
    // coo_format_ratingsMtx_itemID_host_training = (int *)  malloc(num_entries_training  *  sizeof(int)); 
    // coo_format_ratingsMtx_rating_host_training  = (float *)malloc(num_entries_training  *  sizeof(float));
    // checkErrors(csr_format_ratingsMtx_userID_host_training);
    // checkErrors(coo_format_ratingsMtx_itemID_host_training);
    // checkErrors(coo_format_ratingsMtx_rating_host_training); 

    // checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_training,  csr_format_ratingsMtx_userID_dev_training,  ratings_rows_training *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_training, coo_format_ratingsMtx_itemID_dev_training, num_entries_training  *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_training,  coo_format_ratingsMtx_rating_dev_training,  num_entries_training  *  sizeof(float), cudaMemcpyDeviceToHost));

    // cudaFree(csr_format_ratingsMtx_userID_dev_training);
    // cudaFree(coo_format_ratingsMtx_itemID_dev_training);
    // cudaFree(coo_format_ratingsMtx_rating_dev_training);
    
    // int *   csr_format_ratingsMtx_userID_host_testing  = NULL;
    // int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    // float * coo_format_ratingsMtx_rating_host_testing  = NULL;

    // csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc(ratings_rows_testing *  sizeof(int)); 
    // coo_format_ratingsMtx_itemID_host_testing = (int *)  malloc(num_entries_testing  *  sizeof(int)); 
    // coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  sizeof(float)); 
    // checkErrors(csr_format_ratingsMtx_userID_host_testing);
    // checkErrors(coo_format_ratingsMtx_itemID_host_testing);
    // checkErrors(coo_format_ratingsMtx_rating_host_testing); 

    // checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  sizeof(float), cudaMemcpyDeviceToHost));

    // cudaFree(csr_format_ratingsMtx_userID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_itemID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_rating_dev_testing);


    //============================================================================================
    // We want to find orthogonal matrices U, V such that R ~ U*V^T
    // 
    // R is batch_size_GU by ratings_cols
    // U is batch_size_GU by num_latent_factors
    // V is ratings_cols by num_latent_factors
    //============================================================================================
    LOG(std::endl);

    bool        print_training_error    = true;


    float training_rate;      
    float regularization_constant;         

    const float testing_fraction            = 0.25; //percent of known entries used for testing
    bool        compress                    = false;
    bool        compress_when_testing       = false;
    bool        regularize_U                = true;
    bool        regularize_R                = true;
    bool        regularize_R_distribution   = true;
    bool        normalize_V_rows            = false;

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            //Dataset_Name = "MovieLens 20 million";
            training_rate               = (float)0.1;      //use for movielens
            regularization_constant     = (float)0.1;         //use for movielens
            regularize_U                = true;
            regularize_R                = true;
            regularize_R_distribution   = true;
            normalize_V_rows            = false;
            compress_when_testing       = true;
            break;
        }case 2:{ // code to be executed if n = 2;
            //Dataset_Name = "Rent The Runaway";
            training_rate           = (float)0.01;      //use for rent the runway
            regularization_constant = (float)1.0;         //use for rent the runway
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }

    const int num_iterations = 10000;
    const int num_batches    = 500;
    const int testing_rate   = 1;

    LOG("training_rate : "          <<training_rate);
    LOG("regularization_constant : "<<regularization_constant);
    LOG("testing_fraction : "       <<testing_fraction);
    LOG("regularize_U : "           <<regularize_U);
    LOG("regularize_R : "           <<regularize_R);
    LOG("regularize_R_distribution : " <<regularize_R_distribution);
    LOG("compress : "               <<compress);
    LOG("compress_when_testing : "  <<compress_when_testing);
    LOG("num_iterations : "         <<num_iterations);
    LOG("num_batches : "            <<num_batches);
    LOG("testing_rate : "           <<testing_rate);

    float * testing_error = NULL;
    testing_error = (float *)malloc((num_iterations / testing_rate) * sizeof(float)); 
    checkErrors(testing_error);

    const long long int batch_size_training = std::max((long long int)1, ratings_rows_training / num_batches);
    const long long int batch_size_GU       = ratings_rows_GU; 
    //const long long int batch_size_GU       = std::min((long long int)100, ratings_rows_GU);
    const long long int batch_size_testing  = std::min((long long int)200, ratings_rows_testing);
    LOG(std::endl);
    LOG("batch_size_testing : " <<batch_size_testing);
    LOG("batch_size_training : "<<batch_size_training);
    LOG("batch_size_GU : "      <<batch_size_GU);
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

    ABORT_IF_LE(batch_size_training, (long long int)1, "batch size training is small");
    ABORT_IF_LE(batch_size_testing, (long long int)1, "batch size testinging is small");
    ABORT_IF_LE(batch_size_GU, (long long int)1, "batch size GU is small");

    long long int num_batches_training = ratings_rows_training / batch_size_training;
    long long int num_batches_GU       = ratings_rows_GU       / batch_size_GU;
    long long int num_batches_testing  = ratings_rows_testing  / batch_size_testing;
    

    long long int num_latent_factors = (long long int)((float)batch_size_GU * (float)0.95);
    const float percent              = (float)0.95;


    float * U_GU;       // U_GU is ratings_rows_GU * ratings_rows_GU
    float * U_training;
    float * U_testing;
    float * V;          // V_GU is ratings_cols * ratings_cols
    float * R_training;
    float * R_testing;
    if(Debug && 0){
        // const int batch_size_GU = 3;
        // const int ratings_cols = 2;
        // /*       | 1 2  |
        //  *   A = | 4 5  |
        //  *       | 2 1  |
        //  */
        // float A[batch_size_GU*ratings_cols] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
        // cudaMemcpy(full_ratingsMtx_dev_GU, A, sizeof(float)*batch_size_GU*ratings_cols, cudaMemcpyHostToDevice);
    }

    long long int min_ = std::min(batch_size_GU, ratings_cols);
    bool temp = Conserve_GPU_Mem;
    const long long int Training_bytes = (batch_size_GU       * min_ +
                                          batch_size_training * std::max(min_, batch_size_training) + 
                                          batch_size_testing  * std::max(min_, batch_size_testing) + 
                                          ratings_cols        * min_ +
                                          ratings_cols        * batch_size_training +
                                          ratings_cols        * batch_size_testing)* (long long int)sizeof(float) ;
    if(allocatedMem + Training_bytes > (long long int)((double)devMem * (double)0.75)) 
    {
        Conserve_GPU_Mem = 1;
    };


    if(!temp && Conserve_GPU_Mem){
        LOG("Conserving Memory Now");
        //put the GA ratings mtx on the CPU;
        full_ratingsMtx_host_GU = (float *)malloc(GU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_GU);
        checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_GU, full_ratingsMtx_dev_GU, GU_mtx_size_bytes, cudaMemcpyDeviceToHost));
        cudaFree(full_ratingsMtx_dev_GU);
        update_Mem((float)(-1.0) * GU_mtx_size_bytes );
    };
    if(Conserve_GPU_Mem){
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_GU, batch_size_GU * ratings_cols * sizeof(float)));
        update_Mem(batch_size_GU * ratings_cols* sizeof(float) );
    }
    
    bool swapped = false;  
    
    // LOG(ratings_cols * ratings_cols * sizeof(float)) ;
    checkCudaErrors(cudaMalloc((void**)&U_GU,       batch_size_GU       * min_                       * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * min_                       * sizeof(float)));
    // cudaDeviceSynchronize();
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();  


    checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training * std::max(min_, batch_size_training) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training * ratings_cols                        * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_, batch_size_testing)  * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * sizeof(float)));
    update_Mem(Training_bytes);












    float* old_R_GU;
    long long int total_testing_iterations = (long long int) 10000;

    checkCudaErrors(cudaMalloc((void**)&old_R_GU, batch_size_GU * ratings_cols * sizeof(float)));










    //============================================================================================
    // Begin Training
    //============================================================================================  
    LOG(std::endl<<std::endl<<"                              Begin Training..."<<std::endl); 
    gettimeofday(&training_start, NULL);
    int num_tests = 0;
    int count_GU_rounds = 0;
    for(int it = 0; it < num_iterations; it ++){
        if(training_rate < (float)0.00001) break;
        float testing_error_temp = (float)0.0;
        float training_error_temp = testing_error_temp;
        long long int total_training_nnz = (long long int)0;
        long long int total_testing_nnz = (long long int)0;
        int count_tests = 0;
        for(int batch = 0; batch < num_batches; batch ++){
            if(training_rate < (float)0.00001) break;
            if( print_training_error){
                //LOG(std::endl<<"                                       ~ITERATION "<<it<<", BATCH "<<batch<<"~"<<std::endl);
                LOG(std::endl<<"                              ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch<<" ( / "<<num_batches<<" )");
            }
            if(Debug && 0){
                //LOG(std::endl<<"                              ITERATION "<<it<<", BATCH "<<batch);
                LOG(std::endl<<"                              ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch<<" ( / "<<num_batches<<" )");
            }
            
            long long int training_batch = (long long int)batch % num_batches_training;
            long long int first_row_in_batch_training = batch_size_training * (long long int)training_batch;

            
            long long int GA_batch = (long long int)batch % num_batches_GU;
            long long int first_row_in_batch_GU = (batch_size_GU * (long long int)GA_batch) ;


            if(Debug){
                LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");
                LOG("batch_size_training : "<<batch_size_training);
                LOG("( next first_row_in_batch_training : "<<first_row_in_batch_training + batch_size_training<<" )");
                LOG("first_row_in_batch_GU : "<<first_row_in_batch_GU<<  " ( / "<<ratings_rows_GU<<" )");
                LOG("batch_size_GU : "<<batch_size_GU);
                LOG("( next first_row_in_batch_GU : "<<first_row_in_batch_GU + batch_size_GU<<" )");
            };
            if(first_row_in_batch_training + batch_size_training > ratings_rows_training) {
                LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");
                LOG("batch_size_training : "<<batch_size_training);
                LOG("( next first_row_in_batch_training : "<<first_row_in_batch_training + batch_size_training<<" )");
                LOG("SKIPPING train iteration "<<it<<", batch "<<batch);
                break;
            }
            if(first_row_in_batch_GU + batch_size_GU > ratings_rows_GU) {
                LOG("first_row_in_batch_GU : "<<first_row_in_batch_GU<<  " ( / "<<ratings_rows_GU<<" )");
                LOG("batch_size_GU : "<<batch_size_GU);
                LOG("( next first_row_in_batch_GU : "<<first_row_in_batch_GU + batch_size_GU<<" )");
                LOG("SKIPPING train iteration "<<it<<", batch "<<batch);
                break;
            };

            float* full_ratingsMtx_dev_GU_current_batch = NULL;
            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_GU, full_ratingsMtx_host_GU + ratings_cols * first_row_in_batch_GU, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyHostToDevice));
                full_ratingsMtx_dev_GU_current_batch = full_ratingsMtx_dev_GU;
            }else{
                full_ratingsMtx_dev_GU_current_batch = full_ratingsMtx_dev_GU + ratings_cols * first_row_in_batch_GU;
            }

            if(row_major_ordering){
                //remember that ratings_GU is stored in row major ordering
                if (batch_size_GU != ratings_rows_GU || !swapped){
                    LOG("swap matrix indexing from row major to column major");
                    swap_ordering<float>(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, row_major_ordering);
                    swapped = true;
                }
            }
            if(Debug && 0){
                save_device_mtx_to_file(full_ratingsMtx_dev_GU_current_batch, batch_size_GU, ratings_cols, "full_ratingsMtx_dev_GU_current_batch", false);

            }
            if(batch > 0 || it > 0){
                float R_GU_abs_exp = gpu_expected_abs_value<float>(batch_size_GU * ratings_cols, full_ratingsMtx_dev_GU_current_batch);
                float R_GU_abs_max = gpu_abs_max<float>           (batch_size_GU * ratings_cols, full_ratingsMtx_dev_GU_current_batch); 
                LOG("full_ratingsMtx_dev_GU_current_batch_abs_max = "<<R_GU_abs_max) ;
                LOG("full_ratingsMtx_dev_GU_current_batch_abs_exp = "<<R_GU_abs_exp) ;
                ABORT_IF_EQ(R_GU_abs_max, R_GU_abs_exp, "R_GU is constant");
                ABORT_IF_LESS((float)10.0 * abs_max_training, std::abs(R_GU_abs_max), "unstable growth");
                ABORT_IF_LESS( std::abs(R_GU_abs_max), abs_max_training / (float)10.0 , "unstable shrinking");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }






            //============================================================================================
            // Find U_GU, V such that U_GU * V^T ~ R_GU 
            //============================================================================================  
            
            gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                        batch_size_GU, ratings_cols, 
                                        &num_latent_factors, percent,
                                        full_ratingsMtx_dev_GU_current_batch, U_GU, V, 1);

            /*
                At this point U_GU is batch_size_GU by batch_size_GU in memory stored in column major
                ordering and V is ratings_cols by batch_size_GU stored in column major ordering

                There is no extra work to compress U_GU into batch_size_GU by num_latent_factors, or
                to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
                columns of each matrix
            */
            LOG("num_latent_factors = "<< num_latent_factors);
            if(compress){
                //ABORT_IF_NEQ(0, 1, "Not Yet Supported");
            }

            //============================================================================================
            // Compute  R_training * V = U_training
            // Compute  Error = R_training -  U_training * V^T  <-- sparse
            //============================================================================================ 
            //if(Debug) LOG("iteration "<<it<<" made it to check point");

            // checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training * std::max(min_, batch_size_training) * sizeof(float)));
            // checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training * ratings_cols                        * sizeof(float)));
            // checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_, batch_size_testing)  * sizeof(float)));
            // checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * sizeof(float)));

            int* csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_dev_training +  first_row_in_batch_training;

            long long int nnz_training = gpu_get_num_entries_in_rows(0, batch_size_training - 1, csr_format_ratingsMtx_userID_dev_training_batch);
            ABORT_IF_LESS(nnz_training, 1, "nnz < 1");
            total_training_nnz += nnz_training;
            long long int first_coo_ind_training = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_training_batch);

            float* coo_training_errors;
            //float* testing_entries;
            checkCudaErrors(cudaMalloc((void**)&coo_training_errors, nnz_training * sizeof(float)));
            //checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_training * sizeof(float)));

            if(Debug ){
                LOG("first_coo_ind in this training batch : "<<first_coo_ind_training<< " ( / "<<num_entries_training<<" )");
                LOG("nnz in this training batch : "<<nnz_training);
                LOG("( next first_coo_ind in training batch : "<<first_coo_ind_training + nnz_training <<" )");
                // save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev_training_batch, batch_size_training + 1, "csr_format_ratingsMtx_userID_dev_training_batch");
                // LOG("first entry of csr_format_ratingsMtx_userID_dev_training_batch: ");
                // print_gpu_array_entries<int>(csr_format_ratingsMtx_userID_dev_training_batch, 1);
                // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training + (int)first_coo_ind_training, nnz_training, "coo_format_ratingsMtx_rating_dev_training_batch");
                // save_device_array_to_file<int>(coo_format_ratingsMtx_itemID_dev_training + (int)first_coo_ind_training, nnz_training, "coo_format_ratingsMtx_itemID_dev_training");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }



            // gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
            //                    batch_size_training, batch_size_GU, num_latent_factors, ratings_cols,
            //                    nnz_training, first_coo_ind_training, compress, 
            //                    testing_entries, coo_training_errors, testing_fraction,
            //                    coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, 
            //                    csr_format_ratingsMtx_userID_dev_training_batch,         // <-- already has shifted to correct start
            //                    coo_format_ratingsMtx_itemID_dev_training + first_coo_ind_training,
            //                    V, U_training, R_training, "training", (float)0.1, (float)0.01);

            gpu_R_error_training<float>(dn_handle, sp_handle, sp_descr,
                                       batch_size_training, batch_size_GU, num_latent_factors, ratings_cols,
                                       nnz_training, first_coo_ind_training, compress, coo_training_errors,
                                       coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, 
                                       csr_format_ratingsMtx_userID_dev_training_batch,         // <-- already has shifted to correct start
                                       coo_format_ratingsMtx_itemID_dev_training + first_coo_ind_training,
                                       V, U_training, R_training, (float)0.1, (float)0.01);


            training_error_temp += gpu_sum_of_squares<float>(nnz_training, coo_training_errors);
            if(num_batches_GU - 1 == GA_batch){
                if( print_training_error && num_batches_GU > 1){
                    //LOG("           ~Finished round "<<count_GU_rounds<<" of GA training~"<<std::endl); 
                    LOG("num_latent_factors = "<< num_latent_factors);
                    long long int nnz_ = (long long int)((float)total_training_nnz /* * testing_fraction*/);
                    LOG("TRAINING AVERAGE SQUARED ERROR : "<< training_error_temp / (float)(nnz_)); 

                    // float temp = gpu_sum_of_squares<float>(nnz_training, testing_entries);
                    // float temp = gpu_sum_of_squares_of_diff(dn_handle, nnz_training, 
                    //                                         coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, 
                    //                                         testing_entries);
                    
                    // LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));
                    // LOG("training error normalized: "<< training_error_temp / temp<<std::endl); 
                    // LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 

                } 
                count_GU_rounds += 1; 
                training_error_temp = 0; 
                total_training_nnz = (long long int)0;
                
            }

            if(Debug){

                //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, testing_entries, coo_training_errors, nnz_training, "ratings_testing_errors");
                
                float coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_training_errors);
                float coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_training_errors);
                LOG("coo_training_errors_abs_max = "<<coo_training_errors_abs_max) ;
                LOG("coo_training_errors_abs_max over range of ratings = "<<coo_training_errors_abs_max / range_training) ;
                LOG("coo_training_errors_abs_exp = "<<coo_training_errors_abs_exp) ;
                LOG("coo_training_errors_abs_exp over range of ratings = "<<coo_training_errors_abs_exp / range_training) ;

                // coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                // coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                // LOG("coo_training_abs_max = "<<coo_training_errors_abs_max) ;
                // LOG("coo_training_abs_exp = "<<coo_training_errors_abs_exp) ;

                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }

            //cudaFree(testing_entries);

            if( it % testing_rate == 0){
                long long int first_row_in_batch_testing  = (batch_size_testing * (long long int)batch) /* % ratings_rows_testing*/;
                if(first_row_in_batch_testing + batch_size_testing > ratings_rows_testing) {
                    //LOG("SKIPPING test iteration "<<it<<", batch "<<batch);
                    //break;
                }else{
                    LOG("      ~~~ TESTING ~~~ "); 
                    if(Debug){
                        
                        LOG("first_row_in_batch_testing : "<<first_row_in_batch_testing<< " ( / "<<ratings_rows_testing<<" )");
                        LOG("batch_size_testing : "<<batch_size_testing);
                        LOG("( next first_row_in_batch_testing : "<<first_row_in_batch_testing + batch_size_testing<<" )");
                    };
                    int* csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing +  first_row_in_batch_testing;
                    
                    long long int nnz_testing = gpu_get_num_entries_in_rows(0, batch_size_testing - 1, csr_format_ratingsMtx_userID_dev_testing_batch);
                    ABORT_IF_LESS(nnz_testing, 1, "nnz < 1");
                    total_testing_nnz += nnz_testing;
                    long long int first_coo_ind_testing = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_testing_batch);
                    
                    float* coo_testing_errors;
                    float* testing_entries;
                    checkCudaErrors(cudaMalloc((void**)&coo_testing_errors, nnz_testing * sizeof(float)));
                    checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_testing * sizeof(float)));

                    if(Debug){
                        
                        LOG("first_coo_ind in this TESTING batch : "<<first_coo_ind_testing<< " ( / "<<num_entries_testing<<" )");
                        LOG("nnz in this TESTING batch : "<<nnz_testing);
                        LOG("( nest first_coo_ind in TESTING batch : "<<first_coo_ind_testing+ nnz_testing<<" )");
                        // save_device_mtx_to_file<float>(R, batch_size_training, ratings_cols, "error", false);
                        // LOG("Press Enter to continue.") ;
                        // std::cin.ignore();
                    }
                    //============================================================================================
                    // Compute  R_testing * V = U_testing
                    // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
                    //============================================================================================ 
                    // gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
                    //                    batch_size_testing, batch_size_GU, num_latent_factors, ratings_cols,
                    //                    nnz_testing, first_coo_ind_testing, compress, 
                    //                    testing_entries, coo_testing_errors, testing_fraction,
                    //                    coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
                    //                    csr_format_ratingsMtx_userID_dev_testing_batch, 
                    //                    coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
                    //                    V, U_testing, R_testing, "testing", (float)0.1, (float)0.01);

                    float testing_error_on_training_entries_temp;
                    long long int total_iterations_temp = (long long int)0;
                    gpu_R_error_testing<float>(dn_handle, sp_handle, sp_descr,
                                       batch_size_testing, batch_size_GU, num_latent_factors, ratings_cols,
                                       nnz_testing, first_coo_ind_testing, compress_when_testing, 
                                       testing_entries, coo_testing_errors, testing_fraction,
                                       coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
                                       csr_format_ratingsMtx_userID_dev_testing_batch, 
                                       coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
                                       V, U_testing, R_testing, (float)0.1, (float)0.01,
                                       &testing_error_on_training_entries_temp, &testing_error_temp, 
                                       &total_iterations_temp);

                    //gpu_reverse_bools<float>(nnz_testing,  testing_entries);
                    //gpu_hadamard<float>(nnz_testing, testing_entries, coo_testing_errors );
                    //save_device_arrays_side_by_side_to_file<float>(coo_testing_errors, testing_entries, nnz_testing, "testing_entry_errors");

                    //testing_error_temp += gpu_sum_of_squares<float>(nnz_testing, coo_testing_errors);
                    
                    cudaFree(coo_testing_errors);
                    cudaFree(testing_entries);  

                    count_tests +=1;

                    if(num_batches_testing < num_batches_GU && num_batches_testing == count_tests){

                        LOG("~~Iteration "<<it<<" TESTING~~"); 

                        float nnz_ = (float)total_testing_nnz * testing_fraction;
                        LOG("num testing entries : "<<  nnz_ );
                        long long int num_batches_ = (long long int)((float)(batch_size_testing * num_batches_testing) * testing_fraction);
                        LOG("AVERAGE SQUARED TESTING ERROR : "<< testing_error_temp / nnz_ ); 

                    
                        testing_error[num_tests] = testing_error_temp / nnz_ ;
                        //testing_error[num_tests] = testing_error_temp / (float)(nnz_ * 2.0);
                        save_host_array_to_file<float>(testing_error, num_tests, "testing_error");
                        num_tests += 1;
                        total_testing_nnz = (long long int)0;
                    }
                    if((float)total_iterations_temp > (float)total_testing_iterations * (float)1.25){
                        LOG("total_iterations_temp > total_testing_iterations * 1.25 : "<<total_iterations_temp<<" > "<<total_testing_iterations<<" * "<<1.25<<" = "<< (float)total_testing_iterations * (float)1.25);
                        checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_GU_current_batch, old_R_GU,
                                                    batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));

                        training_rate = training_rate / (float)10.0;
                        LOG("REDUCING LEARNING RATE TO : "<<training_rate);
                        break;
                    }else{
                       total_testing_iterations = total_iterations_temp; 
                    }
                    LOG("      ~~~ DONE TESTING ~~~ "<<std::endl); 
                    
                }


            }//end is testing iter


            //============================================================================================
            // Update  V = V * (1 -alpha * lambda) + alpha * Error^T * U_training 
            // (Update  U = U * (1 -alpha * lambda) + alpha * Error * V ) <- random error?????
            //============================================================================================ 
            //if(Debug) LOG("iteration "<<it<<" made it to check point");
            /*
                m,n,k
                This function performs one of the following matrix-matrix operations:

                C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

                A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

                op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                and

                op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
                array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.

                n is the number of columns of dense matrix op(B) and C.
            */

            float* delta_V;

            if(1){
                checkCudaErrors(cudaMalloc((void**)&delta_V, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors) * sizeof(float)));
                checkCudaErrors(cudaMemcpy(delta_V, V, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors) * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            float alpha =  training_rate;
            float beta = (float)1.0 - alpha * regularization_constant;
            gpu_spXdense_MMM<float>(sp_handle, true, false, batch_size_training, 
                                    (compress == false) ? batch_size_GU : num_latent_factors, 
                                    ratings_cols, nnz_training, first_coo_ind_training, &alpha, sp_descr, 
                                    coo_training_errors, 
                                    csr_format_ratingsMtx_userID_dev_training_batch, 
                                    coo_format_ratingsMtx_itemID_dev_training + first_coo_ind_training,
                                    U_training, batch_size_training, &beta, V, ratings_cols, false);
            if(normalize_V_rows){
                gpu_normalize_mtx_rows_or_cols(ratings_cols, (compress == false) ? batch_size_GU : num_latent_factors,  
                                      false, V, true);
                gpu_normalize_mtx_rows_or_cols(ratings_cols, (compress == false) ? batch_size_GU : num_latent_factors,  
                                      false, V, false);
            }

            /*
                N is number of rows of matrix op(B) and C.
                M number of columns of matrix op(A) and C.
                K is number of columns of op(B) and rows of op(A).
                
                op(B) is N by K
                op(A) is K by M
                C is N by M

                performs C=alpha op ( B ) op ( A ) + beta C
            */


            /*
            gpu_noisey_gemm<float>(dn_handle, false, false, 
                            batch_size_GU, (compress == false) ? batch_size_GU : num_latent_factors, 
                            ratings_cols,
                            beta, //(float)1.0
                            V, training_rate, beta,
                            U_GU);
            */
            if(regularize_U){
                if(compress){
                    gpu_scale<float>(dn_handle, batch_size_GU * num_latent_factors, beta, U_GU);
                }else{
                    gpu_scale<float>(dn_handle, batch_size_GU * batch_size_GU, beta, U_GU);
                }                
            }

            

            if(1){
                float* copy;
                checkCudaErrors(cudaMalloc((void**)&copy, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors) * sizeof(float)));
                checkCudaErrors(cudaMemcpy(copy, delta_V, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors) * sizeof(float), cudaMemcpyDeviceToDevice));
                gpu_axpby<float>(dn_handle, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors), 
                                 (float)(-1.0), V,
                                 (float)(1.0), delta_V);
                float delta_abs_exp = gpu_expected_abs_value<float>(ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors), delta_V);
                float delta_abs_max = gpu_abs_max<float>(ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors), delta_V); 
                LOG("delta V maximum absolute value = "<<delta_abs_max) ;
                LOG("delta V expected absolute value = "<<delta_abs_exp) ;
                ABORT_IF_EQ(delta_abs_max, delta_abs_exp, "delta V is constant");
                // save_device_arrays_side_by_side_to_file(copy, V, delta_V, ratings_cols * ((compress == false) ? batch_size_GU : num_latent_factors), "old_new_delta_V");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
                cudaFree(delta_V);
                cudaFree(copy);
            }

            //============================================================================================
            // Update  R_GU = U_GU * V^T
            //============================================================================================ 
            //if(Debug) LOG("iteration "<<it<<" made it to check point");

            checkCudaErrors(cudaMemcpy(old_R_GU, full_ratingsMtx_dev_GU_current_batch, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));

            float* delta_R_GU;

            if(1){
                checkCudaErrors(cudaMalloc((void**)&delta_R_GU, batch_size_GU * ratings_cols * sizeof(float)));
                checkCudaErrors(cudaMemcpy(delta_R_GU, full_ratingsMtx_dev_GU_current_batch, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            gpu_gemm<float>(dn_handle, true, false, 
                            ratings_cols, batch_size_GU, 
                            (compress == false) ? batch_size_GU : num_latent_factors,
                            (regularize_R == true) ? training_rate : (float)1.0,
                            V, U_GU, 
                            (regularize_R == true) ? beta : (float)0.0,
                            full_ratingsMtx_dev_GU_current_batch);

            if (regularize_R_distribution == true ){
                center_rows(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, 
                    val_when_var_is_zero, user_means_GU,  user_var_GU);
            }

            if(1){
                save_device_mtx_to_file<float>(V, ratings_cols, num_latent_factors, "V_compressed");
                save_device_mtx_to_file<float>(U_GU, batch_size_GU, num_latent_factors, "U_GU_compressed");
                float* copy;
                checkCudaErrors(cudaMalloc((void**)&copy, batch_size_GU * ratings_cols * sizeof(float)));
                checkCudaErrors(cudaMemcpy(copy, delta_R_GU, batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
                gpu_axpby<float>(dn_handle, batch_size_GU * ratings_cols, 
                                 (float)(-1.0), full_ratingsMtx_dev_GU_current_batch,
                                 (float)(1.0), delta_R_GU);
                float delta_abs_exp = gpu_expected_abs_value<float>(batch_size_GU * ratings_cols, delta_R_GU);
                float delta_abs_max = gpu_abs_max<float>(batch_size_GU * ratings_cols, delta_R_GU); 
                LOG("delta R_GU maximum absolute value = "<<delta_abs_max) ;
                LOG("delta R_GU expected absolute value = "<<delta_abs_exp) ;
                // save_device_arrays_side_by_side_to_file(copy, full_ratingsMtx_dev_GU_current_batch, delta_R_GU, batch_size_GU * ratings_cols, "old_new_delta");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
                cudaFree(delta_R_GU);
                cudaFree(copy);
            }
            if(0){
                if(Conserve_GPU_Mem){
                    save_host_mtx_to_file<float>(full_ratingsMtx_host_GU, ratings_rows_GU, ratings_cols, "full_ratingsMtx_GU");
                }else{
                    save_device_mtx_to_file<float>(full_ratingsMtx_dev_GU, ratings_rows_GU, ratings_cols, "full_ratingsMtx_GU", false);
                }
            }
            
            

            if(row_major_ordering && batch_size_GU != ratings_rows_GU){
                //remember that ratings_GU is stored in row major ordering
                LOG("swap matrix indexing from column major to row major");
                swap_ordering<float>(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, !row_major_ordering);
            }
            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_GU + ratings_cols * first_row_in_batch_GU, full_ratingsMtx_dev_GU_current_batch, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToHost));
            };


            cudaFree(coo_training_errors);


            if(batch_size_GU == ratings_rows_GU){
                LOG("shuffle GU matrix rows");
                if(Conserve_GPU_Mem){
                    cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols, 
                                                 false, full_ratingsMtx_host_GU, 1);
                }else{
                    //shuffle GU rows
                    gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols,  
                                                false, full_ratingsMtx_dev_GU, 1);


                    //shuffle training rows??
                }
            }

            float* errors;
            int* selection;
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
            if(!Conserve_GPU_Mem){
                sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
                     ratings_rows_training, num_entries_training, 
                     csr_format_ratingsMtx_userID_dev_training, 
                     coo_format_ratingsMtx_itemID_dev_training,
                     coo_format_ratingsMtx_rating_dev_training, 
                     selection, errors);
                float k_means_er = gpu_sum(ratings_rows_training,  errors);
                LOG("err norm when clustering over training norm: "<<std::sqrt(k_means_er) / std::sqrt(gpu_sum_of_squares(num_entries_training,  coo_format_ratingsMtx_rating_dev_training)));
                LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                save_device_array_to_file<float>(errors, ratings_rows_training, "k_means_errors");
                save_device_array_to_file<int>(selection, ratings_rows_training, "k_means_selection");
            }

            cudaFree(errors);
            cudaFree(selection);

        }//end for loop on batches


        if(batch_size_GU != ratings_rows_GU){
            LOG("shuffle GU matrix rows");
            if(Conserve_GPU_Mem){
                cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols, 
                                             row_major_ordering, full_ratingsMtx_host_GU, 1);
            }else{
                //shuffle GU rows
                gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols,  
                                            row_major_ordering, full_ratingsMtx_dev_GU, 1);


                //shuffle training rows??
            }
        }
        if(it % (num_iterations / 4) == 0){
           training_rate =  training_rate / (float)10.0;
        }
        gettimeofday(&training_end, NULL);
        training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
        LOG("average training iteration time : "<<readable_time(training_time / (double)(it + 1)));
    }//end for loop on iterations
    //save_host_array_to_file<float>(testing_error, (num_iterations / testing_rate), "testing_error");

    float* errors;
    int* selection;
    checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
    if(!Conserve_GPU_Mem){
        sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
             ratings_rows_training, num_entries_training, 
             csr_format_ratingsMtx_userID_dev_training, 
             coo_format_ratingsMtx_itemID_dev_training,
             coo_format_ratingsMtx_rating_dev_training, 
             selection, errors);
        float k_means_er = gpu_sum(ratings_rows_training,  errors);
        LOG("err norm when clustering : "<<std::sqrt(k_means_er));
        LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
        save_device_array_to_file<float>(errors, ratings_rows_training, "errors");
        save_device_array_to_file<int>(selection, ratings_rows_training, "selection");
    }

    cudaFree(errors);
    cudaFree(selection);

    cudaDeviceSynchronize();
    gettimeofday(&training_end, NULL);
    training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
    LOG("training_time : "<<readable_time(training_time));
    //============================================================================================
    // Destroy
    //============================================================================================
    LOG("Cleaning Up...");
    //free(user_means_training_host);
    free(user_means_testing_host);
    free(user_var_testing_host);
    free(testing_error);
    if (full_ratingsMtx_host_GU  ) { free(full_ratingsMtx_host_GU); }


    cudaFree(U_GU);
    cudaFree(U_training);
    cudaFree(U_testing);
    cudaFree(old_R_GU);
    cudaFree(V);
    cudaFree(R_training);
    cudaFree(R_testing);
    update_Mem((batch_size_GU * std::min(batch_size_GU, ratings_cols) +batch_size_training * batch_size_training + ratings_cols * std::min(batch_size_GU, ratings_cols))* static_cast<long long int>(sizeof(float))* (-1));
    

    cudaFree(user_means_GU);                             update_Mem(ratings_rows_GU * sizeof(float) * (-1));
    cudaFree(user_var_GU);                               update_Mem(ratings_rows_GU * sizeof(float) * (-1));


    cudaFree(csr_format_ratingsMtx_userID_dev_training);
    cudaFree(coo_format_ratingsMtx_itemID_dev_training);
    cudaFree(coo_format_ratingsMtx_rating_dev_training);
    
    cudaFree(full_ratingsMtx_dev_GU);
    
    cudaFree(csr_format_ratingsMtx_userID_dev_testing);
    cudaFree(coo_format_ratingsMtx_itemID_dev_testing);
    cudaFree(coo_format_ratingsMtx_rating_dev_testing);
    

    update_Mem((ratings_rows_GU * ratings_cols + /*ratings_rows_training * ratings_cols +*/ num_entries_testing + num_entries_training)* sizeof(float));
    update_Mem(( (ratings_rows_testing + 1)  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(float))
               + (ratings_rows_training + 1) * static_cast<long long int>(sizeof(int)) + num_entries_training * static_cast<long long int>(sizeof(int)) + num_entries_training * static_cast<long long int>(sizeof(float))
               + (ratings_rows_GU + 1)       * static_cast<long long int>(sizeof(int)) + num_entries_GU       * static_cast<long long int>(sizeof(int)) + num_entries_GU       * static_cast<long long int>(sizeof(float))) * (-1) );
    

    if(Content_Based){
        cudaFree(csr_format_keyWordMtx_itemID_dev);          update_Mem((ratings_cols + 1)      * sizeof(int) * (-1));
        cudaFree(coo_format_keyWordMtx_keyWord_dev);          update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
    }

    cublasDestroy          (dn_handle);
    cusolverDnDestroy      (dn_solver_handle);

    cusparseDestroy        (sp_handle);
    cusparseDestroyMatDescr(sp_descr);
    cusolverSpDestroy      (sp_solver_handle);

    cudaDeviceSynchronize();
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("program_time : "<<readable_time(program_time));

    //if(Debug && memLeft!=devMem)LOG("WARNING POSSIBLE DEVICE MEMORY LEAK");
         
}
