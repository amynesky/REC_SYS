/**
    Author: Amy Nesky
**/
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

bool Content_Based = 0;         // This means that we use extra knowledge about the user preferences or about the item relationships
bool random_initialization = 1; // This means that we initialize the full GU rating mtx randomly
bool Conserve_GPU_Mem = 1;      // This means that the full GU rating mtx is stored on the host 

#define update_Mem(new_mem) \
    allocatedMem += static_cast<long long int>(new_mem); \
    memLeft = static_cast<long long int>(devMem) - allocatedMem;
    // if(Debug) LOG(allocatedMem<<" allocated bytes on the device");\
    // if(Debug) LOG(memLeft<<" available bytes left on the device");
    // ABORT_IF_LESS(memLeft, 0, "Out of Memory"); 

    //
    //ABORT_IF_LESS(allocatedMem, (long long int)((double)devMem * (double)0.75), "Out of Memory"); \
    // if(Debug) LOG((int)devMem <<" total bytes on the device");\
    // if(Debug) LOG(new_mem<<" change in memory on the device");\






// #define allocate_V() \
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
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

    LOG("random_initialization : "<< random_initialization);// This means that we initialize the full GU rating mtx randomly
    LOG("Conserve_GPU_Mem : "<< Conserve_GPU_Mem);          // This means that the full GU rating mtx is stored on the host
    LOG("Content_Based : "<< Content_Based<<std::endl);     // This means that we use extra knowledge about the user preferences or about the item relationships



    CSVReader csv_Ratings(csv_Ratings_Path);
    

    const long long int num_entries = temp_num_entries;
    //const int num_entries = 10000; //for debuging code
    
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
        LOG("Content_Based --> This means that we use extra knowledge about the user preferences or about the item relationships") ;
        
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
    if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
        checkCudaErrors(cudaMalloc((void**)&csr_format_keyWordMtx_itemID_dev, (ratings_cols + 1) * sizeof(int)));
        update_Mem( (ratings_cols + 1) * sizeof(int) );

        cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_keyWordMtx_itemID_dev, num_entries_keyWord_mtx, 
                                           ratings_cols, csr_format_keyWordMtx_itemID_dev, CUSPARSE_INDEX_BASE_ZERO); 
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
            fprintf(stdout, "Conversion from COO to CSR format failed\n");
            return 1; 
        } 
        checkCudaErrors(cudaFree(coo_format_keyWordMtx_itemID_dev));           update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
    }














    //============================================================================================
    // split the data into testing data and training data
    //============================================================================================

    










    const float probability_GU       = (float)20.0/(float)100.0;
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


    if(random_initialization){
        ratings_rows_GU_temp  /= (long long int)1000;
        ratings_rows_GU_temp  *= (long long int)1000; // now ratings_rows_GU_temp is divisible by 100
    }else{
        ratings_rows_GU_temp = group_sizes[2];
    }
    const long long int num_entries_GU        = ratings_rows_GU_temp;


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
    
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_userID_dev));
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


    
    checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev));            update_Mem((ratings_rows + 1) * sizeof(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev));            update_Mem(num_entries * sizeof(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev));            update_Mem(num_entries * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_by_group));   update_Mem(num_groups * sizeof(int*) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_by_group));   update_Mem(num_groups * sizeof(int*) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_by_group));   update_Mem(num_groups * sizeof(float*) * (-1));
    checkCudaErrors(cudaFree(ratings_rows_by_group));                       update_Mem(num_groups * sizeof(int) * (-1));
    checkCudaErrors(cudaFree(probability_of_groups_dev));                   update_Mem(num_groups * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(group_indicies));                              update_Mem(num_groups * sizeof(int)* (-1));
    
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


    collect_user_means(user_means_training, user_var_training,  (long long int)ratings_rows_training,
                       csr_format_ratingsMtx_userID_dev_training,
                       coo_format_ratingsMtx_rating_dev_training,
                       user_means_testing, user_var_testing,    (long long int)ratings_rows_testing,
                       csr_format_ratingsMtx_userID_dev_testing,
                       coo_format_ratingsMtx_rating_dev_testing,
                       user_means_GU, user_var_GU,              random_initialization? (long long int)0 : (long long int)ratings_rows_GU,
                       csr_format_ratingsMtx_userID_dev_GU,
                       coo_format_ratingsMtx_rating_dev_GU);

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
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_GU));             update_Mem((ratings_rows_GU + 1) * sizeof(int) * (-1));
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
    
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_testing));        update_Mem(num_entries_GU        * sizeof(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_training));       update_Mem(num_entries_GU        * sizeof(float) * (-1));

    
    
    float range_training = gpu_range<float>         (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float min_training   = gpu_min<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float max_training   = gpu_max<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float abs_max_training = std::max(max_training, std::abs(min_training));
    const float expt_training  = gpu_expected_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float expt_abs_training  = gpu_expected_abs_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);

    float range_testing  = gpu_range<float>(num_entries_testing,   coo_format_ratingsMtx_rating_dev_testing);
    const float min_testing   = gpu_min<float>           (num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float max_testing   = gpu_max<float>           (num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float abs_max_testing = std::max(std::abs(max_testing), std::abs(min_testing));
    const float expt_testing  = gpu_expected_value<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float expt_abs_testing  = gpu_expected_abs_value<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);

    range_training = (float)2.0 * expt_abs_training;

    //LOG(std::endl<<"range_GU = "         <<range_GU) ;
    //LOG("range_testing = "    <<range_testing) ;
    //LOG("range_training = "   <<range_training) ;
    LOG("max_training = "     <<max_training) ;
    LOG("min_training = "     <<min_training) ;
    LOG("abs_max_training = " <<abs_max_training) ;
    LOG("expt_training = "    <<expt_training) ;
    LOG("expt_abs_training = "<<expt_abs_training) ;

    LOG("max_testing = "     <<max_testing) ;
    LOG("min_testing = "     <<min_testing) ;
    LOG("abs_max_testing = " <<abs_max_testing) ;
    LOG("expt_testing = "    <<expt_testing) ;
    LOG("expt_abs_testing = "<<expt_abs_testing) ;

   
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
    LOG("GU matrix row_major_ordering = " <<row_major_ordering);

    float * full_ratingsMtx_dev_GU = NULL;
    float * full_ratingsMtx_host_GU = NULL;

    const long long int GU_mtx_size = (long long int)ratings_rows_GU * (long long int)ratings_cols;
    const long long int GU_mtx_size_bytes = (long long int)ratings_rows_GU * (long long int)ratings_cols * (long long int)sizeof(float);
    LOG(std::endl<<"Will need "<<GU_mtx_size<< " floats for the GU mtx.") ;
    LOG("Will need "<<GU_mtx_size_bytes<< " bytes for the GU mtx.") ;
    if(allocatedMem + GU_mtx_size_bytes > (long long int)((double)devMem * (double)0.75)){
        LOG("Conserving Memory Now --> This means that the full GU rating mtx is stored on the host ");
        Conserve_GPU_Mem = 1;
    }
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

    
    int*   csr_format_ratingsMtx_userID_host_GU = NULL;
    int*   coo_format_ratingsMtx_itemID_host_GU = NULL;
    float* coo_format_ratingsMtx_rating_host_GU = NULL;
    
    if(Conserve_GPU_Mem){
        // This means that the full GU rating mtx is stored on the host
        
        full_ratingsMtx_host_GU = (float *)malloc(GU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_GU);
        
        //cpu_set_all<float>(full_ratingsMtx_host_GU, GU_mtx_size, (float)0.0);
        //host_rng_uniform(ratings_rows_GU * ratings_cols, min_training, max_training, full_ratingsMtx_host_GU);
        host_rng_uniform(ratings_rows_GU * ratings_cols, (float)((-1.0)* std::sqrt(3.0)), (float)(std::sqrt(3.0)), full_ratingsMtx_host_GU);

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

            if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
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

            checkCudaErrors(cudaDeviceSynchronize());
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
        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_GU));         update_Mem((ratings_rows_GU + 1) * sizeof(int)   * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_GU));         update_Mem(num_entries_GU        * sizeof(int)   * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_GU));         update_Mem(num_entries_GU        * sizeof(float) * (-1));
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
    
    checkCudaErrors(cudaFree(user_means_testing));                        update_Mem(ratings_rows_testing * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(user_var_testing));                          update_Mem(ratings_rows_testing * sizeof(float) * (-1));

    //the stored means are useless now
    checkCudaErrors(cudaFree(user_means_GU));                             update_Mem(ratings_rows_GU * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(user_var_GU));                               update_Mem(ratings_rows_GU * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(user_means_training));                       update_Mem(ratings_rows_training * sizeof(float) * (-1));
    checkCudaErrors(cudaFree(user_var_training));                         update_Mem(ratings_rows_training * sizeof(float) * (-1));


    if (user_means_testing_host) free(user_means_testing_host);
    if (user_var_testing_host) free(user_var_testing_host);

    //============================================================================================
    // We want to find orthogonal matrices U, V such that R ~ U*V^T
    // 
    // R is batch_size_GU by ratings_cols
    // U is batch_size_GU by num_latent_factors
    // V is ratings_cols by num_latent_factors
    //============================================================================================
    LOG(std::endl);

    bool        print_training_error    = true;


    float meta_training_rate;
    float micro_training_rate;      
    float regularization_constant;         

    const float testing_fraction            = 0.2; //percent of known entries used for testing
    bool        compress_when_training      = false;
    bool        compress_when_testing       = false;
    bool        update_U                    = true;
    bool        regularize_R                = true;
    bool        regularize_R_distribution   = false;
    bool        normalize_V_rows            = false;
    bool        SV_with_U                   = false;

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            //Dataset_Name = "MovieLens 20 million";
            meta_training_rate          = (float)0.01; 
            micro_training_rate         = (float)0.0001;      //use for movielens
            regularization_constant     = (float)0.01;         //use for movielens
            update_U                    = true;
            regularize_R                = true;
            regularize_R_distribution   = false;
            normalize_V_rows            = false;
            compress_when_testing       = true;
            break;
        }case 2:{ // code to be executed if n = 2;
            //Dataset_Name = "Rent The Runaway";
            meta_training_rate           = (float)0.01;      //use for rent the runway
            regularization_constant = (float)1.0;         //use for rent the runway
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }
    // if(Conserve_GPU_Mem){
    //     compress_when_training = true;
    // }
    if(compress_when_training){
        compress_when_testing = true;
    }

    const int num_iterations = 10000;  // number of training iterations
    const int num_batches    = 100;     // number of training batches per iteration (batches index into training data)
    const int num_blocks     = 27 * 2;      // number of blocks of generic users (a single block of generic users is updated in a batch)
    const int testing_rate   = 1;      // 

    LOG("meta_training_rate : "        <<meta_training_rate);
    LOG("micro_training_rate : "       <<micro_training_rate);
    LOG("regularization_constant : "   <<regularization_constant);
    LOG("testing_fraction : "          <<testing_fraction);
    LOG("update_U : "                  <<update_U);
    LOG("regularize_R : "              <<regularize_R);
    LOG("regularize_R_distribution : " <<regularize_R_distribution);
    LOG("compress_when_training : "    <<compress_when_training);
    LOG("compress_when_testing : "     <<compress_when_testing);
    LOG("num_iterations : "            <<num_iterations);
    LOG("num_batches : "               <<num_batches);
    LOG("num_blocks: "                 <<num_blocks);
    LOG("testing_rate : "              <<testing_rate);
    LOG("SV_with_U : "                 <<SV_with_U);

    float * testing_error = NULL;
    testing_error = (float *)malloc((num_iterations / testing_rate) * sizeof(float)); 
    checkErrors(testing_error);
    cpu_set_all<float>(testing_error, (num_iterations / testing_rate), (float)0.0);

    const long long int batch_size_training = std::max((long long int)1, ratings_rows_training / (num_batches));
    const long long int batch_size_GU       = std::max((long long int)1, ratings_rows_GU / (num_blocks));
    const long long int batch_size_testing  = std::min((long long int)200, std::min(ratings_rows_testing, batch_size_GU));
    LOG(std::endl);
    LOG("batch_size_testing : " <<batch_size_testing);
    LOG("batch_size_training : "<<batch_size_training);
    LOG("batch_size_GU : "      <<batch_size_GU);
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

    ABORT_IF_LE(batch_size_training, (long long int)1, "batch size training is too small");
    ABORT_IF_LE(batch_size_testing, (long long int)1, "batch size testing is too small");
    ABORT_IF_LE(batch_size_GU, (long long int)1, "batch size GU is too small");
    ABORT_IF_LE((long long int)3000, batch_size_GU, "batch size GU is too large");
    ABORT_IF_LE((long long int)3000, batch_size_training, "batch size training is too large");
    ABORT_IF_LE((long long int)3000, batch_size_testing, "batch size testing is too large");

    ABORT_IF_NEQ(ratings_rows_GU % batch_size_GU, (long long int)0, "ratings_rows_GU % batch_size_GU != 0"<<std::endl);

    long long int num_batches_training = (long long int)(std::ceil((float)ratings_rows_training  / (float)batch_size_training)); // = num_batches
    //long long int num_batches_GU       = std::ceil(ratings_rows_GU       / batch_size_GU);
    long long int num_batches_testing  = (long long int)(std::ceil((float)ratings_rows_testing  / (float)batch_size_testing));

    LOG("ratings_rows_GU : " <<ratings_rows_GU);
    LOG("num_entries_GU : "<<num_entries_GU);
    LOG("batch_size_GU : " <<batch_size_GU);
    LOG("num_blocks : " <<num_blocks<<std::endl);

    LOG("ratings_rows_training : " <<ratings_rows_training);
    LOG("num_entries_training : "<<num_entries_training);
    LOG("batch_size_training : " <<batch_size_training);
    LOG("num_batches_training : " <<num_batches_training<<std::endl);

    LOG("ratings_rows_testing : " <<ratings_rows_testing);
    LOG("num_entries_testing : "<<num_entries_testing);
    LOG("batch_size_testing : " <<batch_size_testing);
    LOG("num_batches_testing : " <<num_batches_testing<<std::endl);


    LOG("ratings_cols : " <<ratings_cols<<std::endl);
    
    const float percent              = (float)0.999;  // How much of the full SV spectrum do you want to keep as latent factors?
    long long int num_latent_factors = (long long int)((float)ratings_rows_GU * percent);
    long long int max_num_latent_factors = (long long int)12000;

    /*
        There are a couple different factorization possibilities:
            - You could factor the whole R_GU (When R_GU is large this is done by factoring blocks and agregating the factored results)
            - You could factor only a horizontal strip of R_GU containing a subset of the GU users 
    */

    long long int min_dim_block = std::min(batch_size_GU, ratings_cols); // minimum dimension when factoring only a block of R_GU
    long long int min_dim_ = std::min(ratings_rows_GU, ratings_cols);    // minimum dimension when factoring the entire R_GU mtx


    float * old_R_GU;         // R_GU is ratings_rows_GU * ratings_cols
    float * U_GU_host;        // U_GU is ratings_rows_GU * ratings_rows_GU
    float * U_GU_dev;         // U_GU is ratings_rows_GU * ratings_rows_GU
    float * V_host;           // V_GU is ratings_cols * ratings_cols
    float * V_dev;            // V_GU is ratings_cols * ratings_cols


    bool temp = Conserve_GPU_Mem;
    const long long int Training_bytes = (ratings_rows_GU     * min_dim_ +
                                          batch_size_training * std::max(min_dim_block, batch_size_training) + 
                                          batch_size_testing  * std::max(min_dim_, batch_size_testing) + 
                                          ratings_cols        * min_dim_ +
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
        checkCudaErrors(cudaFree(full_ratingsMtx_dev_GU));
        update_Mem((float)(-1.0) * GU_mtx_size_bytes );
    };

    int *   csr_format_ratingsMtx_userID_host_testing = NULL;
    int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    float * coo_format_ratingsMtx_rating_host_testing = NULL;
    int *   csr_format_ratingsMtx_userID_host_training = NULL;
    int *   coo_format_ratingsMtx_itemID_host_training = NULL;
    float * coo_format_ratingsMtx_rating_host_training = NULL;

    float* SV = NULL;

    if(Conserve_GPU_Mem){
        //============================================================================================
        // Conserve Memory
        //============================================================================================
        csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc((ratings_rows_testing + 1) *  sizeof(int)); 
        coo_format_ratingsMtx_itemID_host_testing  = (int *)  malloc(num_entries_testing  *  sizeof(int)); 
        coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  sizeof(float));
        csr_format_ratingsMtx_userID_host_training = (int *)  malloc((ratings_rows_training + 1) *  sizeof(int)); 
        coo_format_ratingsMtx_itemID_host_training = (int *)  malloc(num_entries_training  *  sizeof(int)); 
        coo_format_ratingsMtx_rating_host_training = (float *)malloc(num_entries_training  *  sizeof(float));

        old_R_GU  = (float *)malloc(ratings_rows_GU * ratings_cols  *  sizeof(float));

        checkErrors(csr_format_ratingsMtx_userID_host_testing);
        checkErrors(coo_format_ratingsMtx_itemID_host_testing);
        checkErrors(coo_format_ratingsMtx_rating_host_testing); 
        checkErrors(csr_format_ratingsMtx_userID_host_training);
        checkErrors(coo_format_ratingsMtx_itemID_host_training);
        checkErrors(coo_format_ratingsMtx_rating_host_training); 
        checkErrors(old_R_GU); 

        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) *  sizeof(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  sizeof(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_training,  csr_format_ratingsMtx_userID_dev_training,  (ratings_rows_training + 1) *  sizeof(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_training, coo_format_ratingsMtx_itemID_dev_training, num_entries_training  *  sizeof(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_training,  coo_format_ratingsMtx_rating_dev_training,  num_entries_training  *  sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing));  update_Mem((ratings_rows_testing + 1)*  sizeof(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing));  update_Mem(num_entries_testing  *  sizeof(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing));  update_Mem(num_entries_testing  *  sizeof(float) * (-1) );
        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training));  update_Mem((ratings_rows_training + 1)*  sizeof(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training));  update_Mem(num_entries_training  *  sizeof(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training));  update_Mem(num_entries_training  *  sizeof(float) * (-1) );
        
        if(Debug && 0){
            save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_host_testing_1");
            save_host_array_to_file<int>  (coo_format_ratingsMtx_itemID_host_testing,  num_entries_testing, "coo_format_ratingsMtx_itemID_host_testing");
            save_host_array_to_file<float>  (coo_format_ratingsMtx_rating_host_testing,  num_entries_testing, "coo_format_ratingsMtx_rating_host_testing");
        }

        U_GU_host = (float *)malloc(ratings_rows_GU * ratings_rows_GU * sizeof(float));
        V_host = (float *)malloc(ratings_cols * min_dim_ * sizeof(float));
        checkErrors(U_GU_host);
        checkErrors(V_host);

        
        num_latent_factors = std::min(ratings_rows_GU, max_num_latent_factors);
        if(Debug && 0) {
            checkCudaErrors(cudaDeviceSynchronize()); 
            LOG("num_latent_factors = "<< num_latent_factors);
            LOG("min_dim_ = "<< min_dim_);
        }

        checkCudaErrors(cudaMalloc((void**)&V_dev, ratings_cols * num_latent_factors * sizeof(float)));
        update_Mem(ratings_cols * num_latent_factors * sizeof(float) );
        if(!row_major_ordering){
            ABORT_IF_EQ(0,1,"try again with row_major_ordering = true.")
        }

        // checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_GU, batch_size_GU * ratings_cols * sizeof(float)));
        // update_Mem(batch_size_GU * ratings_cols* sizeof(float) );

        SV = (float *)malloc(ratings_rows_GU *  sizeof(float)); 
        checkErrors(SV);
    }else{
        if(num_blocks > 1){
            ABORT_IF_EQ(0, 1, "Does not train blocks when Conserve_GPU_Mem is false.")
        }
        checkCudaErrors(cudaMalloc((void**)&U_GU_dev,       ratings_rows_GU     * min_dim_                     * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&V_dev,          ratings_cols        * min_dim_                     * sizeof(float)));
        update_Mem((ratings_rows_GU * min_dim_ + ratings_cols * min_dim_) * sizeof(float) );
        checkCudaErrors(cudaMalloc((void**)&old_R_GU, ratings_rows_GU * ratings_cols * sizeof(float)));
        update_Mem(ratings_rows_GU * ratings_cols * sizeof(float));
    }



    // LOG(ratings_cols * ratings_cols * sizeof(float)) ;

    // checkCudaErrors(cudaDeviceSynchronize());
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();  












    
    long long int total_testing_iterations = (long long int) 10000; //deprecated
    


    int count_batches_so_far = 0;
    



    if(Debug) LOG(memLeft<<" available bytes left on the device");









    float epsilon = (float)0.001;

    const float min_training_rate = (float)0.000001;

    float min_error_so_far = (float)100000.0;





    //============================================================================================
    // Assume the rows of R_GU are the ratings_rows_GU-means for the training users - what is the error?
    //============================================================================================ 

    if(1){
        float k_means_er = (float)0.0;
        float* errors;
        int* selection;
        if(Conserve_GPU_Mem){
            errors  = (float *)malloc(ratings_rows_training * sizeof(float));
            selection  = (int *)malloc(ratings_rows_training * sizeof(int));
            checkErrors(errors);
            checkErrors(selection);
            cpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_host_GU, 
                                         ratings_rows_training, num_entries_training, 
                                         csr_format_ratingsMtx_userID_host_training, 
                                         coo_format_ratingsMtx_itemID_host_training,
                                         coo_format_ratingsMtx_rating_host_training, 
                                         selection, errors);
            //k_means_er = cpu_sum(ratings_rows_training,  errors);
            //LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("expected MSQER when clustering : "<<cpu_expected_value(ratings_rows_training, errors));
            save_host_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_host_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            free(errors);
            free(selection);
        }else{
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
            update_Mem(2 * ratings_rows_training * sizeof(int));
            gpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
                                     ratings_rows_training, num_entries_training, 
                                     csr_format_ratingsMtx_userID_dev_training, 
                                     coo_format_ratingsMtx_itemID_dev_training,
                                     coo_format_ratingsMtx_rating_dev_training, 
                                     selection, errors, row_major_ordering);
            k_means_er = gpu_sum(ratings_rows_training,  errors);
            LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
            save_device_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_device_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            checkCudaErrors(cudaFree(errors));
            checkCudaErrors(cudaFree(selection));
            update_Mem(2 * ratings_rows_training * sizeof(int) * (-1));
        }
    }




    //============================================================================================
    // Begin Training
    //============================================================================================  
    LOG(std::endl<<std::endl<<"                              Begin Training..."<<std::endl); 
    gettimeofday(&training_start, NULL);
    int num_tests = 0;
    
    std::string blank = "";
    bool not_done = true;
    int it = 0;
    while(not_done && it < num_iterations && meta_training_rate >= min_training_rate){
    //for(int it = 0; it < num_iterations; it ++){

        float testing_error_on_training_entries_temp = (float)0.0;
        float total_testing_iterations = (float)0.0;
        float k_means_er = (float)0.0;
        //============================================================================================
        // TESTING
        //============================================================================================ 
        bool do_ = true;
        if( it % testing_rate == 0){
            LOG(std::endl<<"      ~~~ TESTING ~~~ "); 
            if(Debug){
                LOG(memLeft<<" available bytes left on the device");
                LOG("batch_size_testing : "<<batch_size_testing);
            }
            
            float * SV_dev;
            checkCudaErrors(cudaMalloc((void**)&SV_dev, ratings_rows_GU * sizeof(float)));
            update_Mem(ratings_rows_GU * sizeof(float));
            if(Conserve_GPU_Mem){
                /*
                    cpu_orthogonal_decomp<float>(ratings_rows_GU, ratings_cols, row_major_ordering,
                                            &num_latent_factors, percent,
                                            full_ratingsMtx_host_GU, U_GU, V_host, SV_with_U, SV);
                */
                LOG("num_blocks = "<< num_blocks);
                LOG("batch_size_GU = "<< batch_size_GU);
                
                gpu_block_orthogonal_decomp_from_host<float>(dn_handle, dn_solver_handle,
                                                             ratings_rows_GU, ratings_cols,
                                                             &num_latent_factors, percent,
                                                             full_ratingsMtx_host_GU, U_GU_host, 
                                                             V_host, batch_size_GU, SV_with_U, SV);
                                                             
                                                             
                LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< ratings_rows_GU<< " )");

                num_latent_factors = std::min(num_latent_factors, std::min(min_dim_ , max_num_latent_factors));

                checkCudaErrors(cudaMemcpy(V_dev, V_host, ratings_cols * num_latent_factors, cudaMemcpyHostToDevice));
                //if(Debug) {checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");}
                if(row_major_ordering){
                    //gpu_swap_ordering<float>(ratings_cols, num_latent_factors, V_dev, true);
                }
                if(Debug) {
                    //save_host_array_to_file<float>(SV, ratings_rows_GU, "singular_values", strPreamble(blank));
                    //save_host_mtx_to_file<float>(U_GU_host, ratings_rows_GU, num_latent_factors, "U_GU_compressed");
                }
                checkCudaErrors(cudaMemcpy(SV_dev, SV, ratings_rows_GU, cudaMemcpyHostToDevice));
                /*
                    At this point U_GU is ratings_rows_GU by ratings_rows_GU in memory stored in row major
                    ordering and V is ratings_cols by ratings_rows_GU stored in column major ordering

                    There is no extra work to compress to compress V into ratings_cols by num_latent_factors, 
                    just take the first num_latent_factors columns of each matrix.  
                    The columns of U_GU are mixed in memory.
                */
            }else{
                if(row_major_ordering){
                    //remember that R_GU is stored in row major ordering
                    LOG("swap matrix indexing from row major to column major");
                    gpu_swap_ordering<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, row_major_ordering);
                    
                }
                if(Debug && 0){
                    save_device_mtx_to_file(full_ratingsMtx_dev_GU, ratings_rows_GU, ratings_cols, "full_ratingsMtx_dev_GU", false);

                }

                float R_GU_abs_exp = gpu_expected_abs_value<float>(ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU);
                float R_GU_abs_max = gpu_abs_max<float>           (ratings_rows_GU * ratings_cols, full_ratingsMtx_dev_GU); 
                LOG("full_ratingsMtx_dev_GU_current_batch_abs_max = "<<R_GU_abs_max) ;
                LOG("full_ratingsMtx_dev_GU_current_batch_abs_exp = "<<R_GU_abs_exp) ;
                ABORT_IF_EQ(R_GU_abs_max, R_GU_abs_exp, "R_GU is constant");
                ABORT_IF_LESS((float)10.0 * abs_max_training, std::abs(R_GU_abs_max), "unstable growth");
                ABORT_IF_LESS( std::abs(R_GU_abs_max), abs_max_training / (float)10.0 , "unstable shrinking");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();

                gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                            ratings_rows_GU, ratings_cols, 
                                            &num_latent_factors, percent,
                                            full_ratingsMtx_dev_GU, U_GU_dev, 
                                            V_dev, SV_with_U, SV_dev);

                //save_device_mtx_to_file<float>(U_GU_dev, ratings_rows_GU, num_latent_factors, "U_GU_compressed");
                /*
                    At this point U_GU is ratings_rows_GU by ratings_rows_GU in memory stored in column major
                    ordering and V is ratings_cols by ratings_rows_GU stored in column major ordering

                    There is no extra work to compress U_GU into ratings_rows_GU by num_latent_factors, or
                    to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
                    columns of each matrix
                */
            }  
    

            


            int *   csr_format_ratingsMtx_userID_dev_testing_      = NULL;
            int *   coo_format_ratingsMtx_itemID_dev_testing_      = NULL;
            float * coo_format_ratingsMtx_rating_dev_testing_      = NULL;
            int*    csr_format_ratingsMtx_userID_dev_testing_batch = NULL;
            int*    coo_format_ratingsMtx_itemID_dev_testing_batch = NULL;
            float*  coo_format_ratingsMtx_rating_dev_testing_batch = NULL;
            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing_,  (batch_size_testing + 1) * sizeof(int)));
                update_Mem((batch_size_testing + 1) * sizeof(int) );
            }

            for(int batch__ = 0; batch__ < std::min(10, (int)num_batches_testing); batch__++){
                //for(int batch = 0; batch < num_batches_testing; batch++){

                LOG(std::endl<<"                                          ~~~ TESTING Batch "<<batch__<<" ( / "<<num_batches_testing<<" ) ~~~ "); 
                int batch = 0;
                getRandIntsBetween(&batch , 0 , (int)num_batches_testing - 1, 1);
                LOG("batch id : "<<batch);

                long long int batch_size_testing_temp = batch_size_testing;
                long long int first_row_index_in_batch_testing  = (batch_size_testing * (long long int)batch) /* % ratings_rows_testing*/;
                if(first_row_index_in_batch_testing + batch_size_testing >= ratings_rows_testing) {
                    batch_size_testing_temp = ratings_rows_testing - first_row_index_in_batch_testing;
                    if(Debug){
                        LOG("left over batch_size_testing : "<<batch_size_testing_temp);
                    }
                } 
                if(Debug){
                    LOG(memLeft<<" available bytes left on the device");
                    LOG("first_row_index_in_batch_testing : "<<first_row_index_in_batch_testing<< " ( / "<<ratings_rows_testing<<" )"); 
                };




                long long int nnz_testing;
                long long int first_coo_ind_testing;
                if(Conserve_GPU_Mem){
                    if(Debug && 0){
                        save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_host_testing_2");
                    }
                    csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_host_testing +  first_row_index_in_batch_testing;
                    first_coo_ind_testing = csr_format_ratingsMtx_userID_dev_testing_batch[0];
                    int last_entry_index = (csr_format_ratingsMtx_userID_dev_testing_batch + batch_size_testing_temp)[0];

                    nnz_testing = (long long int)last_entry_index - first_coo_ind_testing;
                    LOG("first_coo_ind_testing : "<<first_coo_ind_testing  << " ( / "<< num_entries_testing<< " )");
                    LOG("last_entry_index : "<<last_entry_index);
                    LOG("nnz_testing : "<<nnz_testing);

                    if(nnz_testing <= 0){
                        LOG("nnz_testing : "<<nnz_testing);
                        ABORT_IF_EQ(0, 0, "nnz_testing <= 0");
                    }
                    
                    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing_,  nnz_testing        * sizeof(int)));
                    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing_,  nnz_testing        * sizeof(float)));
                    update_Mem(nnz_testing * sizeof(int) );
                    update_Mem(nnz_testing * sizeof(float) );

                    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_testing_,  csr_format_ratingsMtx_userID_dev_testing_batch,  (batch_size_testing_temp + 1) *  sizeof(int),   cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_testing_,  coo_format_ratingsMtx_itemID_host_testing + first_coo_ind_testing, nnz_testing  *  sizeof(int),   cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_testing_,  coo_format_ratingsMtx_rating_host_testing + first_coo_ind_testing,  nnz_testing  *  sizeof(float), cudaMemcpyHostToDevice));
                    
                    csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing_;
                    coo_format_ratingsMtx_itemID_dev_testing_batch = coo_format_ratingsMtx_itemID_dev_testing_;
                    coo_format_ratingsMtx_rating_dev_testing_batch = coo_format_ratingsMtx_rating_dev_testing_;
                }else{
                    csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing +  first_row_index_in_batch_testing;
                    nnz_testing = gpu_get_num_entries_in_rows(0, batch_size_testing_temp - 1, csr_format_ratingsMtx_userID_dev_testing_batch);
                    if(nnz_testing <=0){
                        LOG("nnz_testing : "<<nnz_testing);
                        ABORT_IF_EQ(0, 0, "nnz_testing <= 0");
                    }
                    first_coo_ind_testing = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_testing_batch);
                    
                    coo_format_ratingsMtx_itemID_dev_testing_batch = coo_format_ratingsMtx_itemID_dev_testing +  first_coo_ind_testing;
                    coo_format_ratingsMtx_rating_dev_testing_batch = coo_format_ratingsMtx_rating_dev_testing +  first_coo_ind_testing;
                }

                ABORT_IF_LESS(nnz_testing, 1, "nnz < 1");
                
                float* coo_testing_errors;
                float* testing_entries;
                checkCudaErrors(cudaMalloc((void**)&coo_testing_errors, nnz_testing * sizeof(float)));
                checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_testing * sizeof(float)));
                update_Mem(2 * nnz_testing * sizeof(float));

                if(Debug && 0){
                    LOG("testing requires " <<2 * nnz_testing * sizeof(float) + batch_size_testing_temp  * std::max(min_dim_, batch_size_testing_temp) * sizeof(float) 
                        + batch_size_testing_temp  * ratings_cols * sizeof(float) +
                        (batch_size_testing_temp + 1) * sizeof(int) + nnz_testing * sizeof(int) + nnz_testing * sizeof(float) << " bytes of memory");
                    LOG("first_coo_ind in this TESTING batch : "<<first_coo_ind_testing<< " ( / "<<num_entries_testing<<" )");
                    LOG("nnz in this TESTING batch : "<<nnz_testing);
                    LOG("( next first_coo_ind in TESTING batch : "<<first_coo_ind_testing + nnz_testing<<" )");
                    // save_device_mtx_to_file<float>(R, batch_size_training, ratings_cols, "error", false);
                    // LOG("Press Enter to continue.") ;
                    // std::cin.ignore();
                }
                //============================================================================================
                // Compute  R_testing * V = U_testing
                // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
                //============================================================================================ 
                float * U_testing;
                float * R_testing;
                checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing_temp  * (compress_when_testing ? num_latent_factors : ratings_rows_GU)  * sizeof(float)));
                checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing_temp  * ratings_cols                        * sizeof(float)));
                update_Mem(batch_size_testing_temp  * (compress_when_testing ? num_latent_factors : ratings_rows_GU)  * sizeof(float));
                update_Mem(batch_size_testing_temp  * ratings_cols                        * sizeof(float));
                // gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
                //                    batch_size_testing_temp, ratings_rows_GU, num_latent_factors, ratings_cols,
                //                    nnz_testing, first_coo_ind_testing, compress, 
                //                    testing_entries, coo_testing_errors, testing_fraction,
                //                    coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
                //                    csr_format_ratingsMtx_userID_dev_testing_batch, 
                //                    coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
                //                    V, U_testing, R_testing, "testing", (float)0.1, (float)0.01);


                gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
                                   batch_size_testing_temp, ratings_rows_GU, num_latent_factors, ratings_cols,
                                   nnz_testing, first_coo_ind_testing, compress_when_testing, 
                                   testing_entries, coo_testing_errors, testing_fraction,
                                   coo_format_ratingsMtx_rating_dev_testing_batch, 
                                   csr_format_ratingsMtx_userID_dev_testing_batch, 
                                   coo_format_ratingsMtx_itemID_dev_testing_batch,
                                   V_dev, U_testing, R_testing, micro_training_rate, regularization_constant, batch__,
                                   &testing_error_on_training_entries_temp, testing_error + num_tests, 
                                   &total_testing_iterations, SV_with_U, SV_dev);
                checkCudaErrors(cudaFree(U_testing));
                checkCudaErrors(cudaFree(R_testing));
                update_Mem(batch_size_testing_temp  * (compress_when_testing ? num_latent_factors : ratings_rows_GU)  * sizeof(float) * (-1));
                update_Mem(batch_size_testing_temp  * ratings_cols                        * sizeof(float) * (-1));
                //gpu_reverse_bools<float>(nnz_testing,  testing_entries);
                //gpu_hadamard<float>(nnz_testing, testing_entries, coo_testing_errors );
                //save_device_arrays_side_by_side_to_file<float>(coo_testing_errors, testing_entries, nnz_testing, "testing_entry_errors");

                //testing_error_temp += gpu_sum_of_squares<float>(nnz_testing, coo_testing_errors);
                
                
                checkCudaErrors(cudaFree(coo_testing_errors));
                checkCudaErrors(cudaFree(testing_entries));
                update_Mem(2 * nnz_testing * sizeof(float) * (-1));  

                if(Conserve_GPU_Mem){
                    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing_));
                    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing_));
                    update_Mem(nnz_testing * sizeof(int) * (-1));
                    update_Mem(nnz_testing * sizeof(float) * (-1));
                }
            }//for loop on test batches
            if(1){
                LOG("Iteration "<<it<<" MSQ testing error per training entry: "<< testing_error_on_training_entries_temp);
                LOG("Iteration "<<it<<" expected number of training iterations: "<< total_testing_iterations);
                LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< testing_error[num_tests]);
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }
            //testing_error[num_tests] /= ((float)num_entries_testing * testing_fraction);
            //LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< testing_error[num_tests]);

            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing_));
                update_Mem((batch_size_testing + 1) * sizeof(int) * (-1));
            }else{
                if(row_major_ordering){
                    //remember that R_GU is stored in row major ordering
                    LOG("swap matrix indexing from row major to column major");
                    gpu_swap_ordering<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, !row_major_ordering);
                    
                }
            }
            checkCudaErrors(cudaFree(SV_dev));
            update_Mem(ratings_rows_GU * sizeof(float) * (-1));
            //LOG("HERE!"); checkCudaErrors(cudaDeviceSynchronize()); LOG("HERE!");

        

            //============================================================================================
            // Assume the rows of R_GU are the ratings_rows_GU-means for the training users - what is the error?
            //============================================================================================ 


            if(0){
                float* errors;
                int* selection;
                if(Conserve_GPU_Mem){
                    errors  = (float *)malloc(ratings_rows_training * sizeof(float));
                    selection  = (int *)malloc(ratings_rows_training * sizeof(int));
                    checkErrors(errors);
                    checkErrors(selection);
                    cpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_host_GU, 
                                                 ratings_rows_training, num_entries_training, 
                                                 csr_format_ratingsMtx_userID_host_training, 
                                                 coo_format_ratingsMtx_itemID_host_training,
                                                 coo_format_ratingsMtx_rating_host_training, 
                                                 selection, errors);
                    k_means_er = cpu_sum(ratings_rows_training,  errors);
                    LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                    LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                    save_host_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
                    save_host_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
                    free(errors);
                    free(selection);
                }else{
                    checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
                    checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
                    update_Mem(2 * ratings_rows_training * sizeof(int));
                    gpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
                                             ratings_rows_training, num_entries_training, 
                                             csr_format_ratingsMtx_userID_dev_training, 
                                             coo_format_ratingsMtx_itemID_dev_training,
                                             coo_format_ratingsMtx_rating_dev_training, 
                                             selection, errors, row_major_ordering);
                    k_means_er = gpu_sum(ratings_rows_training,  errors);
                    LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                    LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                    save_device_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
                    save_device_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
                    checkCudaErrors(cudaFree(errors));
                    checkCudaErrors(cudaFree(selection));
                    update_Mem(2 * ratings_rows_training * sizeof(int) * (-1));
                }
            }
            LOG("      ~~~ DONE TESTING ~~~ "<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl); 
            if(it > 0){
              if(testing_error[num_tests] < epsilon || ::isinf(testing_error[num_tests]) || ::isnan(testing_error[num_tests])){
                LOG("Finished at iteration : "<<it);
                do_ = false;
                not_done = false;
                break;
              }

              if ((testing_error[num_tests - 1] - testing_error[num_tests]) < -(float)2.0 * min_error_so_far){
                meta_training_rate = meta_training_rate / (float)10.0;
                if(Debug) {
                  LOG("Jumped over minimum iteration : "<<it);
                  LOG("min_error_so_far : "<<min_error_so_far);
                  LOG("new error : "<<testing_error[num_tests]);
                  LOG("meta_training_rate : "<<meta_training_rate);
                }
                //we jumped over the minimum
                if(Conserve_GPU_Mem){
                    host_copy(ratings_rows_GU * ratings_cols, old_R_GU, full_ratingsMtx_host_GU);
                }else{
                    checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_GU, old_R_GU,
                                            ratings_rows_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
                }
                do_ = false;

              }else{
                  //have we stopped improving?
                  if(std::abs(testing_error[num_tests - 1] - testing_error[num_tests]) < meta_training_rate / (float)10.0){
                    meta_training_rate = meta_training_rate / (float)10.0;
                    LOG("Reducing meta_training_rate : "<<it);
                    LOG("previous error : "<<testing_error[num_tests - 1]);
                    LOG("new error : "<<testing_error[num_tests]);
                    LOG("diff : "<<std::abs(testing_error[num_tests - 1] - testing_error[num_tests]));
                    LOG("meta_training_rate : "<<meta_training_rate);
                  }
              }

            }
            min_error_so_far = std::min(min_error_so_far, testing_error[num_tests]);
            num_tests += 1;
            save_host_array_to_file(testing_error, num_tests, "meta_testing_error", strPreamble(blank));
        }//end is testing?
        

        //============================================================================================
        // TRAINING
        //============================================================================================ 

        if(meta_training_rate < min_training_rate || do_ == false) break;
        int count_GU_rounds = 0;
        float training_error_temp = (float)0.0;

        long long int total_training_nnz = (long long int)0;
        testing_error_on_training_entries_temp = (float)0.0;
        float total_training_iterations = (float)0.0;
        float max_abs_delta_R_GU = (float)0.0;
        float exp_abs_delta_R_GU = (float)0.0;
        float max_abs_R_GU = (float)0.0;
        float exp_abs_R_GU = (float)0.0;

        int *   csr_format_ratingsMtx_userID_dev_training_      = NULL;
        int *   coo_format_ratingsMtx_itemID_dev_training_      = NULL;
        float * coo_format_ratingsMtx_rating_dev_training_      = NULL;
        int*    csr_format_ratingsMtx_userID_dev_training_batch = NULL;
        int*    coo_format_ratingsMtx_itemID_dev_training_batch = NULL;
        float*  coo_format_ratingsMtx_rating_dev_training_batch = NULL;
        if(Conserve_GPU_Mem){
            checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_training_,  (batch_size_training + 1) * sizeof(int)));
            update_Mem((batch_size_training + 1) * sizeof(int) );
        }


        for(int batch__ = 0; batch__ < (int)std::min(num_batches, num_blocks); batch__++){
        //for(int batch = 0; batch < num_batches; batch++){

            if(meta_training_rate < min_training_rate || do_ == false) break;
            if( print_training_error){
                //LOG(std::endl<<"                                       ~ITERATION "<<it<<", BATCH "<<batch__<<"~"<<std::endl);
                LOG(std::endl<<std::endl<<"                              ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch__<<" ( / "<<num_batches<<" ), GU Round "<<count_GU_rounds<<" ( / "<<num_batches / num_blocks<<" )");
            }
            if(Debug && 0){
                //LOG(std::endl<<"                              ITERATION "<<it<<", BATCH "<<batch__);
                LOG(std::endl<<std::endl<<"                              ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch__<<" ( / "<<num_batches<<" ), GU Round "<<count_GU_rounds<<" ( / "<<num_batches / num_blocks<<" )");
            }
 
            int batch = 0;
            getRandIntsBetween(&batch , 0 , (int)num_batches - 1, 1);
            LOG("batch id : "<<batch);
            
            long long int batch_size_training_temp = batch_size_training;
            long long int first_row_in_batch_training = batch_size_training * (long long int)batch; /* % ratings_rows_testing*/;
            if(first_row_in_batch_training + batch_size_training >= ratings_rows_training) {
                batch_size_training_temp = batch_size_training - first_row_in_batch_training;
                if(Debug){
                    LOG("left over batch_size_training : "<<batch_size_training_temp);
                }
            }             
            long long int GU_batch = (long long int)batch__ % num_blocks;
            long long int first_row_in_batch_GU = (batch_size_GU * (long long int)GU_batch) ;            
            //============================================================================================
            // Find U_GU, V such that U_GU * V^T ~ R_GU 
            //============================================================================================  
            





            if(Debug){
                LOG("GU_batch : "<< GU_batch <<" ( / "<<num_blocks<<" )");
                LOG(memLeft<<" available bytes left on the device");
                //LOG("num_latent_factors = "<< num_latent_factors);
            }



            if(Debug){
                LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");
                LOG("batch_size_training : "<<batch_size_training_temp);
                //LOG("( next first_row_in_batch_training : "<<first_row_in_batch_training + batch_size_training_temp<<" )");
                LOG("first_row_in_batch_GU : "<<first_row_in_batch_GU<<  " ( / "<<ratings_rows_GU<<" )");
                LOG("batch_size_GU : "<<batch_size_GU);
                //LOG("( next first_row_in_batch_GU : "<<first_row_in_batch_GU + batch_size_GU<<" )");
            };


            float* full_ratingsMtx_dev_GU_current_batch = NULL;
            if(Conserve_GPU_Mem){
                // old way
                checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_GU_current_batch, batch_size_GU * ratings_cols * sizeof(float)));
                update_Mem(batch_size_GU * ratings_cols* sizeof(float) );
                checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_GU_current_batch, full_ratingsMtx_host_GU + ratings_cols * first_row_in_batch_GU, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyHostToDevice));

                checkCudaErrors(cudaMalloc((void**)&U_GU_dev,  batch_size_GU  * min_dim_block  * sizeof(float)));
                update_Mem(batch_size_GU  * min_dim_block  * sizeof(float));
            }else{
                full_ratingsMtx_dev_GU_current_batch = full_ratingsMtx_dev_GU + ratings_cols * first_row_in_batch_GU;
            }

            if(row_major_ordering){
                //remember that ratings_GU is stored in row major ordering
                LOG("swap matrix indexing from row major to column major");
                gpu_swap_ordering<float>(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, row_major_ordering);
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

            float * SV_dev;
            checkCudaErrors(cudaMalloc((void**)&SV_dev, batch_size_GU * sizeof(float)));

            gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                        batch_size_GU, ratings_cols, 
                                        &num_latent_factors, percent,
                                        full_ratingsMtx_dev_GU_current_batch, 
                                        U_GU_dev, V_dev, SV_with_U, SV_dev);

            checkCudaErrors(cudaMemcpy(U_GU_host, U_GU_dev, batch_size_GU * min_dim_block, cudaMemcpyDeviceToHost));

            //save_device_mtx_to_file<float>(U_GU, ratings_rows_GU, num_latent_factors, "U_GU_compressed");

            /*
                At this point U_GU is batch_size_GU by batch_size_GU in memory stored in column major
                ordering and V is ratings_cols by batch_size_GU stored in column major ordering

                There is no extra work to compress U_GU into batch_size_GU by num_latent_factors, or
                to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
                columns of each matrix
            */
             

            //============================================================================================
            // Compute  R_training * V = U_training
            // Compute  Error = R_training -  U_training * V^T  <-- sparse
            //============================================================================================ 
            //if(Debug) LOG("iteration "<<it<<" made it to check point");

            long long int nnz_training;
            long long int first_coo_ind_training;
            if(Conserve_GPU_Mem){
                if(Debug && 0){
                    save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_host_training_2");
                }
                csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_host_training +  first_row_in_batch_training;
                first_coo_ind_training = csr_format_ratingsMtx_userID_dev_training_batch[0];
                int last_entry_index = (csr_format_ratingsMtx_userID_dev_training_batch + batch_size_training_temp)[0];

                nnz_training = (long long int)last_entry_index - first_coo_ind_training;
                LOG("first_coo_ind_training : "<<first_coo_ind_training);
                LOG("last_entry_index : "<<last_entry_index);
                LOG("nnz_training : "<<nnz_training);

                if(nnz_training <= 0){
                    LOG("nnz_training : "<<nnz_training);
                    ABORT_IF_EQ(0, 0, "nnz_training <= 0");
                }
                
                checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_training_,  nnz_training        * sizeof(int)));
                checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_training_,  nnz_training        * sizeof(float)));
                update_Mem(nnz_training * sizeof(int) );
                update_Mem(nnz_training * sizeof(float) );

                checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_training_,  csr_format_ratingsMtx_userID_dev_training_batch,  (batch_size_training_temp + 1) *  sizeof(int),   cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_training_,  coo_format_ratingsMtx_itemID_host_training + first_coo_ind_training, nnz_training  *  sizeof(int),   cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_training_,  coo_format_ratingsMtx_rating_host_training + first_coo_ind_training,  nnz_training  *  sizeof(float), cudaMemcpyHostToDevice));
                
                csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_dev_training_;
                coo_format_ratingsMtx_itemID_dev_training_batch = coo_format_ratingsMtx_itemID_dev_training_;
                coo_format_ratingsMtx_rating_dev_training_batch = coo_format_ratingsMtx_rating_dev_training_;
            }else{
                csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_dev_training +  first_row_in_batch_training;
                nnz_training = gpu_get_num_entries_in_rows(0, batch_size_training_temp - 1, csr_format_ratingsMtx_userID_dev_training_batch);
                if(nnz_training <=0){
                    LOG("nnz_training : "<<nnz_training);
                    ABORT_IF_EQ(0, 0, "nnz_training <= 0");
                }
                first_coo_ind_training = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_training_batch);
                
                coo_format_ratingsMtx_itemID_dev_training_batch = coo_format_ratingsMtx_itemID_dev_training + first_coo_ind_training;
                coo_format_ratingsMtx_rating_dev_training_batch = coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training;
            }
            ABORT_IF_LESS(nnz_training, 1, "nnz < 1");
            total_training_nnz += nnz_training;


            float* coo_training_errors;
            //float* testing_entries;
            checkCudaErrors(cudaMalloc((void**)&coo_training_errors, nnz_training * sizeof(float)));
            //checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_training * sizeof(float)));
            update_Mem(nnz_training * sizeof(float));

            if(Debug && 0){
                LOG("training requires " <<nnz_training * sizeof(float) + batch_size_training  * std::max(min_dim_block, batch_size_training) * sizeof(float) 
                            + batch_size_training  * ratings_cols * sizeof(float) +
                            (batch_size_training + 1) * sizeof(int) + nnz_training * sizeof(int) + nnz_training * sizeof(float) << " bytes of memory");

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

            float * U_training;
            float * R_training;
            checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training_temp * (compress_when_training ? num_latent_factors : min_dim_block) * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training_temp * ratings_cols                        * sizeof(float)));
            update_Mem(batch_size_training_temp * (compress_when_training ? num_latent_factors : min_dim_block) * sizeof(float));
            update_Mem(batch_size_training_temp * ratings_cols                        * sizeof(float));
            

            gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
                               batch_size_training_temp, batch_size_GU, num_latent_factors, ratings_cols,
                               nnz_training, first_coo_ind_training, compress_when_training, 
                               NULL, coo_training_errors, (float)0.0,
                               coo_format_ratingsMtx_rating_dev_training_batch, 
                               csr_format_ratingsMtx_userID_dev_training_batch, 
                               coo_format_ratingsMtx_itemID_dev_training_batch,
                               V_dev, U_training, R_training, micro_training_rate, regularization_constant, batch__,
                               &testing_error_on_training_entries_temp, NULL, 
                               &total_training_iterations, SV_with_U, SV_dev);

            

            // gpu_R_error_training<float>(dn_handle, sp_handle, sp_descr,
            //                            batch_size_training_temp, batch_size_GU, num_latent_factors, ratings_cols,
            //                            nnz_training, first_coo_ind_training, false /*compress_when_training*/, coo_training_errors,
            //                            coo_format_ratingsMtx_rating_dev_training_batch, 
            //                            csr_format_ratingsMtx_userID_dev_training_batch,         // <-- already has shifted to correct start
            //                            coo_format_ratingsMtx_itemID_dev_training_batch,
            //                            V_dev, U_training, R_training, training_rate, regularization_constant, SV_with_U, SV_dev);

            checkCudaErrors(cudaFree(R_training));
            update_Mem(batch_size_training_temp * ratings_cols * sizeof(float) * (-1));

            if (SV_dev) checkCudaErrors(cudaFree(SV_dev));
            update_Mem(min_dim_ * sizeof(float) * (-1));
            

            training_error_temp += gpu_sum_of_squares<float>(nnz_training, coo_training_errors);

            if(num_blocks - 1 == GU_batch){
                if( print_training_error ){
                    //LOG("           ~Finished round "<<count_GU_rounds<<" of GA training~"<<std::endl); 
                    long long int nnz_ = (long long int)((float)total_training_nnz /* * testing_fraction*/);
                    //LOG("TRAINING AVERAGE SQUARED ERROR : "<< training_error_temp / (float)(nnz_));
                    LOG("TRAINING AVERAGE SQUARED ERROR : "<< testing_error_on_training_entries_temp); 

                    // float temp = gpu_sum_of_squares<float>(nnz_training, testing_entries);
                    // float temp = gpu_sum_of_squares_of_diff(dn_handle, nnz_training, 
                    //                                         coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, 
                    //                                         testing_entries);
                    
                    // LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));
                    // LOG("training error normalized: "<< training_error_temp / temp<<std::endl); 
                    // LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 

                } 
                count_GU_rounds += 1; 
                training_error_temp = (float)0.0; 
                total_training_nnz = (long long int)0;
                
            }

            if(Debug){

                //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, testing_entries, coo_training_errors, nnz_training, "ratings_testing_errors");
                
                float coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_training_errors);
                float coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_training_errors);
                LOG("coo_training_errors_abs_max = "<<coo_training_errors_abs_max) ;
                //LOG("coo_training_errors_abs_max over range of ratings = "<<coo_training_errors_abs_max / range_training) ;
                LOG("coo_training_errors_abs_exp = "<<coo_training_errors_abs_exp) ;
                //LOG("coo_training_errors_abs_exp over range of ratings = "<<coo_training_errors_abs_exp / range_training) ;

                // coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                // coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                // LOG("coo_training_abs_max = "<<coo_training_errors_abs_max) ;
                // LOG("coo_training_abs_exp = "<<coo_training_errors_abs_exp) ;

                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }

            //checkCudaErrors(cudaFree(testing_entries));




            //============================================================================================
            // Update  V = V * (1 - alpha * lambda) + alpha * Error^T * U_training 
            // (Update  U = U * (1 - alpha * lambda) + alpha * Error * V ) <- random error?????
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
                checkCudaErrors(cudaMalloc((void**)&delta_V, ratings_cols * min_dim_block * sizeof(float)));
                checkCudaErrors(cudaMemcpy(delta_V, V_dev, ratings_cols * min_dim_block * sizeof(float), cudaMemcpyDeviceToDevice));
                update_Mem(ratings_cols * min_dim_block * sizeof(float));
            }

            float alpha =  meta_training_rate;
            float beta = (float)1.0 - alpha * regularization_constant;
            gpu_spXdense_MMM<float>(sp_handle, true, false, batch_size_training_temp, min_dim_block, 
                                    ratings_cols, nnz_training, first_coo_ind_training, &alpha, sp_descr, 
                                    coo_training_errors, 
                                    csr_format_ratingsMtx_userID_dev_training_batch, 
                                    coo_format_ratingsMtx_itemID_dev_training_batch,
                                    U_training, batch_size_training_temp, &beta, V_dev, ratings_cols, false);

            if(normalize_V_rows && SV_with_U){
                LOG("Normalizing the rows of V...");
                gpu_normalize_mtx_rows_or_cols(ratings_cols, min_dim_block,  
                                      false, V_dev, false);
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
                                batch_size_GU, (compress_when_training == false) ? batch_size_GU : num_latent_factors, 
                                ratings_cols,
                                beta, //(float)1.0
                                V, training_rate, beta,
                                U_GU);
            */


            

            if(1){
                // float* copy;
                // checkCudaErrors(cudaMalloc((void**)&copy, ratings_cols * min_dim_block * sizeof(float)));
                // update_Mem(ratings_cols * min_dim_block * sizeof(float) );
                // checkCudaErrors(cudaMemcpy(copy, delta_V, ratings_cols * min_dim_block * sizeof(float), cudaMemcpyDeviceToDevice));
                gpu_axpby<float>(dn_handle, ratings_cols * min_dim_block, 
                                 (float)(-1.0), V_dev,
                                 (float)(1.0), delta_V);
                float delta_abs_exp = gpu_expected_abs_value<float>(ratings_cols * min_dim_block, delta_V);
                float delta_abs_max = gpu_abs_max<float>(ratings_cols * min_dim_block, delta_V); 
                LOG("delta V maximum absolute value = "<<delta_abs_max) ;
                LOG("delta V expected absolute value = "<<delta_abs_exp) ;
                ABORT_IF_EQ(delta_abs_max, delta_abs_exp, "delta V is constant");
                // save_device_arrays_side_by_side_to_file(copy, V, delta_V, ratings_cols * min_dim_block, "old_new_delta_V");
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
                if(delta_V) {
                    checkCudaErrors(cudaFree(delta_V));
                    update_Mem(ratings_cols * min_dim_block * sizeof(float) * (-1));
                }
                // checkCudaErrors(cudaFree(copy));
                // update_Mem(ratings_cols * min_dim_block * sizeof(float) * (-1));
            }

        


            if(update_U){
                LOG("Update U_GU...");

                // if(compress_when_training){
                //     gpu_scale<float>(dn_handle, batch_size_GU * num_latent_factors, beta, U_GU_dev);
                // }else{
                //     gpu_scale<float>(dn_handle, batch_size_GU * batch_size_GU, beta, U_GU_dev);
                // }   

                float* errors;
                int* selection;
                checkCudaErrors(cudaMalloc((void**)&errors, batch_size_training_temp * sizeof(float)));
                checkCudaErrors(cudaMalloc((void**)&selection, batch_size_training_temp * sizeof(int)));
                

                // gpu_dense_nearest_row<float>(batch_size_GU, min_dim_block, U_GU_dev, 
                //                              batch_size_training_temp, U_training, 
                //                              selection, errors, false);

                gpu_sparse_nearest_row<float>(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, 
                                             batch_size_training_temp, nnz_training, 
                                             csr_format_ratingsMtx_userID_dev_training_batch, 
                                             coo_format_ratingsMtx_itemID_dev_training_batch,
                                             coo_format_ratingsMtx_rating_dev_training_batch, 
                                             selection, errors, false);

                //int min_selection = gpu_min<int>(batch_size_training_temp, selection);

                float k_means_er_training = gpu_sum(batch_size_training_temp,  errors);
                //LOG("err norm when clustering U rows : "<<std::sqrt(k_means_er));
                LOG("mean sqed err when clustering U rows : "<<k_means_er_training / (float)(nnz_training));
                if(Debug){
                    save_device_array_to_file<float>(errors, (int)batch_size_training_temp, "km_errors", strPreamble(blank));
                    save_device_array_to_file<int>(selection, (int)batch_size_training_temp, "km_selection", strPreamble(blank));                        
                }
                checkCudaErrors(cudaFree(errors));

                float* U_GU_old;
                if(Debug){
                    checkCudaErrors(cudaMalloc((void**)&U_GU_old, batch_size_GU * min_dim_block * sizeof(int)));
                    checkCudaErrors(cudaMemcpy(U_GU_old, U_GU_dev, batch_size_GU * min_dim_block * sizeof(float), cudaMemcpyDeviceToDevice));
                }

                //long long int skip = min_selection * batch_size_GU;
                if(Debug){
                    //LOG("min_selection : "<<min_selection);
                    // save_device_mtx_to_file<float>(U_GU_old + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_old_0", true, strPreamble(blank));
                    // save_device_mtx_to_file<float>(U_GU + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_0", true, strPreamble(blank));
                }
                gpu_calculate_KM_error_and_update(batch_size_GU, min_dim_block, U_GU_dev, 
                                                 batch_size_training_temp, U_training, 
                                                 selection, meta_training_rate, regularization_constant);
                checkCudaErrors(cudaFree(selection));

                if(Debug){
                    // save_device_mtx_to_file<float>(U_GU_old + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_old_1", true, strPreamble(blank));
                    // save_device_mtx_to_file<float>(U_GU + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_1", true, strPreamble(blank));
                
                    gpu_axpby<float>(dn_handle, (batch_size_GU * min_dim_block), (float)(-1.0), U_GU_dev, (float)(1.0), U_GU_old);
                
                    // save_device_mtx_to_file<float>(U_GU_old + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_old_2", true, strPreamble(blank));
                    // save_device_mtx_to_file<float>(U_GU + skip, 3, (compress_when_training == false) ? batch_size_GU : num_latent_factors, "U_GU_2", true, strPreamble(blank));
                
                    float delta_abs_exp = gpu_expected_abs_value<float>((batch_size_GU * min_dim_block), U_GU_old);
                    float delta_abs_max = gpu_abs_max<float>((batch_size_GU * min_dim_block), U_GU_old); 
                    LOG("delta U maximum absolute value = "<<delta_abs_max) ;
                    LOG("delta U expected absolute value = "<<delta_abs_exp) ;
                    ABORT_IF_EQ(delta_abs_max, delta_abs_exp, "delta U is constant");                    
                    checkCudaErrors(cudaFree(U_GU_old));
                }

            }

            //============================================================================================
            // Update  R_GU = U_GU * V^T
            //============================================================================================ 
            //if(Debug) LOG("iteration "<<it<<" made it to check point");
            if(GU_batch == 0){
                // store backup
                if(Conserve_GPU_Mem){
                    host_copy(ratings_rows_GU * ratings_cols, full_ratingsMtx_host_GU, old_R_GU);
                }else{
                    checkCudaErrors(cudaMemcpy(old_R_GU, full_ratingsMtx_dev_GU, 
                                            ratings_rows_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
                    gpu_swap_ordering<float>(batch_size_GU, ratings_cols, old_R_GU + ratings_cols * first_row_in_batch_GU, !row_major_ordering);
                }
            }
            float* delta_R_GU;
            if(1){
                checkCudaErrors(cudaMalloc((void**)&delta_R_GU, min_dim_block * ratings_cols * sizeof(float)));
                update_Mem(min_dim_block * ratings_cols * sizeof(float));
                checkCudaErrors(cudaMemcpy(delta_R_GU, full_ratingsMtx_dev_GU_current_batch, 
                                            min_dim_block * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
                
                if(Debug  && 0){
                    //save_device_mtx_to_file<float>(delta_R_GU, batch_size_GU, 1, "delta_R_GU_0", true, strPreamble(blank));
                    //save_device_mtx_to_file<float>(full_ratingsMtx_dev_GU, batch_size_GU, 1, "R_GU_0", true, strPreamble(blank));
                }
                
            }
            gpu_gemm<float>(dn_handle, true, false, 
                            ratings_cols, batch_size_GU, 
                            (compress_when_training == false) ? min_dim_block : num_latent_factors,
                            (regularize_R == true) ? meta_training_rate : (float)1.0,
                            V_dev, U_GU_dev, 
                            (regularize_R == true) ? beta : (float)0.0,
                            full_ratingsMtx_dev_GU_current_batch);
            if (regularize_R_distribution == true ){
                float* user_means_GU;
                float* user_var_GU;
                checkCudaErrors(cudaMalloc((void**)&user_means_GU, batch_size_GU * sizeof(float)));
                checkCudaErrors(cudaMalloc((void**)&user_var_GU,   batch_size_GU * sizeof(float)));
                update_Mem(2 * batch_size_GU * sizeof(float));
                center_rows(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, 
                            val_when_var_is_zero, user_means_GU,  user_var_GU);
                checkCudaErrors(cudaFree(user_means_GU));
                checkCudaErrors(cudaFree(user_var_GU));
                update_Mem(2 * batch_size_GU * sizeof(float) * (-1));
                if(Debug  && 0){
                    //save_device_mtx_to_file<float>(delta_R_GU, batch_size_GU, 1, "delta_R_GU_1", true, strPreamble(blank));
                    save_device_mtx_to_file<float>(full_ratingsMtx_dev_GU_current_batch, batch_size_GU, 1, "R_GU_1", true, strPreamble(blank));
                }
            }
            if(1){
                if(Debug  && 0){
                    save_device_mtx_to_file<float>(V_dev, ratings_cols, num_latent_factors, "V_compressed");
                    save_device_mtx_to_file<float>(U_GU_dev, batch_size_GU, num_latent_factors, "U_GU_compressed");
                }
                //float* copy;
                //checkCudaErrors(cudaMalloc((void**)&copy, batch_size_GU * ratings_cols * sizeof(float)));
                //update_Mem(batch_size_GU * ratings_cols * sizeof(float));
                //checkCudaErrors(cudaMemcpy(copy, delta_R_GU, batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToDevice));
                gpu_axpby<float>(dn_handle, batch_size_GU * ratings_cols, 
                                 (float)(-1.0), full_ratingsMtx_dev_GU_current_batch,
                                 (float)(1.0), delta_R_GU);
                float delta_abs_exp = gpu_expected_abs_value<float>(batch_size_GU * ratings_cols, delta_R_GU);
                float delta_abs_max = gpu_abs_max<float>(batch_size_GU * ratings_cols, delta_R_GU); 
                LOG("delta R_GU maximum absolute value = "<<delta_abs_max) ;
                LOG("delta R_GU expected absolute value = "<<delta_abs_exp) ;
                max_abs_delta_R_GU = std::max(max_abs_delta_R_GU, delta_abs_max);
                if(Debug  && 0){
                    // save_device_arrays_side_by_side_to_file(copy, full_ratingsMtx_dev_GU_current_batch, delta_R_GU, batch_size_GU * ratings_cols, "old_new_delta");
                    // LOG("Press Enter to continue.") ;
                    // std::cin.ignore();
                }
                checkCudaErrors(cudaFree(delta_R_GU));
                //checkCudaErrors(cudaFree(copy));
                update_Mem(/*2 * */batch_size_GU * ratings_cols * sizeof(float) * (-1));
            }
            

        

            if(row_major_ordering){
                //remember that ratings_GU is stored in row major ordering
                LOG("swap matrix indexing from column major to row major");
                gpu_swap_ordering<float>(batch_size_GU, ratings_cols, full_ratingsMtx_dev_GU_current_batch, !row_major_ordering);
            }
            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_GU + ratings_cols * first_row_in_batch_GU, full_ratingsMtx_dev_GU_current_batch, 
                                            batch_size_GU * ratings_cols * sizeof(float), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaFree(full_ratingsMtx_dev_GU_current_batch));
            };
            checkCudaErrors(cudaFree(U_training));
            update_Mem(batch_size_training_temp  * std::max(min_dim_block, batch_size_training_temp)  * sizeof(float) * (-1));
            checkCudaErrors(cudaFree(coo_training_errors));
            update_Mem(nnz_training * sizeof(float) * (-1));




            /*
                //============================================================================================
                // Assume the rows of R_GU are the ratings_rows_GU-means for the training users - what is the error?
                //============================================================================================ 

                float* errors;
                int* selection;
                if(Conserve_GPU_Mem){
                    errors  = (float *)malloc(ratings_rows_training * sizeof(float));
                    selection  = (int *)malloc(ratings_rows_training * sizeof(int));
                    checkErrors(errors);
                    checkErrors(selection);
                    cpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_host_GU, 
                                                 ratings_rows_training, num_entries_training, 
                                                 csr_format_ratingsMtx_userID_host_training, 
                                                 coo_format_ratingsMtx_itemID_host_training,
                                                 coo_format_ratingsMtx_rating_host_training, 
                                                 selection, errors);
                    float mean_guess_error = cpu_sum_of_squares<float>(num_entries_training, coo_format_ratingsMtx_rating_host_training);
                    float k_means_er = cpu_sum(ratings_rows_training,  errors);
                    LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                    LOG("err norm when clustering over err when guessing mean one cluster: "<<std::sqrt(k_means_er) / std::sqrt(mean_guess_error));
                    LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                    if(Debug && 0){
                        save_host_array_to_file<float>(errors, ratings_rows_training, "errors");
                        save_host_array_to_file<int>(selection, ratings_rows_training, "selection");
                    }
                    free(errors);
                    free(selection);
                }else{
                    checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
                    checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
                    update_Mem(2 * ratings_rows_training * sizeof(int));
                    gpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
                                             ratings_rows_training, num_entries_training, 
                                             csr_format_ratingsMtx_userID_dev_training, 
                                             coo_format_ratingsMtx_itemID_dev_training,
                                             coo_format_ratingsMtx_rating_dev_training, 
                                             selection, errors, row_major_ordering);
                    float k_means_er = gpu_sum(ratings_rows_training,  errors);
                    LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                    LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                    save_device_array_to_file<float>(errors, ratings_rows_training, "errors");
                    save_device_array_to_file<int>(selection, ratings_rows_training, "selection");
                    checkCudaErrors(cudaFree(errors));
                    checkCudaErrors(cudaFree(selection));
                    update_Mem(2 * ratings_rows_training * sizeof(int) * (-1));
                }
            */
            
            //sanity check
            // if(gpu_abs_max(nnz_training, coo_format_ratingsMtx_rating_dev_training_batch) > abs_max_training){
            //     ABORT_IF_EQ(0, 0, "abs_max_training should not grow");
            // }
            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training_));
                checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training_));
                checkCudaErrors(cudaFree(U_GU_dev));
                update_Mem(nnz_training * sizeof(int) * (-1) );
                update_Mem(nnz_training * sizeof(float) * (-1) ); 
                update_Mem(batch_size_GU  * min_dim_block * sizeof(float) * (-1) );
            }
            checkCudaErrors(cudaDeviceSynchronize());
            gettimeofday(&program_end, NULL);
            program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
            //printf("program_time: %f\n", program_time);   
            LOG("RUN TIME SO FAR: "<<readable_time(program_time));
            
            // count_batches_so_far++;
            // gettimeofday(&training_end, NULL);
            // training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
            // //LOG("average training iteration time : "<<readable_time(training_time / (double)(count_batches_so_far)));
            // //LOG("predicted run time left: "<<readable_time((training_time / (double)(count_batches_so_far)) * (double)(num_iterations * num_batches - count_batches_so_far))<<std::endl);
            // LOG("training_time : "<<readable_time(training_time));
        }//end for loop on batches




        if(Conserve_GPU_Mem){
            checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training_));
            update_Mem((batch_size_training + 1) * sizeof(int) * (-1));

        }

        if(num_blocks != (long long int)1){
            LOG("shuffle GU matrix rows");
            if(Conserve_GPU_Mem){
                // cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols, 
                //                              row_major_ordering, full_ratingsMtx_host_GU, 1);
            }else{
                //shuffle GU rows
                gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_GU, ratings_cols,  
                                            row_major_ordering, full_ratingsMtx_dev_GU, 1);


                //shuffle training rows??
            }
        }
        if(it % (num_iterations / 4) == 0){
           meta_training_rate =  meta_training_rate / (float)10.0;
        }
        if(1){
            LOG(std::endl<<std::endl<<std::endl);
            LOG("Iteration "<<it<<" MSQ training error: "<< testing_error_on_training_entries_temp);
            LOG("Iteration "<<it<<" expected number of training iterations: "<< total_training_iterations);
            LOG("Iteration "<<it<<" expected number of testing iterations: "<< total_testing_iterations);
            LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< testing_error[num_tests - 1]);
            LOG("Iteration "<<it<<" err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("Iteration "<<it<<" MSQ when clustering : "<<k_means_er / (float)num_entries_training);
            // LOG("Press Enter to continue.") ;
            // std::cin.ignore();
        }
        LOG("Finished Iteration "<<it<<"."<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl);
        it++;
    }//end for loop on iterations





    //============================================================================================
    // Assume the rows of R_GU are the ratings_rows_GU-means for the training users - what is the error?
    //============================================================================================ 


    if(1){
        float k_means_er = (float)0.0;
        float* errors;
        int* selection;
        if(Conserve_GPU_Mem){
            errors  = (float *)malloc(ratings_rows_training * sizeof(float));
            selection  = (int *)malloc(ratings_rows_training * sizeof(int));
            checkErrors(errors);
            checkErrors(selection);
            cpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_host_GU, 
                                         ratings_rows_training, num_entries_training, 
                                         csr_format_ratingsMtx_userID_host_training, 
                                         coo_format_ratingsMtx_itemID_host_training,
                                         coo_format_ratingsMtx_rating_host_training, 
                                         selection, errors);
            //k_means_er = cpu_sum(ratings_rows_training,  errors);
            //LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("expected MSQER when clustering : "<<cpu_expected_value(ratings_rows_training, errors));
            save_host_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_host_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            free(errors);
            free(selection);
        }else{
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * sizeof(int)));
            update_Mem(2 * ratings_rows_training * sizeof(int));
            gpu_sparse_nearest_row<float>(ratings_rows_GU, ratings_cols, full_ratingsMtx_dev_GU, 
                                     ratings_rows_training, num_entries_training, 
                                     csr_format_ratingsMtx_userID_dev_training, 
                                     coo_format_ratingsMtx_itemID_dev_training,
                                     coo_format_ratingsMtx_rating_dev_training, 
                                     selection, errors, row_major_ordering);
            k_means_er = gpu_sum(ratings_rows_training,  errors);
            LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
            save_device_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_device_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            checkCudaErrors(cudaFree(errors));
            checkCudaErrors(cudaFree(selection));
            update_Mem(2 * ratings_rows_training * sizeof(int) * (-1));
        }
    }




    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&training_end, NULL);
    training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
    LOG("training_time : "<<readable_time(training_time));
    //============================================================================================
    // Destroy
    //============================================================================================
    LOG("Cleaning Up...");
    //free(user_means_training_host);
    if (user_means_testing_host) free(user_means_testing_host);
    if (user_var_testing_host) free(user_var_testing_host);
    if (testing_error) free(testing_error);
    if (full_ratingsMtx_host_GU) { free(full_ratingsMtx_host_GU); }
    if (full_ratingsMtx_dev_GU) { checkCudaErrors(cudaFree(full_ratingsMtx_dev_GU)); }

    if (U_GU_host) free(U_GU_host);
    if (U_GU_dev) checkCudaErrors(cudaFree(U_GU_dev));
    if (V_dev) checkCudaErrors(cudaFree(V_dev));
    

    if(Conserve_GPU_Mem){
        if (V_host) free(V_host);
        if (csr_format_ratingsMtx_userID_host_testing) free(csr_format_ratingsMtx_userID_host_testing);
        if (coo_format_ratingsMtx_itemID_host_testing) free(coo_format_ratingsMtx_itemID_host_testing);
        if (coo_format_ratingsMtx_rating_host_testing) free(coo_format_ratingsMtx_rating_host_testing); 
        if (csr_format_ratingsMtx_userID_host_training) free(csr_format_ratingsMtx_userID_host_training);
        if (coo_format_ratingsMtx_itemID_host_training) free(coo_format_ratingsMtx_itemID_host_training);
        if (coo_format_ratingsMtx_rating_host_training) free(coo_format_ratingsMtx_rating_host_training); 
        if (old_R_GU) free(old_R_GU);
    }else{
        if (old_R_GU) checkCudaErrors(cudaFree(old_R_GU));
    }
    
    


    update_Mem((batch_size_GU * std::min(batch_size_GU, ratings_cols) +batch_size_training * batch_size_training + ratings_cols * std::min(batch_size_GU, ratings_cols))* static_cast<long long int>(sizeof(float))* (-1));
    

    if (user_means_GU) { checkCudaErrors(cudaFree(user_means_GU)); update_Mem(batch_size_GU * sizeof(float) * (-1)); }
    if (user_var_GU) { checkCudaErrors(cudaFree(user_var_GU));   update_Mem(batch_size_GU * sizeof(float) * (-1)); }


    if (csr_format_ratingsMtx_userID_dev_training) checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training));
    if (coo_format_ratingsMtx_itemID_dev_training) checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training));
    if (coo_format_ratingsMtx_rating_dev_training) checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training));
    
    if (csr_format_ratingsMtx_userID_dev_testing) checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing));
    if (coo_format_ratingsMtx_itemID_dev_testing) checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing));
    if (coo_format_ratingsMtx_rating_dev_testing) checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing));

 
    if (SV) free(SV);

    update_Mem((ratings_rows_GU * ratings_cols + /*ratings_rows_training * ratings_cols +*/ num_entries_testing + num_entries_training)* sizeof(float));
    update_Mem(( (ratings_rows_testing + 1)  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(float))
               + (ratings_rows_training + 1) * static_cast<long long int>(sizeof(int)) + num_entries_training * static_cast<long long int>(sizeof(int)) + num_entries_training * static_cast<long long int>(sizeof(float))
               + (ratings_rows_GU + 1)       * static_cast<long long int>(sizeof(int)) + num_entries_GU       * static_cast<long long int>(sizeof(int)) + num_entries_GU       * static_cast<long long int>(sizeof(float))) * (-1) );
    

    if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
        if (csr_format_keyWordMtx_itemID_dev) checkCudaErrors(cudaFree(csr_format_keyWordMtx_itemID_dev));          update_Mem((ratings_cols + 1)      * sizeof(int) * (-1));
        if (coo_format_keyWordMtx_keyWord_dev) checkCudaErrors(cudaFree(coo_format_keyWordMtx_keyWord_dev));          update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
    }

    cublasDestroy          (dn_handle);
    cusolverDnDestroy      (dn_solver_handle);
    cusparseDestroy        (sp_handle);
    cusparseDestroyMatDescr(sp_descr);
    cusolverSpDestroy      (sp_solver_handle);

    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("program_time : "<<readable_time(program_time));

    //if(Debug && memLeft!=devMem)LOG("WARNING POSSIBLE DEVICE MEMORY LEAK");
         
}
