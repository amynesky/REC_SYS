#ifndef JSONREADER_H
#define JSONREADER_H

#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
 
/*
 * A class to read data from a csv file.
 */
class JSONReader
{
	std::string fileName;
	std::string delimeter;
 
	public:
	JSONReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }
 
	// return the number of rows in a JSON File
	int num_rows();

	// return the number of rows in a JSON File
	int num_cols();

	// return the number of coo entries in a JSON File with multiple mtx entries per JSON row
	long long int num_entries();



	// Function to fetch data from a JSON File
	std::vector<std::vector<std::string> > getData();

	// Function to fetch data from a JSON File
	void getData(float* data, const int rows, const int cols);

	// Function to fetch data from a JSON File
	void getData(int* coo_format_ratingsMtx_userID_host,
			    int* coo_format_ratingsMtx_movieID_host,
			    float* coo_format_ratingsMtx_rating_host,
			    int num_entries);

	// Function to fetch data from a JSON File
	long long int  makeContentBasedcooKeyWordMtx(int* coo_format_keyWordMtx_movieID_host,
											   int* coo_format_keyWordMtx_keyWord_host,
											   const long long int num_entries);


};
 


#endif //JSONREADER_H