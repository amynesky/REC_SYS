#ifndef CSVREADER_H
#define CSVREADER_H

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
class CSVReader
{
	public:
	CSVReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }

	// return the number of rows in a CSV File
	int num_rows();

	// return the number of rows in a CSV File
	int num_cols();

	// return the number of coo entries in a CSV File with multiple mtx entries per CSV row
	long long int num_entries();



	// Function to fetch data from a CSV File
	std::vector<std::vector<std::string> > getData();

	// Function to fetch data from a CSV File
	template < typename Dtype>
	void getData(Dtype* data, const int rows, const int cols, bool row_major_ordering = false);

	// Function to fetch data from a CSV File
	void getData(int* coo_format_ratingsMtx_userID_host,
			    int* coo_format_ratingsMtx_itemID_host,
			    float* coo_format_ratingsMtx_rating_host,
			    int num_entries, bool all_items, 
			    std::map<int, int>* items_dictionary);

	void getDataJSON(int* coo_format_ratingsMtx_userID_host,
				    int* coo_format_ratingsMtx_itemID_host,
				    float* coo_format_ratingsMtx_rating_host,
				    int num_entries, const int rating_if_missing);

	// Function to fetch data from a CSV File
	long long int  makeContentBasedcooKeyWordMtx(int* coo_format_keyWordMtx_itemID_host,
											   int* coo_format_keyWordMtx_keyWord_host,
											   const long long int num_entries);
	private:
		std::string fileName;
		std::string delimeter;

};
 


#endif //CSVREADER_H