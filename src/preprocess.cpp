#include "preprocess.h"

// void load_data(float ** &dataset, char * dataset_path, long int dataset_size, int data_dimensionality) {
//     cout << ">>> Loading dataset from: " << dataset_path << endl;

//     FILE * ifile_dataset;
//     ifile_dataset = fopen(dataset_path,"rb");
//     if (ifile_dataset == NULL) {
//         cout << "File " << dataset_path << "not found!" << endl;
//         exit(-1);
//     }

//     cout << "Cardinality of dataset is: " << dataset_size << endl;
//     int fread_return;

//     dataset = new float*[dataset_size];
//     for (int i = 0; i < dataset_size; i++)
//     {
//         dataset[i] = new float[data_dimensionality];
//         fread_return = fread(dataset[i], sizeof(float), data_dimensionality, ifile_dataset);                
//     }
//     fclose(ifile_dataset);
// }

// void load_query(float ** &querypoints, char * query_path, int query_size, int data_dimensionality) {
//     cout << ">>> Loading query from: " << query_path << endl;
//     FILE *ifile_query;
//     ifile_query = fopen(query_path,"rb");
//     if (ifile_query == NULL) {
//         cout << "File " << query_path << "not found!" << endl;
//         exit(-1);
//     }

//     int fread_return;

//     querypoints = new float*[query_size];
//     for (int i = 0; i < query_size; i++)
//     {
//         querypoints[i] = new float[data_dimensionality];
//         fread_return = fread(querypoints[i], sizeof(float), data_dimensionality, ifile_query);                
//     }
//     fclose(ifile_query);
// }

// void load_groundtruth(long int ** &gt, char * groundtruth_path, int query_size, int k_size) {
//     cout << ">>> Loading groundtruth from: " << groundtruth_path << endl;
//     FILE *ifile_groundtruth;
//     ifile_groundtruth = fopen(groundtruth_path,"rb");
//     if (ifile_groundtruth == NULL) {
//         cout << "File " << groundtruth_path << "not found!" << endl;
//         exit(-1);
//     }

//     int fread_return;

//     gt = new long int*[query_size];
//     for (int i = 0; i < query_size; i++)
//     {
//         gt[i] = new long int[k_size];
//         fread_return = fread(gt[i], sizeof(long int), k_size, ifile_groundtruth);                
//     }
//     fclose(ifile_groundtruth);
// }

void load_data(float ** &dataset, char * dataset_path, long int dataset_size, int data_dimensionality) {
    cout << ">>> Loading dataset from: " << dataset_path << endl;

    FILE *ifile_dataset = fopen(dataset_path, "rb");
    if (ifile_dataset == NULL) {
        cout << "File " << dataset_path << " not found!" << endl;
        exit(-1);
    }

    cout << "Cardinality of dataset is: " << dataset_size << endl;

    dataset = new float*[dataset_size];

    for (long int i = 0; i < dataset_size; i++) {
        int dim;
        if (fread(&dim, sizeof(int), 1, ifile_dataset) != 1) {
            cout << "Error reading dimension for dataset vector " << i << endl;
            exit(-1);
        }

        if (dim != data_dimensionality) {
            cout << "Dimension mismatch in dataset at vector " << i
                 << ": expected " << data_dimensionality
                 << ", got " << dim << endl;
            exit(-1);
        }

        dataset[i] = new float[data_dimensionality];

        if (fread(dataset[i], sizeof(float), data_dimensionality, ifile_dataset)
            != (size_t)data_dimensionality) {
            cout << "Error reading dataset vector " << i << endl;
            exit(-1);
        }
    }

    fclose(ifile_dataset);
}

void load_query(float ** &querypoints, char * query_path, int query_size, int data_dimensionality) {
    cout << ">>> Loading query from: " << query_path << endl;

    FILE *ifile_query = fopen(query_path, "rb");
    if (ifile_query == NULL) {
        cout << "File " << query_path << " not found!" << endl;
        exit(-1);
    }

    querypoints = new float*[query_size];

    for (int i = 0; i < query_size; i++) {
        int dim;
        if (fread(&dim, sizeof(int), 1, ifile_query) != 1) {
            cout << "Error reading dimension for query vector " << i << endl;
            exit(-1);
        }

        if (dim != data_dimensionality) {
            cout << "Dimension mismatch in query at vector " << i
                 << ": expected " << data_dimensionality
                 << ", got " << dim << endl;
            exit(-1);
        }

        querypoints[i] = new float[data_dimensionality];

        if (fread(querypoints[i], sizeof(float), data_dimensionality, ifile_query)
            != (size_t)data_dimensionality) {
            cout << "Error reading query vector " << i << endl;
            exit(-1);
        }
    }

    fclose(ifile_query);
}

void load_groundtruth(long int ** &gt, char * groundtruth_path, int query_size, int k_size) {
    cout << ">>> Loading groundtruth from: " << groundtruth_path << endl;

    FILE *ifile_groundtruth = fopen(groundtruth_path, "rb");
    if (ifile_groundtruth == NULL) {
        cout << "File " << groundtruth_path << " not found!" << endl;
        exit(-1);
    }

    gt = new long int*[query_size];

    for (int i = 0; i < query_size; i++) {
        int dim;
        if (fread(&dim, sizeof(int), 1, ifile_groundtruth) != 1) {
            cout << "Error reading dimension for groundtruth vector " << i << endl;
            exit(-1);
        }

        if (dim != k_size) {
            cout << "Groundtruth dimension mismatch at query " << i
                 << ": expected " << k_size
                 << ", got " << dim << endl;
            exit(-1);
        }

        gt[i] = new long int[k_size];

        for (int j = 0; j < k_size; j++) {
            int id;
            if (fread(&id, sizeof(int), 1, ifile_groundtruth) != 1) {
                cout << "Error reading groundtruth id at query " << i
                     << ", position " << j << endl;
                exit(-1);
            }

            gt[i][j] = (long int)id;
        }
    }

    fclose(ifile_groundtruth);
}

void transfer_data(float ** &dataset, vector<arma::mat> &data_list, long int dataset_size, int subspace_num, int subspace_dimensionality) {

    int kmeans_dim = subspace_dimensionality / 2;

    for (int subspace_index = 0; subspace_index < subspace_num; subspace_index++)
    {
        arma::mat data_first_half(kmeans_dim, dataset_size, arma::fill::zeros);
        arma::mat data_second_half(kmeans_dim, dataset_size, arma::fill::zeros);

        for (int i = 0; i < dataset_size; i++) {

            for (int j = 0; j < kmeans_dim; j++) {
                data_first_half(j, i)= dataset[i][subspace_index * subspace_dimensionality + j];
            }

            for (int j = kmeans_dim; j < kmeans_dim * 2; j++) {
                data_second_half(j - kmeans_dim, i) = dataset[i][subspace_index * subspace_dimensionality + j];
            }
        }

        data_list.push_back(data_first_half);
        data_list.push_back(data_second_half);

        cout << "Finish initialize the data of " << subspace_index << "-th subspace. " << endl;
    }
}