import getopt
import os
import sys

import h5py
import numpy as np

from osbenchmark.utils.dataset import Context, get_data_set, HDF5DataSet
from sklearn.neighbors import NearestNeighbors


class DeleteDataset(object):
    def __init__(self, input_file, output_file, percent):
        self.index_set = get_data_set(HDF5DataSet.FORMAT_NAME, path=input_file, context=Context.INDEX)
        self.query_set = get_data_set(HDF5DataSet.FORMAT_NAME, path=input_file, context=Context.QUERY)
        self.neighbor_original_set = get_data_set(HDF5DataSet.FORMAT_NAME, path=input_file, context=Context.NEIGHBORS)
        self.output_data_set = create_dataset_file(output_file)
        self.percent = float(percent)

    def generate_delete_set(self):
        corpus_size = self.index_set.size()

        total_docs_to_delete = int(corpus_size * self.percent)

        original_neighbors_size = self.neighbor_original_set.size()
        original_neighbors = self.neighbor_original_set.read(original_neighbors_size)
        unique_arr_tuples = np.unique(original_neighbors.flatten(), return_counts=True)
        indices_to_delete = np.random.choice(unique_arr_tuples[0], total_docs_to_delete)

        vectors = self.index_set.read(corpus_size)
        test_vectors = self.query_set.read(self.query_set.size())
        neighbors_deleted = self.generate_ground_truth_2(vectors, corpus_size, test_vectors, indices_to_delete)
        #neighbors_deleted_array = np.array([val for val in neighbors_deleted.values()])

        self.output_data_set.create_dataset('train', data=vectors)
        self.output_data_set.create_dataset('delete', data=indices_to_delete)
        self.output_data_set.create_dataset('test', data=test_vectors)
        self.output_data_set.create_dataset('neighbors', data=neighbors_deleted)
        self.output_data_set.flush()
        self.output_data_set.close()

    def generate_ground_truth(self, vectors, corpus_size, test_vectors, indices_to_delete):
        indices = np.arange(corpus_size)
        print("\nindices:", indices)
        updates_indices = np.delete(indices, indices_to_delete, axis=0)
        print("\nupdates_indices:", updates_indices)
        update_vectors = np.delete(vectors, indices_to_delete, axis=0)

        pre_computed_euclidean_dis = np.sum(update_vectors ** 2, axis=1)

        top_k_dataset = {}

        k = min(100, len(update_vectors))
        print("k", k)
        for i, t_vector in enumerate(test_vectors):
            distance = pre_computed_euclidean_dis - 2 * np.dot(update_vectors, t_vector)
            sorted_indices = distance.argsort()
            top_k_indices = sorted_indices[:k]

            top_k = updates_indices[top_k_indices]
            top_k_dataset[i] = top_k

        return top_k_dataset

    def generate_ground_truth_2(self, vectors, corpus_size, test_vectors, indices_to_delete):
        indices = np.arange(corpus_size)
        updates_indices = np.delete(indices, indices_to_delete, axis=0)
        print("\nupdates_indices:", updates_indices)
        update_vectors = np.delete(vectors, indices_to_delete, axis=0)

        knn = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean')
        knn.fit(update_vectors)

        _, nearest_neighbor_indices = knn.kneighbors(test_vectors)

        print(nearest_neighbor_indices[0])

        return updates_indices[nearest_neighbor_indices]


def create_dataset_file(output_file) -> h5py.File:
    if os.path.isfile(output_file):
        os.remove(output_file)
    else:
        print(f"Creating the output file at {output_file}")
    data_set_w_filtering = h5py.File(output_file, 'a')

    return data_set_w_filtering


def main(args: list) -> None:
    opts, args = getopt.getopt(args, "", ["input_file_path=", "output_file_path=", "percent="])

    print(f'Options provided are: {opts}')
    print(f'Arguments provided are: {args}')
    input_file_path = None
    output_file_path = None
    percent = None
    for opt, arg in opts:
        if opt == '-h':
            print('--input_file_path <file_path> --output_file_path=<file_path> --percent <0-1>')
            sys.exit()
        if opt in '--input_file_path':
            input_file_path = arg
        elif opt in '--percent':
            percent = float(arg)
            print(f'--percent {percent}')
            if percent < 0 or percent > 1:
                raise Exception('--percent must be between 0 and 1')
        elif opt in '--output_file_path':
            output_file_path = arg

    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    DeleteDataset(input_file_path, output_file_path, percent).generate_delete_set()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
