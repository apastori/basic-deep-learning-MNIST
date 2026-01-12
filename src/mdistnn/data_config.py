from os.path import join

#
# Set file paths based on added MNIST Datasets
#
input_path: str = '../data'

# Training data file paths dictionary
dict_config_train: dict[str, str] = {
    'input_path': input_path,
    'training_images_filepath': join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
    'training_labels_filepath': join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
}

# Test data file paths dictionary
dict_config_test: dict[str, str] = {
    'input_path': input_path,
    'test_images_filepath': join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
    'test_labels_filepath': join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'),
}

# Export dictionaries for use in other Python files
__all__: list[str] = ['dict_config_train', 'dict_config_test']