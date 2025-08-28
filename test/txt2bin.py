import numpy as np

def txt_to_npy(txt_path, npy_path, dtype=np.float32):
    # data = np.loadtxt(txt_path, dtype=dtype, delimiter=",")
    data = np.loadtxt(txt_path, dtype=dtype)
    np.save(npy_path, data.astype(dtype))
def txt_to_bin(txt_path, bin_path, dtype=np.float32):
    # data = np.loadtxt(txt_path, dtype=dtype, delimiter=",")
    data = np.loadtxt(txt_path, dtype=dtype)
    data.astype(dtype).tofile(bin_path)

def npy_to_bin(txt_path, bin_path, dtype=np.float32):
    data = np.load(txt_path)
    data.astype(dtype).tofile(bin_path)

def read_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.fromfile(bin_path, dtype=dtype).reshape((num_rows, num_cols))

def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.memmap(bin_path, dtype=dtype, mode='r', shape=(num_rows, num_cols))

if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/CEOs/"

    txt_file = path + "Netflix_X_17770_300.txt"    # Assume this file already exists
    bin_file = path + "Netflix_X_17770_300.bin"
    num_rows = 17770
    num_cols = 300

    txt_to_bin(txt_file, bin_file)

    txt_file = path + "Netflix_Q_999_300.txt"    # Assume this file already exists
    bin_file = path + "Netflix_Q_999_300.bin"
    num_rows = 999
    num_cols = 300
    txt_to_bin(txt_file, bin_file)


    # Convert .txt to .npy
    # txt_file = path + "Gist_Dot_k_1000_indices.txt"  # Assume this file already exists
    # npy_file = path + "Gist_Dot_k_1000_indices.npy"  # Assume this file already exists
    # txt_to_npy(txt_file, npy_file)
    #
    # txt_file = path + "Sift_Dot_k_1000_indices.txt"  # Assume this file already exists
    # npy_file = path + "Sift_Dot_k_1000_indices.npy"  # Assume this file already exists
    # txt_to_npy(txt_file, npy_file)

    # Convert .npy to .bin
    # npy_to_bin(txt_file, bin_file)

    # ==== STEP 2   : Load the binary file fully into memory ====

    # data = read_bin(bin_file, num_rows, num_cols)
    # print("Loaded full binary into memory")
    # print(data.shape)        # Should print: (8100000, 784)
    # print(data[0, :10])      # First 10 values of the first row

    # ==== STEP 3: Use memory-mapped version ====

    # data_mmap = mmap_bin(bin_file, num_rows, num_cols)
    # print("Memory-mapped binary")
    # print(data_mmap.shape)   # Same shape
    # print(data_mmap[0, :10])  # Access 124th row, first 10 values
