The C++ source code for budgeted MIPS (using CodeBlock)
This software implements several algorithms (see the PKDD/ECML 2020 papers and its arXiv version https://arxiv.org/abs/1908.08656)

To use Eigen Lib,
- In Project/Buid Options/Search Directories, add C:\_Data\Libraries\eigen-eigen-5a0156e40feb (where Eigen Lib locates)

Parameters: 
<PARAM_DATA_N> <PARAM_QUERY_Q> <PARAM_DATA_D> <PARAM_MIPS_TOP_K> <filename_of_data> <filename_of_query> <method> <param_of_method>

- <PARAM_DATA_N>: number of rows (instances) of Data (matrix X)
- <PARAM_QUERY_Q>: number of rows (queries) of Queries (matrix Q)
- <PARAM_DATA_D>: number of dimensions
- <filename_of_data>: filename of Data with matrix format N x D (N is number of objects)
- <filename_of_query>: filename of Query with matrix format Q x D (Q is number of queries)

Method (see InputParser.cpp file for more details on the order of parameters):

- Sequential search
	+ "BF": bruteforce search (no parameters)

- Sampling methods with two additional parameters: 
number of samples: PARAM_MIPS_SAMPLES
number of dot product computations: PARAM_MIPS_DOT_PRODUCTS

	+ "wedge": wedge sampling 
	+ "dWedge": heristic wedge (deterministic)
	+ "shift_Wedge": wedge sampling with shifting pre-processing 
	+ "shift_dWedge": heristic wedge with shifting pre-processing 
	+ "pos_Wedge": only consider the positive contribution of <x, q> for wedge sampling
	+ "diamond": diamond sampling
	+ "dDiamond": heristic diamond sampling
	+ "shift_Diamond": diamond sampling with shifting pre-processing

- Greedy methods with 2 additional parameters: 
number of samples: PARAM_MIPS_SAMPLES
number of dot product computations: PARAM_MIPS_DOT_PRODUCTS

	+ "greedy": GreedyMIPS in NIPS 17 (we only use the parameter M since it determines S)

- LSH codes with 2 additional paramenters: 
number of LSH functions: PARAM_LSH_HASH_FUNCTIONS
number of dot product computation: PARAM_MIPS_DOT_PRODUCTS

	+ "simpleLSH_Code": SimpleLSH in ICML 15
	+ RangeLSH": NormRangeLSH in NIPS 18 (one additional parameter: number of partitions: PARAM_LSH_PARTITIONS)

- LSH tables with 3 additional paramenters: 
number of LSH functions: PARAM_LSH_HASH_FUNCTIONS 
number of LSH tables: PARAM_LSH_HASH_TABLES
number of dot product computations: PARAM_MIPS_DOT_PRODUCTS

	+ "simpleLSH_Code": SimpleLSH in ICML 15
	+ RangeLSH": NormRangeLSH in NIPS 18 (one additional parameter: number of partitions: PARAM_LSH_PARTITIONS)

Sample script:
17770 10 150 10 "_Netflix_X_17770_150.txt" "C_Netflix_Q_1000_150.txt" "dWedge" 10000 100
17770 10 150 10 "_Netflix_X_17770_150.txt" "_Netflix_Q_1000_150.txt" "rangeLSH_Table" 64 128 4 100 

There are some internal parameters
- PARAM_TEST_SAVE_OUTPUT = true; // saving results to file
- PARAM_TEST_SAMPLING_RANDOM_GENERATOR = 3; // choosing the random generator for standard wedge & diamond sampling algorithms


