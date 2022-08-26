The C++ source code for MIPS (using CodeBlock) written by Ninh Pham (ninh.pham@auckland.ac.nz)

To use Eigen Lib, Boost
- In Project/Buid Options/Search Directories, add into Compiler

To use FHT
- Only add 5 files: fht.c, fht.h, fht_imple.c, fast_copy.c, fast_copy.h
- DO not add fht_avx.h and fht_sse.h. These files are used when the CPU supports AVX or SSE operations

Parameters: 
<PARAM_DATA_N> <PARAM_QUERY_Q> <PARAM_DATA_D> <PARAM_MIPS_TOP_K> <filename_of_data> <filename_of_query> <method> <param_of_method>

- <PARAM_DATA_N>: number of rows (instances) of Data
- <PARAM_QUERY_Q>: number of rows (queries) of Queries
- <PARAM_DATA_D>: number of dimensions
- <filename_of_data>: filename of Data with matrix format N x D (N is number of objects)
- <filename_of_query>: filename of Query with matrix format Q x D (Q is number of queries)

Method (see InputParser.cpp file for more details):
- "BF": bruteforce search (no parameters)
- "1CEOs" D = 1024 (number of random projections), b = 100 (post-processing number of inner products)
- "2CEOs" D = 1024, b = 100 
- "CEOs_Est" D = 1024 (number of random projections), s0 = 5 (s = 2*s0 number of concomitants for max and min), b = 100
- "CEOs_TA" D = 1024, s0 = 5, b = 100
- "coCEOs" D = 1024, s0 = 20 (s = 2*s0 since we use more concomitants), down_N = 250 (number of points kept in each dimension - reduction over N), b = 100

Sample Scripts:

MIPS.exe 1000000 1000 960 100 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "BF"

MIPS.exe 1000000 1000 960 10 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "1CEOs" 1024 100
MIPS.exe 1000000 1000 960 10 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "2CEOs" 1024 100
MIPS.exe 1000000 1000 960 100 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "CEOs_Est" 1024 5 100
MIPS.exe 1000000 1000 960 100 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "CEOs_TA" 1024 5 100
MIPS.exe 1000000 1000 960 100 "C:\_Data\Dataset\_MIPS\Gist\_X_1000000_960.txt" "C:\_Data\Dataset\_MIPS\Gist\_Q_1000_960.txt" "coCEOs" 1024 20 250 100

