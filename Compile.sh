#!/bin/bash

g++ -O3 -m64 -std=c++17 -fopenmp -march=native -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/BF.cpp -o obj/Release/BF.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/Concomitant.cpp -o obj/Release/Concomitant.o
gcc -O3 -m64 -fopenmp -march=native  -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/fast_copy.c -o obj/Release/fast_copy.o
gcc -O3 -m64 -fopenmp -march=native  -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/fht.c -o obj/Release/fht.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native -ffast-math  -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/Header.cpp -o obj/Release/Header.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/InputParser.cpp -o obj/Release/InputParser.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native -ffast-math  -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/main.cpp -o obj/Release/main.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native  -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/Test.cpp -o obj/Release/Test.o
g++ -O3 -m64 -std=c++17 -fopenmp -march=native  -ffast-math -I/home/npha145/Library/eigen-3.4.0 -IC:/Library/eigen-3.4.0 -I/home/npha145/Library/boost_1_78_0 -IC:/Library/boost_1_77_0 -I/home/npha145/Library/eigen-3.4.0 -c /home/npha145/Dropbox/Working/_Code/C++/CEOs/Utilities.cpp -o obj/Release/Utilities.o
g++  -o bin/Release/MIPS obj/Release/BF.o obj/Release/Concomitant.o obj/Release/fast_copy.o obj/Release/fht.o obj/Release/Header.o obj/Release/InputParser.o obj/Release/main.o obj/Release/Test.o obj/Release/Utilities.o  -O3 -s -m64 -lgomp -pthread -ffast-math
 
cp /home/npha145/Dropbox/Working/_Code/C++/CEOs/bin/Release/MIPS /home/npha145/Dropbox/Working/_Code/_Experiment/CEOs/Netflix/
cp /home/npha145/Dropbox/Working/_Code/C++/CEOs/bin/Release/MIPS /home/npha145/Dropbox/Working/_Code/_Experiment/CEOs/Gist/
cp /home/npha145/Dropbox/Working/_Code/C++/CEOs/bin/Release/MIPS /home/npha145/Dropbox/Working/_Code/_Experiment/CEOs/Yahoo/
#cp /home/npha145/Dropbox/Working/_Code/C++/FalconnCEOs/bin/Release/FalconnCEOs /home/npha145/Dropbox/Working/_Code/_Experiment/Falconn/word2vec_center/

