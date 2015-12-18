#define MAXSIZE 250000

#include <iostream>
#include <string>
#include <fstream>	//Writing to files
#include <chrono>	//Keep track of time
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
// to remove intellisense highlighting
#include <device_launch_parameters.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <algorithm>
#include "device_launch_parameters.h"

using namespace std::chrono;

int data[MAXSIZE];
//Main CUDA kernel implementing Sieve of Eratosthenes
__global__ static void CUDASieve(int *num, int range, int bNum, int tNum){
	const int threadId = threadIdx.x;
	const int blockId = blockIdx.x;
	int tmp = blockId*tNum + threadId;
	while (tmp < range){
		int i = 1;
		while (((2 * tmp + 3)*i + tmp + 1) < MAXSIZE){
			num[(2 * tmp + 3)*i + tmp + 1] = 0;
			i++;
		}
		tmp += bNum * tNum;
	}
}
void CUDAFilter(int *number, int size){
	for (int i = 0; i<size; i++)
		number[i] = 2 * i + 1;
	number[0] = 2;
}

void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << ms.count() << " millisecs" << std::endl;
}

void CPUgenPrime(uint64_t range, bool mode, std::ofstream &fileOut) {
	//Start the clock
	steady_clock::time_point ts, te;
	ts = steady_clock::now();
	fileOut << "\nCPU version\n" << "\nCPU version generating from range (0" << "~" << range << ")\n\n";
	//Keep track of results
	uint64_t count = 0;
	//Outer loop
	for (uint64_t i = 0; i < range; i++)
		//Inner loop
		for (uint64_t j = 2; j*j <= i; j++) {
			if (i % j == 0)
				break;
			else if (j + 1 > sqrt(i)) {
				//User wants to see output on screen
				if (mode) {
					std::cout << std::fixed << i << "\t";
					fileOut << std::fixed << i << "\t";
					count++;
				}
				//Just write to file if mode is 0
				else
				{
					fileOut << std::fixed << i << "\t";
					count++;
				}
			}
		}
	//Stop the clock
	te = steady_clock::now();

	std::cout << "\n\nTotal number of primes: " << count << std::endl;
	reportTime("\nCPU Program Completed in ", te - ts);

	fileOut << "\n\nTotal number of primes: " << count << std::endl;

	std::cout << "A log file with the current date/time has been placed in the program directory.\n";
	std::cout << "--------------------------------------------------------------------------------\n";
}

std::ofstream fileInit(){
	//Get current date and time
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);

	//Format in Year-Month-Day_Hour_Minute_Seconds
	strftime(buffer, 80, "%y-%m-%d_%H-%M-%S", timeinfo);
	std::string dateTime(buffer);

	//File handles
	std::ofstream fileOut;
	fileOut.open("GenPrime_out_" + dateTime + ".txt");
	return fileOut;
}

int setupRange(int range) {
	if (range == 0) {
		std::cout << "[2/3] Please choose the range(3 ~ 500,000): \n";
		std::cin >> range;

		//Error checking
		if (range > 2 && range <= 500000) {
			return range;
		}
		else {
			std::cout << "Invalid input for range, value set to default 500,000\n";
			return 500000;
		}
	}
	else return range;
}

//Array of MAXSIZE is created and filled with prime numbers, where [i]
//is the prime int and the rest is padded with 0's
//Example: cpudata[i] = {0,1,0,3,0,5,0,7,0,0,0,11,0,0,0...}
void justDoIt(int range, bool mode, std::ofstream& fileOut) {
	//Output to file
	fileOut << "CUDA Multithreading Sieve of Eratosthenes\n" << "CUDA Multithreading generating from range (0" << "~" << range << ")\n\n";

	//Filter out even numbers to simplify calculation
	CUDAFilter(data, (range / 2) + 1);

	//Initialize arrays
	int *gpudata;
	int cpudata[MAXSIZE];

	//Allocate memory
	cudaMalloc((void**)&gpudata, sizeof(int)*MAXSIZE);

	//Copy to GPU
	cudaMemcpy(gpudata, data, sizeof(int)*MAXSIZE, cudaMemcpyHostToDevice);

	//Maximum threads per block for CUDA 5.2 is 1024
	int bNum = 96, tNum = 1024;
	
	//Start the clock
	steady_clock::time_point ts, te;
	ts = steady_clock::now();

	//Kernel call on the GPU
	CUDASieve << <bNum, tNum, 0 >> >(gpudata, range, bNum, tNum);
	
	//Synchronize the device and the host
	cudaDeviceSynchronize();

	//Copy from GPU back onto host
	cudaMemcpy(&cpudata, gpudata, sizeof(int)*MAXSIZE, cudaMemcpyDeviceToHost);

	//Free the memory on the GPU
	cudaFree(gpudata);

	//Reset the device for easy profiling
	cudaDeviceReset();

	//Stop the clock
	te = steady_clock::now();

	//Display on screen
	if (mode == 1) {
		for (int i = 0; i < MAXSIZE; i++) {
			if (cpudata[i] != 0)
				printf("%d\t", cpudata[i]);
		}
	}
	//Count number of primes
	int count = std::count_if(cpudata, cpudata + MAXSIZE, [](int i){ return i; });
	std::cout << "\n\nTotal number of primes: " << count-2 << std::endl;
	
	//Write to file
	for (int i = 0; i < MAXSIZE; i++) {
		if (cpudata[i] != 0) {
			fileOut << cpudata[i] << "\t";
		}
	}
	//Show the amount of time 
	reportTime("GPU Program Completed in ", te - ts);
	fileOut << "\n\nTotal number of primes: " << count - 2 << std::endl;
	std::cout << "A log file with the current date/time has been placed in the program directory.\n";
	std::cout << "--------------------------------------------------------------------------------\n";
}

void menu(int range, bool mode, std::ofstream& fileOut){
	std::cout << "[3/3] Please select the version of the program you want to run\n"
		<< "1. [*****]  CUDA Multithreading Sieve of Eratosthenes version\n"
		<< "2. [***]    Simple CPU version\n"
		<< "3. [**]	Run both versions\n"
		<< "0. Quit\n"
		<< "Option: ";
	int mainMenuOption;
	std::cin >> mainMenuOption;	//Accept user input
		switch (mainMenuOption) {
		case 0:	// User wants to exit
			std::cout << "Thank you for testing our program :)\n"
				<< "Fork us @ https://github.com/bbershadsky/" << std::endl;
			break;
		case 1:
			std::cout << "CUDA Multithreading generating from range (0" << "~" << range << ")\n";
			std::cout << "--------------------------------------------------------------------------------\n";
			justDoIt(range, mode, fileOut);

			//Close the file handle
			fileOut.close();
			break;
		case 2:
			std::cout << "CPU version generating from range (0" << "~" << range << ")\n";
			std::cout << "--------------------------------------------------------------------------------\n";
			CPUgenPrime(range, mode, fileOut);

			//Close the file handle
			fileOut.close();
			break;
		case 3:
			std::cout << "Running all available options\n";
			justDoIt(range, mode, fileOut);
			CPUgenPrime(range, mode, fileOut);

			//Close the file handle
			fileOut.close();
			break;
		default:
			std::cout << "[Invalid option. Only integers 0-3 are allowed]\n";
			menu(range, mode, fileOut);
			break;
		}
}

void setupScreenMode(int range) {
	std::cout << "***Team /dev/null GPU610 PRIME NUMBER GENERATOR v3.5***\n"
		<< "[1/3] Would you like to see the output on screen?\n"
		<< "0 = NO, write to file only\n"
		<< "1 = YES, display on screen\n"
		<< "Show on screen?: ";
		int mode = 1;
		std::cin >> mode;

		//Initialize file handle
		std::ofstream fileOut = fileInit();

		if (mode == 0) {
			std::cout << "***Writing output to file only***\n\n";
			range = setupRange(range);
			menu(range, mode, fileOut);
		}

		else if (mode == 1) {
			std::cout << "***Outputting results on screen***\n\n";
			range = setupRange(range);
			menu(range, mode, fileOut);
		}
		else {
			std::cout << "[Invalid option selected, default option 0 (output to screen) selected]\n\n";
			range = setupRange(range);
			menu(range, 1, fileOut);
		}
}

//Initialize value to be used in the program using command line arguments
int initRuntimeValue(int argc, char* argv[]){
	//Save runtime parameter into local variable, if provided
	int range = 500000;
	if (argc == 1) {
		std::cout << "[No command line parameters provided]\n\n";
		return 0;
	}
	if (argc == 2)
		range = std::atoi(argv[1]);
	if (range > 2 && range < 500000)
		return range;
	else {
		std::cout << "[Bad input for range parameter (must be <= 500,000)]\n"
			<< "Range has been set to 500,000\n";
		return range = 500000;
	}
}

int main(int argc, char* argv[]) {
	//Grab the command line arguments
	int range = initRuntimeValue(argc, argv);

	//Prompt user for mode (verbose or silent)
	setupScreenMode(range);
	std::cout << "Thank you for testing our program :)\n"
		<< "Fork us @ https://github.com/bbershadsky/" << std::endl;
	return 0;
}

/*
CHANGELOG
v1.0 - Generating from simple double loop
v1.0.1 - Command line parameter input
v1.1 - Nicer output format and error feedback
v1.2 - Full 64 bit integer compatibility
v1.3 - Multithreading and CUDA implemented
v2.0 - Completely rewrote program to include menu and multiple run parameters
v3.0 - Full rewrite of CUDAGenPrime to use CUDASieve of Eratosthenes, and initRuntimeValues
v3.1 - Moved new CUDAGenPrime to separate function justDoIt(range);
v3.2 - Reorganized main() into simpler blocks for easier readability and efficiency
v3.3 - Moved most control blocks over to the menu() for easier modification
v3.3.1 - Removed a bunch of unused includes
v3.4 - Successfully fixed file output and implemented count
v3.5 - Final version with usability and performance upgrades
*/