////////////////////////////////////////////////////////////////////////////
//
// Copyright @2017-2021 Dusan and Iliya.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <iostream>

// CUDA runtime.
#include "cuda_runtime.h"

//Main Library header file
#include <BoolSPLG/BoolSPLG_v03.cuh>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
//Utilities main Function - Example using of GPU - CUDA function to show Capable device(s) characteristics
/////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "=============================================================================";
	printf("\nExample, functions which show GPU - CUDA Capable device(s) characteristics.\n");

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//BoolSPLG Properties Library functions
	/////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "=============================================================================\n";
	//Function to check if GPU fulfill BoolSPLG CUDA-capable requires
	BoolSPLGMinimalRequires();

	cout << "\n=============================================================================\n";
	//Function Detected and show CUDA Capable device(s) characteristics
	BoolSPLGCheckProperties();

	cout << "\n=============================================================================\n";
	//Function Detected and show CUDA Capable device(s) characteristics (full, extend informacion)
	BoolSPLGCheckProperties_v1();

	cout << "\n=============================================================================\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//BoolSPLG Properties Library, Simple test function to measure the Memcopy bandwidth of the GPU
	/////////////////////////////////////////////////////////////////////////////////////////////////
	printf("\n   --- Test function to measure the Memcopy Bandwidth of the Nvidia device ---\n\n");
	int size = 1;
	cout << "Input memory size for transfer (MB):";
	cin >> size;
	unsigned int nElements = size * 256 * 1024;
	const unsigned int bytes = nElements * sizeof(int);

	bandwidthTest(bytes, nElements);
	cout << "=============================================================================\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////
	printf("\n   --- End Example, BoolSPLG Library, functions show CUDA Capable device(s) characteristics. ---\n\n");
	/////////////////////////////////////////////////////////////////////////////////////////////////

	return 0;
}
