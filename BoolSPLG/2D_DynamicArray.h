//Heder file 2D DynamicArray
//input header file
#include <iostream>

//===== function for Allocate 2D Dynamic Array ==============
template <typename T>
T **AllocateDynamicArray(int nRows, int nCols)
{
	T **dynamicArray;

	dynamicArray = new T*[nRows];
	for (int i = 0; i < nRows; i++)
		dynamicArray[i] = new T[nCols];

	return dynamicArray;
}
//=========================================================

//===== function for Delete 2D Dynamic Array ==============
template <typename T>
void FreeDynamicArray(T** dArray)
{
	delete[] * dArray;
	delete[] dArray;
}
//=========================================================
