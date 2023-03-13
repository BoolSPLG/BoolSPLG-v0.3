//Heder file "boolsplg_host_sbox_funct.h" - CPU computing S-box functions properties and other functions
//input header files
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <nmmintrin.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
//Function for binary convert
//////////////////////////////////////////////////////////////////////////////////////////////////
void binary1(int number, int* binary_num, int binary) {
	int w = number, c, k, i = 0;
	for (c = binary - 1; c >= 0; c--)
	{
		k = w >> c;

		if (k & 1)
		{
			binary_num[i] = 1;
			i++;
		}
		else
		{
			binary_num[i] = 0;
			i++;
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Set STT form for S-box
//////////////////////////////////////////////////////////////////////////////////////////////////
void SetSTT(int* sbox, int** STT, int* binary_num, int sizeSbox, int binary)
{
	int elementS = 0;
	for (int j = 0; j < sizeSbox; j++)
	{
		elementS = sbox[j];
		binary1(elementS, binary_num, binary);

		for (int i = 0; i < binary; i++)
		{
			STT[i][j] = binary_num[i];
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Set S-box ANF form for S-box 
//////////////////////////////////////////////////////////////////////////////////////////////////
void SetS_ANF(int* sbox, int* SboxElemet_ANF, int sizeSbox, int binary)
{
	int* TT_ANF = (int*)malloc(sizeof(int) * sizeSbox);
	int* ANF_TT = (int*)malloc(sizeof(int) * sizeSbox);
	int* binary_num = (int*)malloc(sizeof(int) * binary);

	int** SANF = AllocateDynamicArray<int>(sizeSbox, binary);
	int** STT = AllocateDynamicArray<int>(binary, sizeSbox);

	SetSTT(sbox, STT, binary_num, sizeSbox, binary);

	//Generate S_ANF
	for (int j = 1; j < binary; j++)
	{
		for (int i = 0; i < sizeSbox; i++)
		{
			TT_ANF[i] = STT[j][i];
		}
		FastMobiushTrans(sizeSbox, TT_ANF, ANF_TT);
		for (int k = 0; k < sizeSbox; k++)
		{
			SANF[k][j] = ANF_TT[k];
		}
	}

	for (int i = 0; i < sizeSbox; i++)
	{
		int pos = binary - 2, dec = 0;
		for (int m = 1; m < binary; m++)
		{
			dec += SANF[i][m] * pow(2, pos);
			pos--;
		}

		SboxElemet_ANF[i] = dec;
	}

	//@@free memory
	FreeDynamicArray<int>(SANF);
	FreeDynamicArray<int>(STT);
	free(binary_num);
	free(TT_ANF);
	free(ANF_TT);
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//CPU computing DDT(S) function
//////////////////////////////////////////////////////////////////////////////////////////////////
int DDT_vector(int sizeSbox, int* sbox, int dx)
{
	int delta=0;
	int* DDT = new int[sizeSbox]();

	int x1, x2, dy;
	for (x1 = 0; x1 < sizeSbox; ++x1) {
		//  for (dx = 0; dx < sbox_size; ++dx) {
		x2 = x1 ^ dx;
		dy = sbox[x1] ^ sbox[x2];
		++DDT[dy];
		// }
		if (DDT[dy] > delta)
			delta = DDT[dy];
	}
	//  for (int i = 0; i < sizeSbox; ++i) {
	//        std::cout << std::setw(4) << diff_table[i];
	//    }
	delete[] DDT;
	return delta;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//CPU computing max deg(S) function
//////////////////////////////////////////////////////////////////////////////////////////////////
int AlgDegMax(int sizeSbox, int* ANF_CPU)
{
	unsigned int ones = 0, max = 0;
	for (int i = 0; i < sizeSbox; i++)
	{
		// code specific to Visual Studio compiler
#if defined (_MSC_VER)
		ones = _mm_popcnt_u32(i) * ANF_CPU[i];
#endif
		// code specific to gcc compiler
#if defined (__GNUC__)
		ones = __builtin_popcount(i) * ANF_CPU[i];
#endif
		if (max < ones)
			max = ones;
		//if((min>ones)&(ones!=0))
		//	min=ones;

		//	cout <<ones<< " ";
	}
	//	cout <<"Alg. Deagree (max):" << max <<" Alg. Deagree (min):" << min <<"\n";
	return max;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//CPU computing min deg(S) function
//////////////////////////////////////////////////////////////////////////////////////////////////
int AlgDegMin(int sizeSbox, int* ANF_CPU)
{
	unsigned int ones = 0, min = 100;
	for (int i = 0; i < sizeSbox; i++)
	{
		// code specific to Visual Studio compiler
#if defined (_MSC_VER)
		ones = _mm_popcnt_u32(i) * ANF_CPU[i];
#endif
		// code specific to gcc compiler
#if defined (__GNUC__)
		ones = __builtin_popcount(i) * ANF_CPU[i];
#endif
		//if(max<ones)
		//	max=ones;
		if ((min > ones) & (ones != 0))
			min = ones;

		//	cout <<ones<< " ";
	}
	//cout << min << " ";
	return min;
	//	cout <<"Alg. Deagree (max):" << max <<" Alg. Deagree (min):" << min <<"\n";
}
//////////////////////////////////////////////////////////////////////////////////////////////////