//Help Heder file "funct_Sbox_CPU.h" - CPU computing S-box functions properties
// System includes
#include <stdio.h>
#include <iostream>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////////////
//Declare global vector
int *PTT, *TT, *WHT, *t, *ANF; // SboxDec, *;
//Declaration for global variables use for S-box properties computation
int Lin_cpu, nl_cpu, delta_cpu, AC_cpu, ADmax_cpu, ADmin_cpu, ACn_cpu, Lin_return, AD_return_max, AC_return;
//////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
//Function Header Compute Properties
//////////////////////////////////////////////////////////////////////////////////////////////////
void HeaderCompProperties(int size, int *SboxDec, int bin, int **STT)
{
	int counterSboxVer = 0; //counter for verification of linearity of whole S-box

	int m = bin - 1;

	for (int e = 1; e <= m; e++)
		t[e] = e + 1;

	for (int j = 0; j<size; j++)
	{
		TT[j] = 0;
	}

	int i = 1;

	while (i != m + 1)
	{
		for (int j = 0; j<size; j++)
		{

			TT[j] = TT[j] ^ STT[i][j];

			if (TT[j] == 1)
				PTT[j] = -1;
			else
				PTT[j] = 1;
		}
		t[0] = 1;
		t[i - 1] = t[i];
		t[i] = i + 1;
		i = t[0];

		//Function: Fast Walsh Transformation function CPU
		FastWalshTrans(size, PTT, WHT);	//Find Walsh spectra on one row
		Lin_return = reduceCPU_max(WHT, size);

		if (Lin_cpu < Lin_return)
			Lin_cpu = Lin_return;

		//Function: Fast Mobiush Transformation function CPU
		FastMobiushTrans(size, TT, ANF);
		
		//Function: returne max deg(S)
		AD_return_max = AlgDegMax(size, ANF);
		//cout << "AD max:" << AD_return_max << endl;

		if (ADmax_cpu < AD_return_max)
			ADmax_cpu = AD_return_max;

		//return min deg(S)
		if (ADmin_cpu > AD_return_max)
			ADmin_cpu = AD_return_max;

		//Function: Autocorelation Transformation function CPU
		fun_pow2(size, WHT);
		FastWalshTransInv(size, WHT);
		AC_return = reduceCPU_AC(size, WHT);

		if (ACn_cpu < AC_return)
			ACn_cpu = AC_return;
		//======================================================

		//@Counter
		counterSboxVer++;

		//Compute delta(S)
		int delta = DDT_vector(sizeSbox, SboxElemet, counterSboxVer);

		if (delta > delta_cpu)
			delta_cpu = delta;
		//======================================================
	}
	nl_cpu = sizeSbox / 2 - (Lin_cpu / 2);		//Compute Nonlinearity

	cout << "\nLinearity, Lin(S):" << Lin_cpu << "\n";
	cout << "Nonlinearity, nl(S):" << nl_cpu << "\n";
	cout << "Differential uniformity, delta(S):" << delta_cpu << "\n";
	cout << "Autocorrelation, AC(S): " << ACn_cpu << "\n";
	cout << "Alg. Deagree (max) deg(S):" << ADmax_cpu << "\n";
	cout << "Alg. Deagree (min) deg(S):" << ADmin_cpu << "\n";
	//================================================================
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Main function for compute Sbox properties
//////////////////////////////////////////////////////////////////////////////////////////////////
void MainSboxProperties(int **STT, int *SboxDec)
{
	//Allocate memory blocks
	PTT = (int *)malloc(sizeof(int)* sizeSbox);
	TT = (int *)malloc(sizeof(int)* sizeSbox);
	t = (int *)malloc(sizeof(int)* binary);

	WHT = (int *)malloc(sizeof(int)* sizeSbox);
	ANF = (int *)malloc(sizeof(int)* sizeSbox);

	Lin_cpu = 0, nl_cpu = 0, delta_cpu = 0, ACn_cpu = 0, ADmax_cpu = 0, ADmin_cpu = sizeSbox;

	HeaderCompProperties(sizeSbox, SboxDec, binary, STT);

	//@@Free memory
	free(PTT);
	free(TT);
	free(t);

	free(ANF);
	free(WHT);
}
//////////////////////////////////////////////////////////////////////////////////////////////////
