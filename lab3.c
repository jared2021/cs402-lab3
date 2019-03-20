#include "cblas.h"
#include "c_timer.h"
#include <stdio.h>
#include <x86intrin.h>

int main()
{
	int randc;
	printf("How many rows and columns is your matrix?\n");
	scanf("%d", &randc);
	const double aandb=1.0;
	double* matrixA [randc*randc];
	double* matrixB [randc*randc];
	double* matrixC [randc*randc];
	double* matrixD [randc*randc];
	init_matrix(randc, matrixA);
	init_matrix(randc, matrixB);
	sse_dgemm(randc, matrixA, matrixB, matrixC);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,randc,randc,randc,aandb,matrixA,randc,matrixB,randc,aandb,matrixD,randc);
	compare_matrix (randc, matrixC, matrixD);
	return 0;
}

void compaare_matrix(int n, double* A1, double* A2){
	int i, j;
	double d1, d2;
	for (i=0;i<n;i++){
		for(j=0;j<n;j++){
			d1= *(A1 + i*n + j);
			d2= *(A2 + i*n + j);
			if(fabs(d2-d1)/(d2) > 1e-6){
				printf("ERRORL %f(my)<>%f(dgemm)\n", d1, d2);
				exit(1);
			}
		}
	}
	printf("Correct result! :-) \n");
}

void init_matrix(int n, double* A){
	int i,j;
	int RAND_MAX=5;
	for (i=0;i<n;i++){
		for(j=0;j<n;j++){
			*(A + i*n +j)= rand() / (RAND_MAX * 1.0);
		}
	}
}

void unoptimized_dgemm(int n, double* A, double* B, double* C)
{
	int i,j,k;
	for (i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			double cij= C[i+j*n]; /* cij= C[i][j] */
			for(k=0;k<n;k++)
			{
				cij += A[i+k*n] * B[k+j*n]; /* cij =+ A[i][k] * B[k][j] */
			}
		}
	}
}

void sse_dgemm (int n, double* A ,double* B, double* C)
{
	int i,j,k;
	for (i=0;i<n;i+=4)
		for (j=0;j<n;j++){
			__m256d c0= _mm256_load_pd(C+i+j*n); /* c0= C[i][j] */
			for(k=0;k<n;k++)
				c0=_mm256_add_pd(c0, /*c0 +=A[i][j] */
				     _mm256_mul_pd(_mm256_load_pd(A+i+k*n),
				     _mm256_broadcast_sd(B+k+j*n)));
			_mm256_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
		}
}
