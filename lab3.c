#include "cblas.h"
#include "c_timer.h"
#include <math.h>
#include <stdio.h>
#include <x86intrin.h>

int main()
{
	int randc; //variable used to store input
	double square; // variable used to get the square root of the user's input
	int root; // variable that stores the square root to be used for paramaters for init_matrix,sse_dgemm,unoptimized_dgemm,cblas_dgemm,and compare_matrix
	printf("How big is your matrix?\n");
	scanf("%d", &randc);
	const double aandb=1.0;
	square=randc;
	root=sqrt(square);
	int squared= randc*randc*sizeof(double*);
	size_t thirty_two=32;
	size_t zero=0;

	double* matrixA[randc*randc];
	double* matrixB[randc*randc];
	double* matrixC[randc*randc];
	double* matrixD[randc*randc];
	double* matrixE[randc*randc];

	int posix_memalign(void** matrixA,size_t thirty_two,size_t squared);
	int posix_mamalign(void** matrixB,size_t thirty_two,size_t squared);
	int posix_mamalign(void** matrixC,size_t thirty_two,size_t squared);
	int posix_mamalign(void** matrixD,size_t thrity_two,size_t squared);
	int posix_mamalign(void** matrixE,size_t thirty_two,size_t squared);

	init_matrix(randc, matrixA);
	init_matrix(randc, matrixB);

	double t0=get_cur_time();
	sse_dgemm(randc, matrixA, matrixB, matrixC);
	double t1=get_cur_time();
	double sse_Execution=t1-t0;

	double t3=get_cur_time();
	unoptimized_dgemm(randc,matrixA,matrixB,matrixD);
	double t4=get_cur_time();
	double uno_Execution=t4-t3;

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,randc,randc,randc,aandb,matrixA,randc,matrixB,randc,aandb,matrixE,randc);
	compare_matrix (randc, matrixC, matrixE);

	double sse_Gflops= 2*pow(randc,3)/ (sse_Execution*pow(10,9));
	double uno_Gflops= 2*pow(randc,3)/ (uno_Execution*pow(10,9));
	printf("The improved dgemm's Gflops is ");
	printf("%lf",sse_Gflops);
	printf("The old dgemm's Gflops is ");
	printf("%lf",uno_Gflops);
	return 0;
}

void compare_matrix(int n, double* A1, double* A2){
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
	int constant=5; 
	for (i=0;i<n;i++){
		for(j=0;j<n;j++){
			*(A + i*n +j)= rand() / (constant * 1.0);
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
