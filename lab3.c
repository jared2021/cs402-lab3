#include "cblas.h"
#include "c_timer.h"
#include <math.h>
#include <stdio.h>
#include <x86intrin.h>

int main()
{
	int randc;
	printf("How big is your matrix?\n");
	scanf("%d", &randc);
	const double aandb=1.0;
	size_t squared= randc*randc*sizeof(double); //change back to int if having problems?
	size_t thirty_two=32;
	size_t zero=0;
	printf("Initialized non-matrix variables.\n");

	double* matrixA[randc*randc];
	int posix_mamalign(void** matrixA,size_t thirty_two,size_t squared);
	printf("Initialzed matrixA.\n");
	double* matrixB[randc*randc];
	int posix_mamalign(void** matrixB,size_t thirty_two,size_t squared);
	printf("initialized matrixB.\n");
	double* matrixC[randc*randc];
	int posix_mamalign(void** matrixC,size_t thirty_two,size_t squared);
	printf("initilized matrixC.\n");
	//double* matrixD[randc*randc];
	//int posix_mamalign(void** matrixD,size_t thirty_two,size_t squared);
	//printf("initiialized matrixD.\n");
	//double* matrixE[randc*randc];
	//printf("Created variable matrixE.\n");
	int posix_mamalign(void** matrixE,size_t thirty_two,size_t squared);
	printf("Initialized all variables.\n");

	//int posix_memalign(void** matrixA,size_t thirty_two,size_t squared);
	//int posix_mamalign(void** matrixB,size_t thirty_two,size_t squared);
	//int posix_mamalign(void** matrixC,size_t thirty_two,size_t squared);
	//int posix_mamalign(void** matrixD,size_t thrity_two,size_t squared);
	//int posix_mamalign(void** matrixE,size_t thirty_two,size_t squared);

	init_matrix(randc, matrixA);
	init_matrix(randc, matrixB);

	printf("Would you like to run the 0)original dgemm or the 1) updated dgemm?\n");
	int option;
	scanf("%d",option);
	if(option==1)
	{
		double t0=get_cur_time();
		sse_dgemm(randc, matrixA, matrixB, matrixC);
		double t1=get_cur_time();
		double sse_Execution=t1-t0;
		double sse_Gflops= 2*pow(randc,3)/(sse_Execution*pow(10,9));
		printf("The improved dgemm's Gflops is ");
		printf("%lf",sse_Gflops);
	}
	if(option==0)
	{
		double t3=get_cur_time();
		unoptimized_dgemm(randc,matrixA,matrixB,matrixC);
		double t4=get_cur_time();
		double uno_Execution=t4-t3;
		double uno_Gflops=2*pow(randc,3)/(uno_Execution*pow(10,9));
		printf("The old dgemm's Gflops is ");
		printf("%lf",uno_Gflops);
	}
	if(option==1)
	{
		double* matrixD[randc*randc];
		int posix_mamalign(void** matrixD,size_t thirty_two,size_t squared);
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,randc,randc,randc,aandb,matrixA,randc,matrixB,randc,aandb,matrixD,randc);
		compare_matrix (randc, matrixC, matrixD);
	}
	//double uno_Gflops= 2*pow(randc,3)/ (uno_Execution*pow(10,9));
	//printf("The improved dgemm's Gflops is ");
	//printf("%lf",sse_Gflops);
	//printf("The old dgemm's Gflops is ");
	//printf("%lf",uno_Gflops);
	
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
