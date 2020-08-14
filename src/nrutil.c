#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#define NR_END 1
#define FREE_ARG char*

//#include <malloc.h>
#define ALLOCATE malloc
#define FREE free

void nrerror(char error_text[])
/* error handler */
{
	fprintf(stderr,"run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

int *ivector(long nl, long nh)
/* allocate an int vector */
{
	int *v;

	v=(int *)ALLOCATE((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl+NR_END;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector */
{
	unsigned long *v;

	v=(unsigned long *)ALLOCATE((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
	if (!v) nrerror("allocation failure in lvector()");
	return v-nl+NR_END;
}

float *vector(long nl, long nh)
/* allocate a float vector */
{
	float *v;

	v=(float *)ALLOCATE((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl+NR_END;
}


double *dvector(long nl, long nh)
/* allocate a double vector */
{
	double *v;

	v=(double *)ALLOCATE((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl+NR_END;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript rangem[nrl..nrh][mcl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

/* allocate pointers to rows */
	m=(double **) ALLOCATE((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in dmatrix()");
	m += NR_END;
	m -= nrl;

/* allocate rows and set pointers to them */
	m[nrl]=(double *) ALLOCATE((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in dmatrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

/* return pointer to array of pointers to rows */
	return m;
}

void free_ivector(int *v, long nl, long nh)
{
	nh=nh;
	FREE((FREE_ARG) (v+nl-NR_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
{
	nh=nh;
	FREE((FREE_ARG) (v+nl-NR_END));
}

void free_vector(float *v, long nl, long nh)
{
	nh=nh;
	FREE((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
{
	nh=nh;
	FREE((FREE_ARG) (v+nl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
{
	nrh=nrh;
	nch=nch;
	FREE((FREE_ARG) (m[nrl]+ncl-NR_END));
	FREE((FREE_ARG) (m+nrl-NR_END));
}




