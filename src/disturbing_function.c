// Author: Sam Hadden
//
#define LAPLACE_EPS 1.0e-10

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define TWOPI 6.283185307179586476925287

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define STRINGIFY(s) str(s)
#define str(s) #s

const char* celmech_build_str = __DATE__ " " __TIME__; // Date and time build string. 
const char* celmech_version_str = "1.0.1";         // **VERSIONLINE** This line gets updated automatically. Do not edit manually.
const char* celmech_githash_str = STRINGIFY(CELMECHGITHASH);             // This line gets updated automatically. Do not edit manually.

double laplace(double s, int i, int j, double a);
double GeneralOrderCoefficient(int res_j, int order, int epower,double a);

int binomialCoeff(int n, int k)
{
  // Base Cases
  if (k==0 || k==n)
    return 1;
 
  // Recur
  return  binomialCoeff(n-1, k-1) + binomialCoeff(n-1, k);
}
int factorial(int f)
{
 	assert(f>=0);
    if ( f == 0 ) 
        return 1;
    return(f * factorial(f - 1));
}

double secularF2(double alpha){
	assert(alpha<1);
	return 0.25 * laplace(0.5,0,1,alpha) + 0.125 * laplace(0.5,0,2,alpha);
};
double secularF10(double alpha){
		assert(alpha<1);
		return 0.5 * laplace(0.5,1,0,alpha) - 0.5 * laplace(0.5,1,1,alpha) - 0.25 * laplace(0.5,1,2,alpha);
};

// Page 247 of M&D '99

// Leading order component of M & D Equation 6.38

double NCOd0(int a, int b, int c){
	if(c == 0) return 1.0;
	if(c==1) return b - 0.5 * a;
	// Else
	double cc = (double) c;
	double nc1,nc2;
	nc1 = NCOd0(a, b+1,c-1);
	nc2 = NCOd0(a, b+2,c-2);
	return 0.25 * (2 * (2*b - a) * nc1 + (b - a) * nc2 ) / cc;
}
double NewcombOperator(int a, int b, int c,int d){
	assert(c==0 || d==0);
	if (d==0){
		return NCOd0(a,b,c);
	}
	else{
		return NCOd0(a,-b,d);
	}
}
double HansenCoefficient(int a, int b, int c){
	int alpha =  c-b > 0 ? c-b:0;
	int beta =  b-c > 0 ? b-c:0;
	return NewcombOperator(a,b,alpha,beta);
}

double GeneralOrderCoefficient(int res_j, int order, int epower,double a){

	int j[7];
	j[0]=0; // ignore to match M&D indexing
	j[1] = res_j;
	j[2] = order - res_j ;
	j[3] = epower - order;
	j[4] = -1 * epower;
	j[5] = 0;
	j[6] = 0; 
	int q = j[4];
	int q1 = -1 * j[3];
	int Nmax = order;
	
	double coeff =0;
	for (int l=0; l<= Nmax; l++){
	int sgn = l%2 ? -1:1;
	double fact = (double) factorial(l);
	double sum = 0;
	for (int k=0; k<= l; k++){
		double ncIn;
		double ncOut;
		int sgn2 = k%2 ? -1:1;
		int binom = sgn2 * binomialCoeff(l, k);
		int jj = j[2] + q;
		if (jj<0) jj=-1*jj;
		ncIn = HansenCoefficient(k,-j[2]-j[4],-j[2]);
		ncOut = HansenCoefficient(-1-k,j[1]+j[3],j[1]);
		sum += binom * ncIn*ncOut* laplace(0.5,jj,l,a);
	}
	coeff += sgn * sum / fact;
	}
	return coeff;

}



/* Code due to Jack Wisdom */
/* compute Laplace coefficients and Leverrier derivatives
          j
     j   d     i
    a   ---   b (a)
          j    s
        da
   by series summation */



double laplace(double s, int i, int j, double a)
{
  double as, term, sum, factor1, factor2, factor3, factor4;
  int k,q, q0;

  as = a*a;

  if(i<0) i = -i;

  if(j<=i)     /* compute first term in sum */
    {
      factor4 = 1.0;
      for(k=0; k<j; k++)
        factor4 *= (i - k);
      sum = factor4;
      q0=0;
    }
  else
    {
       q0 = (j + 1 - i) / 2;
      sum = 0.0;
      factor4 = 1.0;
    }

  /* compute factors for terms in sum */

  factor1 = s;
  factor2 = s + i;
  factor3 = i + 1.0;
  for(q=1;q<q0;q++)   /* no contribution for q = 0 */
    {
      factor1 *= s + q;
      factor2 *= s + i + q;
      factor3 *= i + 1.0 + q;
    }

  term = as * factor1 * factor2 / (factor3 * q);

  /* sum series */
  int ctr = 0;
  while(term*factor4 > LAPLACE_EPS)
    {
      factor4 = 1.0;
      for(k=0;k<j;k++)
        factor4 *= (2*q + i - k);
      sum += term * factor4;
      factor1 += 1.0;
      factor2 += 1.0;
      factor3 += 1.0;
      q++;
      term *= as * factor1 * factor2 / (factor3 * q);
      ctr += 1;
      if (ctr > 100000){ // failsafe in case alpha too close to 1
          return NAN;
      }
    }

  /* fix coefficient */

  for(k=0;k<i;k++)
    sum *= (s + ((double) k))/(((double) k)+1.0);

  if(q0 <= 0)
    sum *= 2.0 * pow(a, ((double) i));
  else
    sum *= 2.0 * pow(a, ((double) 2*q0 + i - 2));

  return(sum);
}
