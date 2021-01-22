#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define ZINDEX(n) (((n) + 2) % 4)
#define ABS(n) (((n) >= 0 ) ? (n) : (-1 * n))
#define INDEX(j,k) ( 8 * (j) + (k))
typedef struct SeriesTerm {
	int k[6];
	int z[4];
	double coeff;
	struct SeriesTerm* next;
} SeriesTerm;

SeriesTerm* add_term(SeriesTerm* term,int k[6], int z[4]){
	SeriesTerm* new_term = malloc(sizeof(SeriesTerm));
	term->next = new_term;
	for(int i=0;i<6;i++) new_term->k[i] = k[i];
	for(int i=0;i<4;i++) new_term->z[i] = z[i];
	return new_term->next;
}

double complex** get_complex_variables_arr(const int nrow, const int ncol){
	double complex* values = calloc(nrow * ncol, sizeof(double complex));
	double complex** rows = malloc( nrow * sizeof(double complex*));
	for (int i=0;i<nrow;i++){
	 rows[i] = values + i*ncol;
	}
	return rows;
}
void free_complex_variables_arr(complex double** arr){
	free(*arr);
	free(arr);
}
complex double mypow(complex double* x,const int n){
	return ((n) >= 0) ? (x[n]) : conj((x[-n]));
}
void get_powers_array(complex double* var, complex double** var_pow ,const int n, const int Nmax){
	for(int i=0;i<n;i++){
	  var_pow[i][0] = 1.;
	  for(int p=1;p<=Nmax;p++){
	    var_pow[i][p] = var_pow[i][p-1] * var[i] ;
	  }
	}
}

complex double evaluate_term(SeriesTerm* term, complex double** exp_Il_arr, double complex** xy_arr){
	const int* k = term->k;
	int z;
	complex double product = term->coeff;
	product *=  mypow(exp_Il_arr[1],k[0]);
	product *=  mypow(exp_Il_arr[0],k[1]);
	for(int i=0; i<4;i++){
		product *= mypow(xy_arr[i],k[i+2]); 
		z = term->z[ZINDEX(i)];
		product *= xy_arr[i][z] * conj(xy_arr[i][z]);
	}
	return product;
}
complex double evaluate_term_and_derivs
(SeriesTerm* term, complex double** exp_Il_arr, double complex** xy_arr,
 complex double* deriv_wrt_xy,complex double* deriv_wrt_xybar){
	const int* k = term->k;
	int z,ki,xy_pow,xybar_pow;
	complex double tmp;
	complex double product = term->coeff;
	product *=  mypow(exp_Il_arr[1],k[0]);
	product *=  mypow(exp_Il_arr[0],k[1]);
	for (int i=0; i<4;i++){
		deriv_wrt_xy[i] = product;
		deriv_wrt_xybar[i] = product;
	}
	for(int i=0; i<4;i++){
		z = term->z[ZINDEX(i)];
		ki = k[i+2];
		xy_pow = z + MAX(0,ki);
		xybar_pow = z - MIN(0,ki);
		tmp = xy_arr[i][xy_pow] * conj(xy_arr[i][xybar_pow]);
		for(int j=0;j<4;j++){
			if (j==i){
				deriv_wrt_xy[j] *= xy_pow * xy_arr[j][MAX(0,xy_pow-1)] * conj(xy_arr[j][xybar_pow]);
				deriv_wrt_xybar[j] *= xybar_pow * xy_arr[j][xy_pow] * conj(xy_arr[j][MAX(0,xybar_pow-1)]);
			}else{
				deriv_wrt_xy[j] *= tmp;
				deriv_wrt_xybar[j] *= tmp;
			}
		}
		product *= tmp;
	}
	return product;
}
complex double evaluate_term_and_jacobian
(SeriesTerm* term, complex double** exp_Il_arr, double complex** xy_arr,
 complex double* deriv_wrt_xy,complex double* deriv_wrt_xybar,complex double* jacobian){
	const int* k = term->k;
	int z,ki,xy_pow,xybar_pow;
	complex double tmp,D_tmp,DD_tmp,Dbar_tmp,DbarDbar_tmp,DDbar_tmp;
	complex double product = term->coeff;
	product *=  mypow(exp_Il_arr[1],k[0]);
	product *=  mypow(exp_Il_arr[0],k[1]);
	for (int i=0; i<4;i++){
		deriv_wrt_xy[i] = product;
		deriv_wrt_xybar[i] = product;
	}
	for (int i=0;i<64;i++) jacobian[i]=product;
	bool j_eq_i,k_eq_i;
	for(int i=0; i<4;i++){
		z = term->z[ZINDEX(i)];
		ki = k[i+2];
		xy_pow = z + MAX(0,ki);
		xybar_pow = z - MIN(0,ki);

		tmp = xy_arr[i][xy_pow] * conj(xy_arr[i][xybar_pow]);
		D_tmp = xy_pow * xy_arr[i][MAX(0,xy_pow-1)] * conj(xy_arr[i][xybar_pow]);
		Dbar_tmp = xybar_pow * xy_arr[i][xy_pow] * conj(xy_arr[i][MAX(0,xybar_pow-1)]);

		DD_tmp = xy_pow * (xy_pow - 1) * xy_arr[i][MAX(0,xy_pow - 2)] * conj(xy_arr[i][xybar_pow]);
		DbarDbar_tmp = xybar_pow * (xybar_pow - 1) * xy_arr[i][xy_pow] * conj(xy_arr[i][MAX(0,xybar_pow-2)]);
		DDbar_tmp = xybar_pow * (xy_pow) * xy_arr[i][MAX(0,xy_pow - 1)] * conj(xy_arr[i][MAX(0,xybar_pow-1)]);
		
		product *= tmp;
		
		for(int j=0;j<4;j++){
			j_eq_i = j==i;
			deriv_wrt_xy[j] *= ( j_eq_i ? D_tmp:tmp);
			deriv_wrt_xybar[j] *= (j_eq_i ? Dbar_tmp:tmp);
			jacobian[INDEX(j,j)] *= (j_eq_i ? DD_tmp:tmp);
			jacobian[INDEX(j+4,j+4)] *= ( j_eq_i ? DbarDbar_tmp:tmp) ;
			jacobian[INDEX(j+4,j)] *= ( j_eq_i ? DDbar_tmp:tmp) ;
			for(int k=0;k<j;k++){
				if(j_eq_i){
					jacobian[INDEX(j,k)] *= D_tmp;
					jacobian[INDEX(j,k+4)] *= D_tmp;
					jacobian[INDEX(j+4,k+4)] *= Dbar_tmp;
					jacobian[INDEX(j+4,k)] *= Dbar_tmp;
				}else{
					k_eq_i = k==i;
					jacobian[INDEX(j,k)] *= (k_eq_i ? D_tmp:tmp);
					jacobian[INDEX(j+4,k)] *= (k_eq_i ? D_tmp:tmp);
					jacobian[INDEX(j,k+4)] *= (k_eq_i ? Dbar_tmp:tmp);
					jacobian[INDEX(j+4,k+4)] *= (k_eq_i ? Dbar_tmp:tmp);
				}
			}
		}
	}
	return product;
}

void evaluate_series
(double complex exp_Il[2],double complex xy[4], SeriesTerm* first_term,const int kmax, const int Nmax,
 double* re, double* im){
	complex double** xy_arr = get_complex_variables_arr(4,Nmax + 1);
	complex double** exp_Il_arr = get_complex_variables_arr(2,kmax + 1);
	get_powers_array(xy,xy_arr,4,Nmax);
	get_powers_array(exp_Il,exp_Il_arr,2,kmax);
	SeriesTerm* term = first_term;
	complex double sum = 0;
	do{
		sum += evaluate_term(term,exp_Il_arr,xy_arr);
		term = term->next;
	} while(term != 0);

	free_complex_variables_arr(xy_arr);
	free_complex_variables_arr(exp_Il_arr);
	*re = creal(sum);
	*im = cimag(sum);
}

void evaluate_series_and_derivs
(double complex exp_Il[2],double complex xy[4], SeriesTerm* first_term,const int kmax, const int Nmax,
 double* re, double* im, double re_deriv_xy[4], double im_deriv_xy[4] ,double re_deriv_xybar[4], double im_deriv_xybar[4]){
	complex double** xy_arr = get_complex_variables_arr(4,Nmax + 1);
	complex double** exp_Il_arr = get_complex_variables_arr(2,kmax + 1);
	get_powers_array(xy,xy_arr,4,Nmax);
	get_powers_array(exp_Il,exp_Il_arr,2,kmax);
	SeriesTerm* term = first_term;
	complex double sum = 0;
	complex double dxy_tot[4] = {0,0,0,0};
	complex double dxybar_tot[4] = {0,0,0,0};
	complex double dxy[4], dxybar[4];
	do{
		sum += evaluate_term_and_derivs(term,exp_Il_arr,xy_arr,dxy,dxybar);
		for(int i=0; i<4;i++){
			dxy_tot[i]+=dxy[i];
			dxybar_tot[i]+=dxybar[i];
		}
		term = term->next;
	} while(term != 0);

	for(int i=0; i<4;i++){
		re_deriv_xy[i] = creal(dxy_tot[i]);
		re_deriv_xybar[i] = creal(dxybar_tot[i]);
		im_deriv_xy[i] = cimag(dxy_tot[i]);
		im_deriv_xybar[i] = cimag(dxybar_tot[i]);
	}

	free_complex_variables_arr(xy_arr);
	free_complex_variables_arr(exp_Il_arr);
	*re = creal(sum);
	*im = cimag(sum);
}
void evaluate_series_and_jacobian
(double complex exp_Il[2],double complex xy[4], SeriesTerm* first_term,const int kmax, const int Nmax,
 double* re, double* im, double re_deriv_xy[4], double im_deriv_xy[4] ,double re_deriv_xybar[4], double im_deriv_xybar[4],
 double re_jacobian[64], double im_jacobian[64]){
	complex double** xy_arr = get_complex_variables_arr(4,Nmax + 1);
	complex double** exp_Il_arr = get_complex_variables_arr(2,kmax + 1);
	get_powers_array(xy,xy_arr,4,Nmax);
	get_powers_array(exp_Il,exp_Il_arr,2,kmax);
	SeriesTerm* term = first_term;
	complex double sum = 0;
	complex double dxy_tot[4] = {0,0,0,0};
	complex double dxybar_tot[4] = {0,0,0,0};
	complex double* jac_tot = calloc(64 , sizeof(complex double));
	complex double* jac = calloc(64 , sizeof(complex double));
	complex double dxy[4], dxybar[4];
	
	do{
		sum += evaluate_term_and_jacobian(term,exp_Il_arr,xy_arr,dxy,dxybar,jac);
		for(int i=0; i<4;i++){
			dxy_tot[i]+=dxy[i];
			dxybar_tot[i]+=dxybar[i];
		}
		for(int i=0; i < 64; i++) jac_tot[i] += jac[i];
		term = term->next;
	} while(term != 0);

	for(int i=0; i<4;i++){
		re_deriv_xy[i] = creal(dxy_tot[i]);
		re_deriv_xybar[i] = creal(dxybar_tot[i]);
		im_deriv_xy[i] = cimag(dxy_tot[i]);
		im_deriv_xybar[i] = cimag(dxybar_tot[i]);

		re_jacobian[INDEX(i,i)] = creal(jac_tot[INDEX(i,i)]);
		im_jacobian[INDEX(i,i)] = cimag(jac_tot[INDEX(i,i)]);
		re_jacobian[INDEX(i+4,i)] = creal(jac_tot[INDEX(i+4,i)]);
		im_jacobian[INDEX(i+4,i)] = cimag(jac_tot[INDEX(i+4,i)]);
		re_jacobian[INDEX(i,i+4)] =  re_jacobian[INDEX(i+4,i)] ;
		im_jacobian[INDEX(i,i+4)] =  im_jacobian[INDEX(i+4,i)] ;
		re_jacobian[INDEX(i+4,i+4)] = creal(jac_tot[INDEX(i+4,i+4)]);
		im_jacobian[INDEX(i+4,i+4)] = cimag(jac_tot[INDEX(i+4,i+4)]);

		for(int j=0; j<i; j++){

			re_jacobian[INDEX(i,j)] = creal(jac_tot[INDEX(i,j)]);
			im_jacobian[INDEX(i,j)] = cimag(jac_tot[INDEX(i,j)]);

			re_jacobian[INDEX(j,i)] = creal(jac_tot[INDEX(i,j)]);
			im_jacobian[INDEX(j,i)] = cimag(jac_tot[INDEX(i,j)]);

			re_jacobian[INDEX(i+4,j)] = creal(jac_tot[INDEX(i+4,j)]);
			im_jacobian[INDEX(i+4,j)] = cimag(jac_tot[INDEX(i+4,j)]);
			re_jacobian[INDEX(j,i+4)] = creal(jac_tot[INDEX(i+4,j)]);
			im_jacobian[INDEX(j,i+4)] = cimag(jac_tot[INDEX(i+4,j)]);

			re_jacobian[INDEX(i,j+4)] = creal(jac_tot[INDEX(i,j+4)]);
			im_jacobian[INDEX(i,j+4)] = cimag(jac_tot[INDEX(i,j+4)]);
			re_jacobian[INDEX(j+4,i)] = creal(jac_tot[INDEX(i,j+4)]);
			im_jacobian[INDEX(j+4,i)] = cimag(jac_tot[INDEX(i,j+4)]);

			re_jacobian[INDEX(i+4,j+4)] = creal(jac_tot[INDEX(i+4,j+4)]);
			im_jacobian[INDEX(i+4,j+4)] = cimag(jac_tot[INDEX(i+4,j+4)]);
			re_jacobian[INDEX(j+4,i+4)] = creal(jac_tot[INDEX(i+4,j+4)]);
			im_jacobian[INDEX(j+4,i+4)] = cimag(jac_tot[INDEX(i+4,j+4)]);
		}
	}
	free_complex_variables_arr(xy_arr);
	free_complex_variables_arr(exp_Il_arr);
	free(jac_tot);
	free(jac);

	*re = creal(sum);
	*im = cimag(sum);
}
