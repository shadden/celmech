#include <stdio.h>
int fmft(double **output, int nfreq, double minfreq, double maxfreq, int flag, 
	 double **input, long ndata);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
int fmft_wrapper(double out[][3], int nfreq, double minfreq, double maxfreq, int flag, 
	double in[][3], long ndata)
{
	double **output; double **input;
	double dt;
	int i,err;
	
	
	input = dmatrix(1,2,1,ndata);
	output = dmatrix(1,3*flag,1,nfreq);
	
	i =0;
	while(i<ndata){
		input[1][i] = in[i][1];
		input[2][i] = in[i][2];
		i++;
	}
	
	dt = in[1][0] - in[0][0];
	err = fmft(output, nfreq,dt * minfreq,dt * maxfreq, flag, input, ndata);
	for(i=0;i<nfreq; i++){
		// Frequency
		out[i][0] = output[3*flag-2][i+1] / dt;
		// Amplitude
		out[i][1] = output[3*flag-1][i+1];
		// Phase 
		out[i][2] = output[3*flag][i+1];
	}
	free_dmatrix(input,1,2,1,ndata);
	free_dmatrix(output,1,3*flag,1,nfreq);
	return err;
}
