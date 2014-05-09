/*
 * bwt.cu
 *
 *  Created on: May 2, 2014
 *      Author: jiang
 */
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include <thrust/version.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "kseq.h"
#include "bwt.h"
// STEP 1: declare the type of file handler and the read() function
KSEQ_INIT(gzFile, gzread)
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
unsigned char nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};
int fasta_parser(char *indexFile,Sequence &sequence)
{
	gzFile fp;
	kseq_t *seq;
	int l;
	fp = gzopen(indexFile, "r"); // STEP 2: open the file handler
	seq = kseq_init(fp); // STEP 3: initialize seq
	while ((l = kseq_read(seq)) >= 0) { // STEP 4: read sequence
		//printf("name: %s\n", seq->name.s);
		//if (seq->comment.l) printf("comment: %s\n", seq->comment.s);
		//printf("seq: %s\n", seq->seq.s);
		sequence.length = seq->seq.l;
		sequence.seq = (uint8_t*) calloc(sequence.length, 1);
		for (int i = 0; i != seq->seq.l; i++)
			sequence.seq[i] = nst_nt4_table[(int) seq->seq.s[i]];
		//if (seq->qual.l) printf("qual: %s\n", seq->qual.s);
	}
	printf("return value: %d\n", l);
	kseq_destroy(seq); // STEP 5: destroy seq
	gzclose(fp); // STEP 6: close the file handler
	return 0;
}



__global__ void count_suffixs(uint32_t *d_A ,const uint8_t *d_sequence,uint64_t l, int prefix_len,uint64_t seq_len,uint32_t prefix)
{
	uint32_t idx = threadIdx.x + blockDim.x *blockIdx.x;
	uint64_t start_pos = idx * l, end_pos = (idx+1) * l; //end_pos not included
	uint32_t count = 0;
	uint32_t value  =0;
	uint32_t debug_count= 0;
	//extern __shared__ uint32_t s_A[];
	if(idx == BLOCKS_NUMBER*THREADS_NUMBER-1) {// last thread
		end_pos = seq_len;
		//printf("thread  %d  is the last one!\n",idx);
	}
	//Each thread scans through start_pos to end_pos
	if(idx != BLOCKS_NUMBER*THREADS_NUMBER-1)
	{
		for(uint64_t i = 0 ; i< l;  i++){
			value = 0;
			for(int k = 0; k < prefix_len; k++){
					value <<= 2;
					value += (uint32_t)d_sequence[start_pos+k+i];
			}
			debug_count ++;
			if(value == prefix) count++;
		}
	}
	else{ //last thread
		for(uint64_t i = 0 ; i< l;  i++){
			if(start_pos + i <= end_pos - prefix_len) //all threads except for last part of last thread must satisfy
			{
				value = 0;
				for(int k = 0; k < prefix_len; k++){
						value <<= 2;
						value += (uint32_t)d_sequence[start_pos+k+i];
				}
				debug_count ++;
				if(value == prefix) count++;
			}
			else if ((start_pos + i > end_pos - prefix_len)  && (start_pos + i < end_pos))
			{ //The last part of last thread is reading end of the text
				int left = seq_len - (start_pos + i);
				value = 0;
				for(int k = 0; k < left; k++)
				{
					value <<= 2;
					value += (uint32_t)d_sequence[start_pos+i+k];
				}
				for(int k = 0; k < prefix_len -left; k++)
				{
					value <<= 2;
					value += (uint32_t)d_sequence[k];
				}
				debug_count ++;
				if(value == prefix) count++;
			}
			else break;
		}
	}
	d_A[idx] = count;
	//printf("thread  %d  run %d times\n",idx,debug_count);
   //printf("prefix is %d thread  %d  count %llu end_pos %d\n",prefix,threadIdx.x, count,end_pos);
}

__global__ void get_suffix_block(const uint32_t *d_A, const uint8_t *d_sequence,uint64_t l, int prefix_len,uint64_t seq_len,uint32_t prefix,uint32_t *d_B){
	uint32_t idx  = threadIdx.x + blockDim.x *blockIdx.x;
	//printf("thread  %d  value  %d\n",threadIdx.x,d_A[idx]);
	uint64_t start_pos = idx * l, end_pos = (idx+1) * l; //end_pos not included
	uint32_t b = 0;
	uint32_t value  =0;
	uint8_t tail_seq[10];
	uint32_t debug_count= 0;
	if(idx == BLOCKS_NUMBER*THREADS_NUMBER-1) {// last thread
			end_pos = seq_len;
	}
	//Each thread scans through start_pos to end_pos
		if(idx != BLOCKS_NUMBER*THREADS_NUMBER-1)
		{
			for(uint64_t i = 0 ; i< l;  i++){
				value = 0;
				for(int k = 0; k < prefix_len; k++){
						value <<= 2;
						value += (uint32_t)d_sequence[start_pos+k+i];
				}
				debug_count ++;
				if(value == prefix){
					d_B[d_A[idx] + b]= start_pos+i ;
					b++;

				}
			}
		}
		else{ //last thread
			for(uint64_t i = 0 ; i< l;  i++){
				if(start_pos + i <= end_pos - prefix_len) //all threads except for last part of last thread must satisfy
				{
					value = 0;
					for(int k = 0; k < prefix_len; k++){
							value <<= 2;
							value += (uint32_t)d_sequence[start_pos+k+i];
					}
					debug_count ++;
					if(value == prefix) {
						d_B[d_A[idx] + b]= start_pos+i ;
						b++;

					}
				}
				else if ((start_pos + i > end_pos - prefix_len)  && (start_pos + i < end_pos))
				{ //The last part of last thread is reading end of the text
					int left = seq_len - (start_pos + i);
					value = 0;
					for(int k = 0; k < left; k++)
					{
						value <<= 2;
						value += (uint32_t)d_sequence[start_pos+i+k];
					}
					for(int k = 0; k < prefix_len -left; k++)
					{
						value <<= 2;
						value += (uint32_t)d_sequence[k];
					}
					debug_count ++;
					if(value == prefix) {
						d_B[d_A[idx] + b]= start_pos+i ;
						b++;

					}
				}
				else break;
			}
		}



}

__global__ void map_pos2key(const uint8_t  *d_sequence,uint64_t seq_length,int prefix_len, int v_len, const uint32_t *d_B,uint32_t B_len, uint32_t *d_B_key)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < B_len){
		uint64_t position = d_B[idx];
		uint32_t value = 0;
		if(position <= (uint64_t)(seq_length - (v_len+1))){
			for(int i = prefix_len; i < v_len; i++)
			{
				value <<= 2;
				value += (uint32_t)d_sequence[position + i];
			}

		}
		else
		{
			int left = seq_length - position;
			value = 0;
			for(int k = 0; k < left; k++)
			{
				value <<= 2;
				value += (uint32_t)d_sequence[position+k];
			}
			for(int k = 0; k < prefix_len -left; k++)
			{
				value <<= 2;
				value += (uint32_t)d_sequence[k];
			}

		}
		d_B_key[idx] = value;
	}

}
/*
__global__ void checking(uint32_t *d_B_key, uint32_t B_len)
{


}
*/
void suffix_blocking(uint32_t prefix, const uint8_t  *d_sequence, uint8_t *h_sequence,uint64_t seq_length,int prefix_len, FILE *bwt_file)
{

	 uint32_t *d_A = NULL; //counting array
	 uint32_t *h_A = NULL;
	 uint32_t *h_B = NULL;
	 uint32_t *d_B = NULL;
	 uint32_t *h_B_key = NULL;
	 uint32_t *d_B_key = NULL;

	 fprintf(stderr,"block %llu processing\n",prefix);
	 //printf("version %d %d",THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION);
	 uint64_t l = seq_length/(THREADS_NUMBER*BLOCKS_NUMBER) +1;

	 //printf("l is %d\n",l);
	 //exit(1);

	 h_A = (uint32_t*) malloc(sizeof(uint32_t) * (THREADS_NUMBER*BLOCKS_NUMBER));
	 if(h_A == NULL){
		 fprintf(stderr,"Out of host memory!\n");

	 }

	 CUDA_CHECK_RETURN(cudaMalloc((void**) &d_A, sizeof(uint32_t) * (THREADS_NUMBER*BLOCKS_NUMBER)));
	 count_suffixs<<<BLOCKS_NUMBER, THREADS_NUMBER>>>(d_A, d_sequence,l,prefix_len,seq_length, prefix);
	 CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	 //exit(1);
	 cudaMemcpy(h_A , d_A, sizeof(uint32_t) * (THREADS_NUMBER*BLOCKS_NUMBER), cudaMemcpyDeviceToHost);
	 uint32_t last = h_A[THREADS_NUMBER*BLOCKS_NUMBER-1];

	 //fprintf(stderr,"count suffix finish\n");
	 //exclusive prefix sum
	 thrust::exclusive_scan(h_A,h_A+THREADS_NUMBER*BLOCKS_NUMBER,h_A);

	 //fprintf(stderr,"scan finish\n");
	 /*
	 for(int i=0; i < THREADS_NUMBER; i++)
	 {
		 fprintf(stdout,"%d ",h_A[i]);
	 }
	 */



	 //calculate number of elments in B
	 uint32_t B_len = h_A[THREADS_NUMBER*BLOCKS_NUMBER-1] + last;
	 h_B = (uint32_t*) malloc(sizeof(uint32_t) * B_len);
	 if(h_B == NULL){
			 fprintf(stderr,"Out of host memory!\n");
			 exit(1);
	 }
	 CUDA_CHECK_RETURN(cudaMalloc((void**) &d_B, sizeof(uint32_t) * B_len));
	 cudaMemcpy(d_A , h_A, sizeof(uint32_t) * (THREADS_NUMBER*BLOCKS_NUMBER), cudaMemcpyHostToDevice);
	 //fprintf(stdout,"\nB_len%d!\n",B_len);
	 get_suffix_block<<<BLOCKS_NUMBER, THREADS_NUMBER>>>(d_A, d_sequence,l, prefix_len, seq_length,prefix,d_B);
	 CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	 cudaMemcpy(h_B , d_B, sizeof(uint32_t) * B_len, cudaMemcpyDeviceToHost);
	 /*
	 fprintf(stdout,"\nh_B:");
	 for(int i=0; i < B_len; i++)
	 {
	 	fprintf(stdout,"%d ",h_B[i]);
	 }
	 */
	 CUDA_CHECK_RETURN(cudaFree(d_A));
	 //fprintf(stderr,"cal member finish\n");

	 //Map position to a key
	 int thread_num =1024;
	 int block_num = B_len/thread_num +1 ;
	 h_B_key = (uint32_t*) malloc(sizeof(uint32_t) * B_len);
	 if(h_B_key == NULL){
			 fprintf(stderr,"Out of host memory!\n");
			 exit(1);
	 }
	 CUDA_CHECK_RETURN(cudaMalloc((void**) &d_B_key, sizeof(uint32_t) * B_len));
	 int v_len = 16 + prefix_len; //assume we can use first v_len characters to distinguish suffixes
	 map_pos2key<<<block_num, thread_num>>>(d_sequence,seq_length,prefix_len, v_len,d_B, B_len,d_B_key);
	 cudaMemcpy(h_B_key , d_B_key, sizeof(uint32_t) * B_len, cudaMemcpyDeviceToHost);

	 /*
	 fprintf(stdout,"\n");
	 for(int i=0; i < B_len; i++)
	 {
	 	fprintf(stdout,"%llu ",h_B_key[i]);
	 }
	 */
	 //Radix sort
	 thrust::sort_by_key(h_B_key,h_B_key + B_len, h_B);
	 //fprintf(stderr,"radix sort finish\n");
	 /*
	 fprintf(stdout,"\nSorted");
	 for(int i=0; i < B_len; i++)
	 {
		 	fprintf(stdout,"(%llu,%llu) ",h_B[i],h_B_key[i]);
	 }
	 */

	 //Construct the BWT
	 //FIXME  Now h_B is almost the suffix array, possibly with some minor errors, since we need a refinement
	 uint8_t bwt_value;
	 for(uint32_t i =0; i<B_len; i++)
	 {
		 if(h_B[i]==0) bwt_value = h_sequence[seq_length -1];
		 else bwt_value = h_sequence[h_B[i]-1];
		 fseek(bwt_file, 0, SEEK_END);
		 fprintf(bwt_file, "%d",bwt_value);
	 }

	 //Parallel checking algorithm

}

int bwt(char *indexFile, uint32_t prefix_len){
	uint8_t *d_sequence =NULL;
	int forward_only = 1;
	uint64_t buf_length;
	Sequence sequence;
	fasta_parser(indexFile,sequence);
	//fprintf(stdout,"seqlength %d ",  sequence.length);
	/*
	fprintf(stdout,"seq:");
	for (int i = 0; i != sequence.length; i++)
	{
		fprintf(stdout,"%d",sequence.seq[i]);
	}
	fprintf(stdout,"\n");
	*/
	FILE *bwt_file;
	char *bwt_fn;
	bwt_fn = (char*) calloc(strlen(indexFile) + 10, 1);
	strcpy(bwt_fn, indexFile);
	strcat(bwt_fn, ".bwt");

	bwt_file = fopen(bwt_fn, "wb");


	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_sequence, sizeof(uint8_t) * sequence.length));
		CUDA_CHECK_RETURN(
			cudaMemcpy(d_sequence, sequence.seq, sizeof(uint8_t) * sequence.length, cudaMemcpyHostToDevice)); //copy text to global memory of GPU
	//There will be 4^prefix_len number of blocks
	uint32_t number_blocks = pow(4.0,(int)prefix_len);
	for(uint32_t p = 0; p < number_blocks; p++){
		  suffix_blocking(p,d_sequence,sequence.seq,sequence.length,prefix_len,bwt_file);
		  //exit(1);
	}
	return 0;
}
