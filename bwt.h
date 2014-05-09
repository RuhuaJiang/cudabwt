/*
 * bwt.h
 *
 *  Created on: May 2, 2014
 *      Author: jiang
 */
#include <stdint.h>
#ifndef BWT_H_
#define BWT_H_


typedef struct{
	uint8_t *seq;
	uint64_t length;
}Sequence;
int bwt(char *indexFile, uint32_t prefix_len);

#define THREADS_NUMBER  1024
#define BLOCKS_NUMBER 1024

#endif /* BWT_H_ */
