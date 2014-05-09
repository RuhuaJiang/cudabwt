/*
 * main.cu
 *
 *  Created on: May 2, 2014
 *      Author: jiang
 */


#include <stdio.h>
#include <stdlib.h>
#include "bwt.h"

int main(int argc, char *argv[])
{
	char *indexFile;
	indexFile = argv[1];
	bwt(indexFile,2);
	return 0;
}

