################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../is.c 

CU_SRCS += \
../bwt.cu \
../main.cu 

CU_DEPS += \
./bwt.d \
./main.d 

OBJS += \
./bwt.o \
./is.o \
./main.o 

C_DEPS += \
./is.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O2 -gencode arch=compute_20,code=sm_20 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O2 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O2 -gencode arch=compute_20,code=sm_20 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -O2 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


