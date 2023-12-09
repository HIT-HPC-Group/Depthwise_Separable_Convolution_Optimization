ROCM_PATH= /public/software/compiler/dtk-21.10.1
HIPCC=/opt/rocm_install/bin/hipcc
HIP_PATH=$(ROCM_PATH)/hip
CXXFLAGS += -std=c++11 
MIOPEN_INC = $(ROCM_PATH)/miopen/include/miopen
MIOPEN_LIB =$(ROCM_PATH)/lib
kernel:  DCU_Pointwise_Kernel.cpp
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -D AMD_PLATFORM $(INCLUDECLASS) $(LIBPATH) $(LIBS)  $^ -o $@
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

.PHONY: clean

clean:
	rm -f kernel *.o
