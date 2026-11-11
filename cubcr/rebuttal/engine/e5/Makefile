CUOBJDUMP ?= cuobjdump
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80 -lineinfo -Xcompiler -fPIC -Xcompiler -fopenmp
BUILD = build

all: $(BUILD)/libcubcr_e5_gpu.so

$(BUILD)/libcubcr_e5_gpu.so: cubcr_e5_gpu.cu cubcr_e5.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -shared cubcr_e5_gpu.cu -o $@

resource: $(BUILD)/libcubcr_e5_gpu.so
	$(CUOBJDUMP) --dump-resource-usage $(BUILD)/libcubcr_e5_gpu.so > $(BUILD)/resource_usage_e5.txt
	$(CUOBJDUMP) --dump-resource-usage ../../../csrc/build/libcubcr_gpu.so > $(BUILD)/resource_usage_paper.txt

ptxas-v:
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c cubcr_e5_gpu.cu -o $(BUILD)/ptxas_check.o 2> $(BUILD)/ptxas_v.txt

clean:
	rm -rf $(BUILD)
.PHONY: all resource ptxas-v clean
