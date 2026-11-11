NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80 -lineinfo -Xcompiler -fPIC -Xcompiler -fopenmp
BUILD = build

all: $(BUILD)/libcubcr_gpu_e3.so $(BUILD)/libcubcr_gpu_copy.so

$(BUILD)/libcubcr_gpu_e3.so: e3_api.cu cubcr_gpu.cu cubcr.h cubcr_gpu.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -shared e3_api.cu -o $@

$(BUILD)/libcubcr_gpu_copy.so: cubcr_gpu.cu cubcr.h cubcr_gpu.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -shared cubcr_gpu.cu -o $@

clean:
	rm -rf $(BUILD)
.PHONY: all clean
