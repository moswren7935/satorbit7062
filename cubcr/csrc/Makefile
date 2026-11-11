CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -march=native -mprefer-vector-width=512 -fopenmp -std=c++17 -fPIC -Wall -Wno-unknown-pragmas
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80 -lineinfo -Xcompiler -fPIC -Xcompiler -fopenmp
BUILD = build

all: $(BUILD)/libcubcr_cpu.so $(BUILD)/libcubcr_gpu.so

$(BUILD)/libcubcr_cpu.so: cubcr_cpu.cpp cubcr.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared cubcr_cpu.cpp -o $@

$(BUILD)/libcubcr_gpu.so: cubcr_gpu.cu cubcr.h cubcr_gpu.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -shared cubcr_gpu.cu -o $@

cpu: $(BUILD)/libcubcr_cpu.so
gpu: $(BUILD)/libcubcr_gpu.so

clean:
	rm -rf $(BUILD)
.PHONY: all cpu gpu clean
