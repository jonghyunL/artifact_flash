CXX       ?= g++
CC        ?= gcc
CUDA_PATH ?= /usr/local/cuda

CFLAGS     = -O2 -Wall -fPIC -std=gnu11
CXXFLAGS   = -O2 -Wall -fPIC -std=gnu++17
INCLUDES   = -I$(CUDA_PATH)/include -I.
LDFLAGS    = -shared -ldl -lpthread -lssl -lcrypto

libgcr_intercept.so: libgcr_intercept.o dirty_templates.o ckpt_crypto.o
	$(CXX) $^ -o $@ $(LDFLAGS)

libgcr_intercept.o: libgcr_intercept.cpp gcr_runtime.h ckpt_crypto.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c libgcr_intercept.cpp

dirty_templates.o: dirty_templates.cpp dirty_templates_kernels.cpp gcr_runtime.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c dirty_templates.cpp

ckpt_crypto.o: ckpt_crypto.c ckpt_crypto.h
	$(CC) $(CFLAGS) $(INCLUDES) -c ckpt_crypto.c

clean:
	rm -f libgcr_intercept.so libgcr_intercept.o dirty_templates.o ckpt_crypto.o

.PHONY: clean
