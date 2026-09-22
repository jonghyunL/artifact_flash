CC        ?= gcc
CUDA_PATH ?= /usr/local/cuda
CFLAGS     = -O2 -Wall -fPIC -std=gnu11
INCLUDES   = -I$(CUDA_PATH)/include -I.
LDFLAGS    = -shared -ldl -lpthread -lssl -lcrypto

libphos_intercept.so: libphos_intercept.c ckpt_crypto.c ckpt_crypto.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ libphos_intercept.c ckpt_crypto.c $(LDFLAGS)

clean:
	rm -f libphos_intercept.so

.PHONY: clean
