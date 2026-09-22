CC     = gcc
CFLAGS = -O2 -Wall

ckpt_prealloc: ckpt_prealloc.c
	$(CC) $(CFLAGS) -o $@ $< -lpthread -lrt

clean:
	rm -f ckpt_prealloc

.PHONY: clean
