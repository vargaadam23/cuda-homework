all: nonce

run: nonce
	./nonce

profile: nonce
	sudo nvprof ./nonce

nonce: *.cu *.cuh
	nvcc -o nonce *.cu -rdc=true -lrt -lpthread -lm

clean:
	-rm nonce nonce.zip *.png report.txt

pack:
	-rm nonce.zip
	zip nonce.zip ./*.h ./*.c Makefile

.PHONY: all clean run pack
