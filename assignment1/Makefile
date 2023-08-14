all: mandelbrot

run: mandelbrot
	./mandelbrot

mandelbrot: *.cpp *.h
	nvcc -o mandelbrot *.cpp *.cu -lrt -lpthread -lm

clean:
	-rm mandelbrot mandelbrot.zip *.png report.txt

pack:
	-rm mandelbrot.zip
	zip mandelbrot.zip ./*.h ./*.c Makefile

.PHONY: all clean run pack
