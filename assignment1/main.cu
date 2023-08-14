#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include "cuda_runtime.h"
#include "cuda.h"

#include "tinycthread.h"
#include "util.h"

__global__ void generate_image_gpu(unsigned char* image, unsigned char* colormap, int width, int height, int max_iterations) {
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	double c_re = (col - width / 2.0) * 4.0 / width;
	double c_im = (row - height / 2.0) * 4.0 / width;

	double x = 0.0;
	double y = 0.0;

	unsigned int iteration = 0;

	while (x * x + y * y <= 4 && iteration < max_iterations) {
		double x_new = x * x - y * y + c_re;
		y = 2 * x * y + c_im;
		x = x_new;
		iteration++;
	}

	if (iteration > max_iterations) {
		iteration = max_iterations;
	}

    unsigned int index = 4 * width * row + 4 * col;
	unsigned char* c = &colormap[iteration * 3];
	image[index + 0] = c[0];
	image[index + 1] = c[1];
	image[index + 2] = c[2];
	image[index + 3] = 255;
}

int main(int argc, char** argv) {
    double times[REPEAT];
    struct timeb start, end;
    int i, r;
    char path[255];

    unsigned char* device_colormap;
    unsigned char* device_image;

    int colormap_size = (MAX_ITERATION+1) * 3;
    int image_size = WIDTH * HEIGHT * 4;

    unsigned char* colormap = (unsigned char*)malloc(colormap_size);
    unsigned char* image = (unsigned char*)malloc(image_size);

    cudaMalloc((void**)&device_colormap, colormap_size);
    cudaMalloc((void**)&device_image, image_size);

    dim3 block_dimension(BLOCK_COLS, BLOCK_ROWS);
	dim3 grid_dimension(WIDTH / BLOCK_COLS, HEIGHT / BLOCK_ROWS);

    init_colormap(MAX_ITERATION, colormap);

    for (r = 0; r < REPEAT; r++) {
        memset(image, 0, WIDTH * HEIGHT * 4);

        ftime(&start);

        cudaMemcpy(device_image, image, image_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_colormap, colormap, colormap_size, cudaMemcpyHostToDevice);

        generate_image_gpu <<<block_dimension,grid_dimension>>>(device_image, device_colormap, WIDTH, HEIGHT, MAX_ITERATION);

        cudaDeviceSynchronize();

        cudaMemcpy(image, device_image, image_size, cudaMemcpyDeviceToHost);

        ftime(&end);
        times[r] = end.time - start.time + ((double)end.millitm - (double)start.millitm)/1000.0;

        sprintf(path, IMAGE, "cpu", r);
        save_image(path, image, WIDTH, HEIGHT);
        progress("cpu", r, times[r]);
    }
    report("cpu", times);

    cudaFree(device_image);
    cudaFree(device_colormap);
    free(image);
    free(colormap);
    return 0;
}
