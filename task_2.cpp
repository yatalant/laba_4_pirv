#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

// CPU-версия поворота изображения
void rotateImageCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Поворот на 90 градусов по часовой стрелке
            output[x * height + (height - y - 1)] = input[y * width + x];
        }
    }
}

// GPU-ядро для поворота изображения
__global__ void rotateImageCUDA(const unsigned char* input, unsigned char* output, int width, int height) {
    // Получаем 2D координаты пикселя в выходном изображении
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Проверяем границы изображения
    if (x < width && y < height) {
        // Вычисляем координаты в исходном изображении
        int original_x = height - y - 1;
        int original_y = x;
        
        // Выполняем поворот
        output[y * width + x] = input[original_y * width + original_x];
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const int image_size = width * height;
    size_t size = image_size * sizeof(unsigned char);
    
    // Выделяем память на хосте
    unsigned char *h_input = new unsigned char[image_size];
    unsigned char *h_output_cpu = new unsigned char[image_size];
    unsigned char *h_output_gpu = new unsigned char[image_size];
    
    // Инициализируем входное изображение случайными значениями
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> dis(0, 255);
    
    for (int i = 0; i < image_size; ++i) {
        h_input[i] = dis(gen);
    }
    
    // Замеряем время выполнения CPU-версии
    auto start_cpu = std::chrono::high_resolution_clock::now();
    rotateImageCPU(h_input, h_output_cpu, width, height);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    
    // Выделяем память на устройстве
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    // Копируем данные на устройство
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Настраиваем параметры запуска ядра
    dim3 blockDim(16, 16); // Блоки 16x16 потоков
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Замеряем время выполнения GPU-версии (включая копирование данных)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Запускаем ядро
    rotateImageCUDA<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    
    // Копируем результат обратно на хост
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Проверяем корректность результатов
    bool correct = true;
    for (int i = 0; i < image_size; ++i) {
        if (h_output_cpu[i] != h_output_gpu[i]) {
            correct = false;
            break;
        }
    }
    
    // Выводим результаты
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "CPU time: " << cpu_time.count() * 1000 << " ms\n";
    std::cout << "GPU time: " << gpu_time_ms << " ms (including data transfer)\n";
    std::cout << "Results are " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Освобождаем память
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
