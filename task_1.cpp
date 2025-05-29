#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

// CPU-версия возведения в степень
void powerArrayCPU(const float* A, float* B, int N, float p) {
    for (int i = 0; i < N; ++i) {
        B[i] = powf(A[i], p);
    }
}

// GPU-ядро для возведения в степень
__global__ void powerArrayCUDA(const float* A, float* B, int N, float p) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем границы массива
    if (idx < N) {
        B[idx] = powf(A[idx], p);
    }
}

int main() {
    const int N = 500000;
    const float p = 0.5f;
    size_t size = N * sizeof(float);
    
    // Выделяем память на хосте
    float *h_A = new float[N];
    float *h_B_cpu = new float[N];
    float *h_B_gpu = new float[N];
    
    // Инициализируем массив A случайными значениями
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);
    
    for (int i = 0; i < N; ++i) {
        h_A[i] = dis(gen);
    }
    
    // Замеряем время выполнения CPU-версии
    auto start_cpu = std::chrono::high_resolution_clock::now();
    powerArrayCPU(h_A, h_B_cpu, N, p);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    
    // Выделяем память на устройстве
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    
    // Копируем данные на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    // Настраиваем параметры запуска ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Замеряем время выполнения GPU-версии (включая копирование данных)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Запускаем ядро
    powerArrayCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N, p);
    cudaDeviceSynchronize();
    
    // Копируем результат обратно на хост
    cudaMemcpy(h_B_gpu, d_B, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Проверяем корректность результатов
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_B_cpu[i] - h_B_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    // Выводим результаты
    std::cout << "CPU time: " << cpu_time.count() * 1000 << " ms\n";
    std::cout << "GPU time: " << gpu_time_ms << " ms (including data transfer)\n";
    std::cout << "Results are " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Освобождаем память
    delete[] h_A;
    delete[] h_B_cpu;
    delete[] h_B_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    
    return 0;
}
