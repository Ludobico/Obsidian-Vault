---
_filters: []
_contexts: []
_links: []
_sort:
  field: rank
  asc: false
  group: false
_template: ""
_templateName: ""
---
CUDA(Compute Unified Device Architecture)는 NVIDIA가 개발한 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델로, 그래픽 처리 장치(GPU)를 사용하여 계산 성능을 대폭 향상시키기 위한 기술입니다. CUDA를 통해 개발자는 GPU의 병렬 처리 능력을 활용하여 계산 집약적인 애플리케이션을 보다 효율적으로 실행할 수 있습니다.

## Features

- GPU는 수천 개의 작은 연산 유닛(코어)을 가지고 있어 대규모 병렬 처리를 수행할 수 있습니다. 이를 통해 동시에 많은 작업을 병렬로 처리할 수 있습니다.

- CUDA는 C, C++, Python 등의 언어에서 GPU를 프로그래밍할 수 있는 확장 기능을 제공합니다. CUDA 코드에서는 GPU에서 실행될 커널 함수를 정의하고, 이를 호출하여 병렬 처리를 수행합니다.

- CUDA는 여러 종류의 메모리 공간을 제공하여 GPU와 CPU 간의 데이터 저송 및 처리를 최적화합니다. 여기에는 전역 메모리, 공유 메모리, 상수 메모리, 텍스쳐 메모리 등이 포함됩니다.

- CUDA는 다양한 수학, 신경망, 영상 처리 등 고성능 컴퓨팅을 위한 라이브러리를 제공합니다. 대표적으로 cuBLAS(선형 대수), cuDNN(딥러닝), cuFFT(푸리에 변환) 등이 있습니다.

## Architectures of CUDA Programming

- CPU는 호스트, GPU는 디바이스로 불립니다. CUDA 프로그램은 일반적으로 호스트 코드와 디바이스 코드로 나뉩니다.

- GPU에서 실행하는 커널 함수는 수많은 스레드가 동시에 실행됩니다.

```cpp
__global__ void add(int *a, int *b, int *c){
int index = thredIdx.x;
c[index] = a[index] + b[index]
}
```

- CPU와 GPU 간의 데이터를 전송할 수 있습니다.

```cpp
int *d_a, *d_b, *d_c;
cudaMalloc((void **)&d_a, size);
cudaMalloc((void **)&d_b, size);
cudaMalloc((void **)&d_c, size);
cudaMemcpy((d_a, a, size, cudaMemcpyHostToDevice));
cudaMemcpy((d_b, a, size, cudaMemcpyHostToDevice));
```

- 호스트 코드와 디바이스 코드를 호출시킬 수 있습니다.

```cpp
add<<<1, N>>>(d_a, d_b, d_c);
```

- 계산된 결과를 GPU에서 CPU로 전송합니다.

```cpp
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
```

