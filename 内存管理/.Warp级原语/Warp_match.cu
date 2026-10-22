#include <stdio.h>
#include <cuda_runtime.h>
    
__global__ void test(){
    int val = (threadIdx.x %2 == 0)?10:20;
    int pred;
    unsigned any_mask = __match_any_sync(0xFFFFFFFF,val);
    unsigned all_mask = __match_all_sync(0xFFFFFFFF,val,&pred);

    if(threadIdx.x == 0){
        printf("any_mask = %d\n",any_mask);
        printf("all_mask = %d\n",all_mask);
        printf("pred = %d\n", pred);
    }
}
int main(){
    test<<<1,32>>>();
    cudaDeviceSynchronize();
    return 0;
}
