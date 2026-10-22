#include<cstdio>
#include<cuda_runtime.h>

__global__ void warShuffleKernel(){
    int ID = threadIdx.x;
    float val = (float)ID;

    // Warp内广播.分歧
    float val_broadcast = __shfl_sync(0xffffffff,val,0);

    // Warp内规约求和.原语
    for(int offset = 16; offset > 0; offset>>=1){
        val += __shfl_down_sync(0xffffffff,val,offset);
    }
    if(ID == 0){
        printf("[线程 0] 广播拿到的值: %.1f, 规约求和的总总和: %.1f (期望值: 496.0)\n", val_broadcast, val);
    }
    if (ID == 5) {
        printf("[线程 5] 广播拿到的值: %.1f, 此时它手里的求和中途值: %.1f\n", val_broadcast, val);
    }
}