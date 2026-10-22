#include<cstdio>
#include<cuda_runtime.h>

__global__ void vote_kernel(){
    int val = threadIdx.x;
    int pred = (val>5);

    int all_res = __all_sync(0xFFFFFFFF,pred);
    int any_res = __any_sync(0xFFFFFFFF,pred);
    unsigned ballot = __ballot_sync(0xFFFFFFFF,pred);
    
    if(threadIdx.x == 0){
        printf("val: 0 1 2 3 4 5 6 7\n");
        printf("pred: ");
        for(int i = 0;i < 8;++i){
            printf("%d ",i>5);
        }
        printf("\n\n");

        printf("all_res: %d\n",all_res);
        printf("any_res: %d\n",any_res);
        printf("ballot: %d\n",ballot);
        for(int i = 7;i>=0;i--){
            printf("%d ",(ballot>>i)&1);
        }
        printf("\n\n");
    }
}

int main(){
    vote_kernel<<<1,8>>>();
    cudaDeviceSynchronize();
    return 0;
}


