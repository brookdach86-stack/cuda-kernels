// @Author: Wain (brookdach86-stack@users.noreply.github.com)
// @GitHub: https://github.com/brookdach86-stack
// @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
#include <math.h>
#include <float.h>

void softmax_cpu(float *input, float *output, int n)
{
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (input[i] > max_val)
        {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < n; i++)
    {
        output[i] /= sum;
    }
}
