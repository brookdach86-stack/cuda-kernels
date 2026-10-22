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
