# rnn_gpu_speed

<div>
<figure >
<img style="border:1px solid; border-color:#daa520ff; margin-left:auto;margin-right:auto;text-align: center;" src="https://hergott.github.io/assets/img/gpu_speed/gpu_speed_plot.png" alt="Recurrent neural networks should run faster on a GPU, and this advantage is amplified by using the CUDNN library." />
<figcaption style="color: #156e82; text-align: center; font-size:100%; font-style: italic; font-weight:normal;margin-left:auto;margin-right:auto;"></figcaption>
</figure> 
</div>

This is a simple [Python](https://www.python.org/) + [TensorFlow](https://www.tensorflow.org/) speed test of a recurrent neural network (RNN) using:

1. CPU
2. GPU+CUDA
3. GPU+CUDA+CUDNN 

The results will vary greatly depending on hardware and the size of the RNN.

But the basic pattern should be the same: a recurrent neural network learns faster on a GPU than on a CPU, and this GPU advantage is amplified by using the [CUDNN library](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn).

<div>
<figure>
<img style="border:1px solid; border-color:#daa520ff; margin-left:auto;margin-right:auto;text-align: center;" src="https://hergott.github.io/assets/img/gpu_speed/gpu_loss_plot.png" alt="In this example, the CUDNN RNN library also delivers a smoother learning path." />
<figcaption style="color: #156e82; text-align: center; font-size:100%; font-style: italic; font-weight:normal;margin-left:auto;margin-right:auto;"></figcaption>
</figure> 
</div>

In this example, the CUDNN RNN library also delivers a smoother learning path.

This helps to show that, in addition to the obvious advantages GPUs offer in parallel computing, GPU-accelerated computing can also be useful for sequential tasks.



