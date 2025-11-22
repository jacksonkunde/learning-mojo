from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


alias SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
alias layout = Layout.row_major(SIZE)
alias GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
alias BLOCK_DIM_X = 1 << log2_ceil(SIZE)

fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # read input into shared memory
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    shared_input = LayoutTensor[
        dtype, 
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()

    # init the values for the out of bounds regions to be min possible value, not sure how to do this...
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[global_i])
    shared_max[local_i] = val

    barrier()

    # max finding algo
    var stride: UInt = BLOCK_DIM_X // 2
    while stride > 0:
        if local_i < stride and shared_max[local_i] < shared_max[local_i + stride]:
            shared_max[local_i] = shared_max[local_i + stride]
        stride //= 2
        barrier()

    var block_max = shared_max[0]

    # Element-wise exponentiate and store exp_val for later
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_input[local_i] = exp_val
    
    barrier()

    # compute total sum
    stride = BLOCK_DIM_X // 2
    while stride > 0:
        if local_i < stride:
            shared_input[local_i] = shared_input[local_i] + shared_input[local_i + stride]
        stride //= 2
        barrier()

    var total_sum = shared_input[0]
    
    # Write normalized output
    if global_i < input_size:
        output[global_i] = exp_val / total_sum






# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 10 lines)
    ...


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            alias kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
    
    

# ANCHOR_END: broadcast_add
def main():
    with DeviceContext() as ctx:
        alias dtype = DType.float32
        output_buffer = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buffer.enqueue_fill(0)
        input_buffer = ctx.enqueue_create_buffer[dtype](SIZE)
        
        # Populate input buffer with values
        with input_buffer.map_to_host() as in_buf_host:
            for i in range(SIZE):
                in_buf_host[i] = i

        # Convert buffers to layout tensors
        var output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buffer)
        var input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buffer)

        alias kernel = softmax_gpu_kernel[layout, SIZE, dtype]
        ctx.enqueue_function_checked[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=GRID_DIM_X,
            block_dim=BLOCK_DIM_X,
        ) 
        ctx.synchronize()

        with output_buffer.map_to_host() as out_buf_host:
            print("out:", out_buf_host)



