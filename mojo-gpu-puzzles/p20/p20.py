import torch
from max.torch import CustomOpLibrary


def conv1d_pytorch(
    input_tensor: torch.Tensor, kernel_tensor: torch.Tensor
) -> torch.Tensor:
    """
    1D convolution using our custom PyTorch operation.

    This demonstrates the transition from MAX Graph (p15) to PyTorch CustomOpLibrary.
    Uses the EXACT same Mojo kernel, but different Python integration!
    """
    # Load our custom operations
    mojo_kernels = Path(__file__).parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    # Create output tensor with same shape as input
    output_tensor = torch.empty_like(input_tensor)

    # Call our custom conv1d operation with explicit output tensor
    # The Mojo signature expects: (out, input, kernel)
    conv1d = ops.conv1d[
        {
            "input_size": input_tensor.shape[0],
            "conv_size": kernel_tensor.shape[0],
        }
    ]

    torch.compile(
        conv1d(
            output_tensor,
            input_tensor,
            kernel_tensor
        )
    )

    return output_tensor


