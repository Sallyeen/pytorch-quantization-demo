from torch.autograd import Function

# 假量化
class FakeQuantize(Function):

    # 调用 FakeQuantize， 即进行 量化与反量化操作
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None