import torch
import flashinfer

assert torch.cuda.is_available()
torch.manual_seed(0)
q = torch.randn(8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(256, 2, 64, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
out = flashinfer.single_decode_with_kv_cache(q, k, v)
ref = torch.nn.functional.scaled_dot_product_attention(
    q[None, :, None, :], k.transpose(0, 1)[None],
    v.transpose(0, 1)[None], enable_gqa=True)[0, :, 0]
torch.testing.assert_close(out, ref, atol=0.003, rtol=0.003)
torch.cuda.synchronize()
print("PASS: FlashInfer GPU decode matches PyTorch attention")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("FlashInfer:", flashinfer.__version__)
