"""
Time fwd+bwd of the BRIGHT siamese wrappers per model, with cudnn.deterministic on and off.
Usage (from the earthshift repo dir, on a GPU node):
    python bench_prithvi.py prithvi dinov3_large_sat
"""
import os, sys, time, torch
sys.path.insert(0, os.getcwd())
from model_manager import ModelManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')
B = 32 if device.type == 'cuda' else 2
x = torch.randn(B, 4, 224, 224, device=device)
y = torch.randint(0, 4, (B, 224, 224), device=device)


def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()


for name in sys.argv[1:]:
    mm = ModelManager(name, 'semseg', 4, 3, 'rgb', device, fusion='siamese', post_channels=1, post_wavelength='sar')
    model = mm.add_fusion(mm.get_model()).to(device)
    for deterministic in (True, False):
        torch.backends.cudnn.deterministic = deterministic
        for mode in ('head', 'full'):
            for p in model.parameters():
                p.requires_grad = mode == 'full'
            if mode == 'head':
                for p in model.head.parameters():
                    p.requires_grad = True
                for n, m in model.named_modules():
                    if n.endswith(('channel_adapter', 'input_adapter')):
                        for p in m.parameters():
                            p.requires_grad = True
            model.train()
            torch.nn.functional.cross_entropy(model(x), y).backward()  # warmup
            sync()
            t = time.time()
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(x), y).backward()
            sync()
            dt = (time.time() - t) / 3
            print(f'{name:18s} deterministic={deterministic!s:5s} {mode:4s} {dt:7.3f}s/step (batch {B})', flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
