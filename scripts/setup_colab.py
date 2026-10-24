"""Prepare ComfyUI and BF16 weights for an 80 GB GPU-only runtime."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import enable_progress_bars

ENGINE_COMMIT = "7cebca52227c63b4180ceed448cbd7ad22903b6d"
MODEL_REPO = "Comfy-Org/Qwen-Image-2.1"
MODEL_FILES = {
    "denoiser": "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "encoder": "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
}

def status(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_progress(path):
    total = path.stat().st_size
    digest = hashlib.sha256()
    done = 0
    last = time.monotonic()
    status(f"SHA256 doğrulaması başlıyor: {path.name}")
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 5 or done == total:
                status(f"SHA256: %{100 * done / max(total, 1):.1f} · {done / 1024**3:.2f}/{total / 1024**3:.2f} GiB")
                last = now
    return digest.hexdigest()


def run(*args, cwd=None):
    status("Komut: " + " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)



def setup(data: Path, runtime: Path, verify: bool):
    from ozzy.comfy_runtime import require_gpu
    enable_progress_bars()
    started = time.monotonic()
    status("Aşama 1/4 · 80 GB GPU, BF16 ve CUDA kontrolü")
    gpu = require_gpu()
    status("GPU: " + json.dumps(gpu))
    data.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    source = runtime / 'ComfyUI'
    status("Aşama 2/4 · Sabitlenmiş ComfyUI ve Python bağımlılıkları; C++ derlemesi yok")
    if not (source / '.git').exists():
        run('git', 'clone', '--no-checkout', 'https://github.com/Comfy-Org/ComfyUI.git', source)
    run('git', 'fetch', '--depth', '1', 'origin', ENGINE_COMMIT, cwd=source)
    run('git', 'checkout', '--detach', ENGINE_COMMIT, cwd=source)
    venv = runtime / 'comfy-venv'
    if not (venv / 'bin' / 'python').exists():
        run(sys.executable, '-m', 'venv', '--system-site-packages', venv)
    python = venv / 'bin' / 'python'
    import importlib.metadata
    constraints = runtime / 'torch-constraints.txt'
    constraints.write_text('\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in ('torch', 'torchvision')))
    run(python, '-m', 'pip', 'install', '-r', source / 'requirements.txt', '-c', constraints)
    run(python, '-c', 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print("ComfyUI CUDA/BF16 hazır:", torch.__version__, flush=True)')
    status("Aşama 3/4 · Quantize edilmemiş BF16 dosyaları → Drive")
    model_root = data / 'models' / 'Qwen-Image-2.1-BF16'
    lock_path = data / 'model-lock-bf16.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {'repo': MODEL_REPO, 'files': {}}
    if lock['repo'] != MODEL_REPO:
        raise RuntimeError('BF16 model kilidi beklenmeyen bir depoya ait.')
    if 'revision' not in lock:
        lock['revision'] = HfApi(token=os.environ.get('HF_TOKEN')).model_info(MODEL_REPO).sha
        lock_path.write_text(json.dumps(lock, indent=2))
    for index, (role, filename) in enumerate(MODEL_FILES.items(), 1):
        status(f'Dosya {index}/3 · {filename} · revision {lock["revision"][:12]}')
        path = Path(hf_hub_download(MODEL_REPO, filename, revision=lock['revision'], local_dir=model_root, token=os.environ.get('HF_TOKEN')))
        old = lock['files'].get(filename, {})
        size = path.stat().st_size
        if verify or not old.get('sha256'):
            sha = sha256_progress(path)
            if old.get('sha256') and sha != old['sha256']:
                raise RuntimeError('SHA256 uyuşmazlığı: ' + filename)
        else:
            status('Dosya mevcut; önceki SHA256 ve boyut kaydı kontrol ediliyor.')
            if old.get('bytes') != size:
                raise RuntimeError('Dosya boyutu değişmiş: ' + filename)
            sha = old['sha256']
        lock['files'][filename] = {'sha256': sha, 'bytes': size}
        temp = lock_path.with_suffix('.tmp')
        temp.write_text(json.dumps(lock, indent=2))
        temp.replace(lock_path)
        status(f'Dosya {index}/3 hazır · {size/1024**3:.2f} GiB')
    status('Aşama 4/4 · ComfyUI model yolları ve GPU-only yapılandırması')
    import yaml
    paths = runtime / 'extra_model_paths.yaml'
    paths.write_text(yaml.safe_dump({'ozzy_bf16': {'base_path': str(model_root), 'diffusion_models': 'diffusion_models', 'text_encoders': 'text_encoders', 'vae': 'vae'}}))
    config = {'backend': 'comfyui', 'profile': 'a100-80', 'gpu_only': True, 'precision': 'bf16',
              'comfy_url': 'http://127.0.0.1:8188', 'engine_commit': ENGINE_COMMIT,
              'model_repo': MODEL_REPO, 'model_revision': lock['revision'],
              'comfy_root': str(source), 'python': str(python), 'model_paths': str(paths), 'runtime': str(runtime)}
    (runtime / 'config.json').write_text(json.dumps(config, indent=2))
    (data / 'runtime-manifest.json').write_text(json.dumps({**config, 'gpu': gpu}, indent=2))
    with (data / 'comfy-python-environment.txt').open('w') as out:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=out, check=True)
    status(f'Kurulum tamamlandı · {(time.monotonic()-started)/60:.1f} dakika. Sonraki hücre ComfyUI motorunu başlatır.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=Path('/content/ozzy-runtime'))
    parser.add_argument('--verify', action='store_true')
    setup(**vars(parser.parse_args()))
