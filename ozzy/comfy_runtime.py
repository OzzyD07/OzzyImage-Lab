import json
import socket
import subprocess
import time
from pathlib import Path

import httpx

from .engine import MODEL_FILES

GPU_FLAGS = ['--gpu-only', '--bf16-unet', '--bf16-text-enc', '--bf16-vae',
             '--disable-async-offload', '--disable-dynamic-vram', '--disable-fast-disk']


def require_gpu():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA ve BF16 destekli 80 GB sınıfı GPU gerekli.')
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 75 * 1024**3:
        raise RuntimeError(f'80 GB GPU gerekli. Bulunan: {device.name}, {device.total_memory/1024**3:.1f} GiB. CPU offload veya 40 GB profiline geçilmeyecek.')
    return {'name': device.name, 'memory_gib': round(device.total_memory/1024**3, 2), 'torch': torch.__version__}


def validate_nodes(info):
    required = ['UNETLoader', 'CLIPLoader', 'VAELoader', 'TextEncodeQwenImage21',
                'EmptyLatentImage', 'QwenImage21Cache', 'KSampler', 'VAEDecode',
                'VAEDecodeTiled', 'SaveImage', 'LoadImage', 'JoinImageWithAlpha']
    missing = set(required) - set(info)
    if missing:
        raise RuntimeError('ComfyUI düğümleri eksik: ' + ', '.join(sorted(missing)))
    for node, field, role in [('UNETLoader', 'unet_name', 'denoiser'), ('CLIPLoader', 'clip_name', 'encoder'), ('VAELoader', 'vae_name', 'vae')]:
        choices = info[node]['input']['required'][field][0]
        if MODEL_FILES[role] not in choices:
            raise RuntimeError('ComfyUI BF16 dosyasını göremiyor: ' + MODEL_FILES[role])


def stop_process(process, log=None):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if log is not None:
        log.close()


def start_comfy(config_path):
    require_gpu()
    config = json.loads(Path(config_path).read_text())
    if config.get('gpu_only') is not True or config.get('backend') != 'comfyui':
        raise RuntimeError('Eski motor yapılandırması. Yeni BF16 kurulum hücresini çalıştırın.')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8188)) == 0:
            raise RuntimeError('8188 portu kullanımda. Önce önceki ComfyUI sürecini durdurun.')
    root, runtime = Path(config['comfy_root']), Path(config['runtime'])
    command = [config['python'], '-u', str(root / 'main.py'), '--listen', '127.0.0.1', '--port', '8188',
               '--extra-model-paths-config', config['model_paths'], '--disable-auto-launch',
               '--input-directory', str(runtime / 'comfy-input'), '--output-directory', str(runtime / 'comfy-output'), *GPU_FLAGS]
    log_path = runtime / 'comfy.log'
    log = log_path.open('w', encoding='utf-8')
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    try:
        for attempt in range(180):
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
            try:
                response = httpx.get(config['comfy_url'] + '/object_info', timeout=5)
                if response.is_success:
                    validate_nodes(response.json())
                    print('ComfyUI hazır · BF16 · GPU-only · API 127.0.0.1:8188', flush=True)
                    return process, log
            except httpx.HTTPError:
                pass
            if attempt % 15 == 0:
                print(f'ComfyUI başlatılıyor · {attempt} sn · günlük: {log_path}', flush=True)
            time.sleep(1)
        raise TimeoutError('ComfyUI başlangıç süresi aşıldı; comfy.log dosyasını inceleyin.')
    except BaseException:
        stop_process(process, log)
        raise
