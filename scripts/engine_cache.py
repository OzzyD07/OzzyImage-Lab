"""Reuse locally built CUDA executables across compatible Colab sessions."""
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path


def fingerprint(commit):
    return {"commit": commit, "system": platform.system(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "architecture": "80",
            "cuda": subprocess.check_output(['nvcc', '--version'], text=True).strip()}


def cache_key(environment):
    return hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[:24]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(binary):
    help_text = subprocess.check_output([str(binary), '--help'], text=True, stderr=subprocess.STDOUT, timeout=60)
    for flag in ('--llm_vision', '--ref-image', '--model-args', '--diffusion-fa', '--vae-tiling'):
        if flag not in help_text:
            raise RuntimeError(f'Motor gerekli parametreyi desteklemiyor: {flag}')
    devices = subprocess.check_output([str(binary), '--list-devices'], text=True, stderr=subprocess.STDOUT, timeout=60)
    if 'CUDA' not in devices.upper():
        raise RuntimeError('Motor CUDA aygıtını göremiyor: ' + devices[-1000:])


def restore(cache_root, runtime, environment, status):
    folder = cache_root / cache_key(environment)
    if not (folder / 'manifest.json').is_file():
        status('Uyumlu Drive motor önbelleği bulunamadı; ilk derleme yapılacak.')
        return None
    try:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['environment'] != environment:
            raise ValueError('Ortam kaydı eşleşmiyor.')
        if not manifest['files'] or 'sd-cli' not in manifest['files']:
            raise ValueError('Önbellek eksik.')
        for name, sha in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Geçersiz önbellek dosyası.')
            if digest(folder / 'bin' / name) != sha:
                raise ValueError('Motor dosyası SHA256 eşleşmiyor.')
        destination = runtime / ('cached-engine-' + cache_key(environment))
        destination.mkdir(parents=True, exist_ok=True)
        for name in manifest['files']:
            shutil.copy2(folder / 'bin' / name, destination / name)
            (destination / name).chmod(0o755)
        binary = destination / 'sd-cli'
        validate(binary)
        status('Drive motor önbelleği doğrulandı. CUDA derlemesi atlandı.')
        return binary
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        status(f'Önbellek kullanılamadı: {exc}. Kaynaktan derlemeye geçiliyor.')
        return None


def save(cache_root, binary, environment, status):
    cache_root.mkdir(parents=True, exist_ok=True)
    folder = cache_root / cache_key(environment)
    staging = cache_root / (cache_key(environment) + f'.tmp-{time.time_ns()}')
    (staging / 'bin').mkdir(parents=True)
    files = {}
    for source in binary.parent.iterdir():
        if source.is_file():
            target = staging / 'bin' / source.name
            shutil.copy2(source, target)
            files[source.name] = digest(target)
    (staging / 'manifest.json').write_text(json.dumps({'environment': environment, 'files': files}, indent=2))
    if folder.exists():
        folder.rename(cache_root / (folder.name + f'.old-{time.time_ns()}'))
    staging.rename(folder)
    status('Derlenmiş CUDA motoru sonraki oturumlar için Drive’a kaydedildi.')
