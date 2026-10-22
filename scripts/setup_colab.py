"""Download pinned weights to Drive and build the CUDA inference engine locally."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ENGINE_COMMIT = "fd21ecfceef2665d2c5f2db8536e3202d864fd30"


def run(*args, cwd=None):
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def setup(data: Path, runtime: Path, profile: str, encoder: str, verify: bool):
    data.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
    print("GPU:", gpu.strip())
    memory_mb = int(gpu.split(",")[1].strip().split()[0])
    if memory_mb < 35000:
        raise RuntimeError("Bu notebook en az 35 GB VRAM olan GPU profili için hazırlanmıştır.")
    if profile == "a100-80" and memory_mb < 70000:
        raise RuntimeError("80 GB profili için GPU belleği yetersiz; a100-40 seçin.")
    if shutil.disk_usage(runtime).free < 12 * 1024**3:
        raise RuntimeError("Derleme için yerel diskte en az 12 GB boş alan gerekli.")
    manifest_path = data / "model-lock.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    specs = {
        "denoiser": ("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q8_0.gguf"),
        "vae": ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors"),
        "encoder": ("unsloth/Qwen3-VL-8B-Instruct-GGUF", f"Qwen3-VL-8B-Instruct-{encoder}.gguf"),
        "vision": ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "mmproj-F16.gguf"),
    }
    config = {"profile": profile, "engine_commit": ENGINE_COMMIT}
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revisions = {}
    for role, (repo, filename) in specs.items():
        old = manifest["models"].get(role, {})
        if old.get("repo") == repo and old.get("filename") == filename:
            revision = old["revision"]
        else:
            if repo not in revisions:
                revisions[repo] = api.model_info(repo).sha
            revision = revisions[repo]
        path = Path(hf_hub_download(repo, filename, revision=revision,
                    local_dir=data / "models" / repo.split("/")[1], token=os.environ.get("HF_TOKEN")))
        record = {"repo": repo, "filename": filename, "revision": revision, "bytes": path.stat().st_size}
        if verify or "sha256" not in old or old.get("revision") != revision or old.get("filename") != filename:
            print("SHA256:", filename)
            with path.open("rb") as stream:
                record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            if old.get("revision") == revision and old.get("filename") == filename and old.get("sha256") and old["sha256"] != record["sha256"]:
                raise RuntimeError(f"Dosya bütünlük kontrolü başarısız: {filename}")
        else:
            record["sha256"] = old["sha256"]
            if record["bytes"] != old["bytes"]:
                raise RuntimeError(f"Dosya boyutu değişmiş: {filename}; doğrulama ile tekrar çalıştırın.")
        manifest["models"][role] = record
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(manifest_path)
        config[role] = str(path.resolve())
    source = runtime / "stable-diffusion.cpp"
    if not (source / ".git").exists():
        run("git", "clone", "--no-checkout", "https://github.com/leejet/stable-diffusion.cpp.git", source)
    run("git", "fetch", "origin", ENGINE_COMMIT, cwd=source)
    run("git", "checkout", "--detach", ENGINE_COMMIT, cwd=source)
    run("git", "submodule", "update", "--init", "--recursive", "--depth", "1", cwd=source)
    run("cmake", "-S", source, "-B", source / "build", "-DSD_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=80", "-DCMAKE_BUILD_TYPE=Release")
    run("cmake", "--build", source / "build", "--config", "Release", "--target", "sd-cli", "-j", "2")
    binary = source / "build" / "bin" / "sd-cli"
    help_text = subprocess.check_output([str(binary), "--help"], text=True)
    for flag in ("--llm_vision", "--ref-image", "--model-args", "--diffusion-fa", "--vae-tiling"):
        if flag not in help_text:
            raise RuntimeError(f"Motor gerekli parametreyi desteklemiyor: {flag}")
    config["binary"] = str(binary)
    config_path = runtime / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    (data / "runtime-manifest.json").write_text(json.dumps({"engine_commit": ENGINE_COMMIT, "profile": profile, "gpu": gpu.strip()}, indent=2))
    with (data / "python-environment.txt").open("w") as out:
        subprocess.run([os.sys.executable, "-m", "pip", "freeze"], stdout=out, check=True)
    print("Hazır:", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/content/ozzy-runtime"))
    parser.add_argument("--profile", choices=["a100-40", "a100-80"], default="a100-40")
    parser.add_argument("--encoder", choices=["Q8_0", "UD-Q4_K_XL"], default="Q8_0")
    parser.add_argument("--verify", action="store_true")
    setup(**vars(parser.parse_args()))
