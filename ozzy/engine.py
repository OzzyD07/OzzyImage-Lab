import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


class Engine:
    def __init__(self, config_path=None, demo=False):
        self.demo = demo
        self.config = json.loads(Path(config_path).read_text()) if config_path else {}

    def capabilities(self):
        return {"backend": "demo" if self.demo else "stable-diffusion.cpp",
                "model": "Qwen-Image-2.1 Q8_0", "max_references": 10,
                "modes": ["generate", "edit", "mask", "transparent", "extract"],
                "mask_method": "reference annotation + optional pixel compositing",
                "native_inpaint": False, "lora": False, "controlnet": False,
                "config_ready": self.demo or all(Path(self.config.get(k, "__missing__")).is_file()
                                                  for k in ("binary", "denoiser", "vae", "encoder", "vision")),
                "profile": self.config.get("profile", "a100-40"),
                "engine_commit": self.config.get("engine_commit"),
                "demo": self.demo}

    def command(self, request, store, output, seed, jobdir):
        c = self.config
        if not self.capabilities()["config_ready"]:
            raise ValueError("Motor veya model dosyaları eksik. Notebook kurulum hücrelerini çalıştırın.")
        prompt = request.prompt
        if request.mode in {"transparent", "extract"}:
            prompt = "This is an RGBA image with transparency. " + prompt + " The image has alpha channel and the background is transparent."
        refs = [store.path("assets", i) for i in request.references]
        if request.mode == "mask":
            base = Image.open(refs[0]).convert("RGB")
            mask = Image.open(store.path("assets", request.mask)).convert("L")
            overlay = Image.new("RGB", base.size, (255, 40, 70))
            annotated = Image.composite(overlay, base, mask.point(lambda v: int(v * .55)))
            annotated_path = jobdir / "annotated.png"
            annotated.save(annotated_path)
            refs[0] = annotated_path
            prompt = "Edit only the red highlighted region of image 1, remove the red annotation in the result, preserve the rest. " + prompt
        command = [c["binary"], "--diffusion-model", c["denoiser"], "--vae", c["vae"],
                   "--llm", c["encoder"], "--llm_vision", c["vision"],
                   "-p", prompt, "-n", request.negative_prompt,
                   "-W", str(request.width), "-H", str(request.height),
                   "--steps", str(request.steps), "--cfg-scale", str(request.cfg),
                   "--sampling-method", "euler", "--seed", str(seed), "--diffusion-fa",
                   "--model-args", f"qwen_image_2_1_prefix_cache={'true' if request.prefix_cache else 'false'}",
                   "-o", str(output)]
        for ref in refs:
            command += ["--ref-image", str(ref)]
        if request.offload:
            command.append("--offload-to-cpu")
        if request.vae_tiling:
            command.append("--vae-tiling")
        return command

    def run(self, request, store, output, seed, jobdir, cancel, progress):
        if self.demo:
            import time
            for step in range(6):
                if cancel.is_set():
                    raise InterruptedError("İptal edildi")
                time.sleep(.08)
                progress(f"Demo {step + 1}/6")
            image = Image.new("RGB", (request.width, request.height), "#111c25")
            draw = ImageDraw.Draw(image)
            for i in range(12):
                margin = int(min(request.width, request.height) * (.03 + i * .035))
                draw.ellipse((margin, margin, request.width-margin, request.height-margin), outline="#c5ef77", width=3)
            draw.text((40, 40), "DEMO / No model inference", fill="white")
            image.save(output)
            return
        command = self.command(request, store, output, seed, jobdir)
        log_path = jobdir / "engine.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, shell=False)
            try:
                while process.poll() is None:
                    if cancel.wait(.5):
                        process.terminate()
                        raise InterruptedError("İptal edildi")
                    with log_path.open("rb") as reader:
                        reader.seek(max(0, log_path.stat().st_size - 1500))
                        progress(reader.read().decode("utf-8", errors="replace"))
                if process.returncode != 0:
                    tail = log_path.read_bytes()[-3000:].decode("utf-8", errors="replace")
                    raise RuntimeError(f"Motor çıkış kodu {process.returncode}: {tail}")
                if not output.exists():
                    candidates = sorted(output.parent.glob(output.stem + "_*.png"))
                    if len(candidates) != 1:
                        raise RuntimeError("Motor beklenen PNG çıktısını üretmedi.")
                    os.replace(candidates[0], output)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
