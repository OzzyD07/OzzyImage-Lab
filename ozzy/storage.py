import base64
import io
import json
import os
import re
import uuid
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 25_000_000
ID = re.compile(r"^[0-9a-f]{32}$")


def atomic_json(path: Path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ("assets", "jobs", "outputs", "presets"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def path(self, category, ident, suffix=".png"):
        if category not in {"assets", "jobs", "outputs", "presets"} or not ID.fullmatch(ident):
            raise ValueError("Geçersiz dosya kimliği.")
        return self.root / category / (ident + suffix)

    def upload(self, data: bytes):
        if len(data) > 20 * 1024 * 1024:
            raise ValueError("Dosya en fazla 20 MB olabilir.")
        try:
            opened = Image.open(io.BytesIO(data))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError("Geçerli bir PNG, JPEG veya WebP görseli yükleyin.") from exc
        with opened as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Yalnızca PNG, JPEG ve WebP desteklenir.")
            if image.width * image.height > 25_000_000:
                raise ValueError("Görsel en fazla 25 megapiksel olabilir.")
            image = ImageOps.exif_transpose(image)
            image.load()
            if image.width < 32 or image.height < 32:
                raise ValueError("Görsel en az 32 × 32 olmalı.")
            ident = uuid.uuid4().hex
            image.convert("RGBA" if "A" in image.getbands() else "RGB").save(self.path("assets", ident))
            return {"id": ident, "width": image.width, "height": image.height,
                    "url": f"/api/assets/{ident}"}

    def upload_base64(self, data):
        if len(data) > 28_000_000:
            raise ValueError("Dosya çok büyük.")
        return self.upload(base64.b64decode(data, validate=True))

    def read(self, category, ident):
        return json.loads(self.path(category, ident, ".json").read_text(encoding="utf-8"))

    def records(self, category, limit=100):
        files = sorted((self.root / category).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [json.loads(p.read_text(encoding="utf-8")) for p in files[:limit]]


def composite_mask(base: Path, generated: Path, mask: Path, blur: int):
    with Image.open(base) as source, Image.open(generated) as result, Image.open(mask) as selection:
        source = source.convert("RGBA")
        result = result.convert("RGBA").resize(source.size, Image.Resampling.LANCZOS)
        selection = selection.convert("L")
        if selection.size != source.size:
            raise ValueError("Maske ve kaynak görsel boyutları aynı olmalı.")
        if blur:
            selection = selection.filter(ImageFilter.GaussianBlur(blur))
        Image.composite(result, source, selection).save(generated)
