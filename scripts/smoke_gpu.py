"""Run actual GPU acceptance checks after the Colab setup cell."""
import argparse
import time

from PIL import Image

from ozzy.engine import Engine
from ozzy.schema import Generation
from ozzy.service import Studio
from ozzy.storage import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    studio = Studio(Store(args.data), Engine(args.config))
    try:
        for mode in ("generate", "edit", "mask", "transparent", "extract"):
            values = {"mode": mode, "prompt": "A red ceramic mug on a clean white table", "seed": 42}
            if mode in {"edit", "mask", "extract"}:
                values["references"] = [reference["id"]]
                values["prompt"] = "Make the mug blue" if mode != "extract" else "Extract only the mug"
            if mode == "mask":
                import io
                from PIL import ImageDraw
                selection = Image.new("L", (1024, 1024))
                ImageDraw.Draw(selection).rectangle((200, 200, 824, 824), fill=255)
                buffer = io.BytesIO()
                selection.save(buffer, format="PNG")
                values["mask"] = studio.store.upload(buffer.getvalue())["id"]
                values["mask_blur"] = 0
            job = studio.submit(Generation(**values))
            while job["status"] in {"queued", "running"}:
                time.sleep(3)
                job = studio.get(job["id"])
            assert job["status"] == "done", job["message"]
            item = job["outputs"][0]
            path = studio.store.path("outputs", item["id"])
            if mode == "generate":
                reference = studio.store.upload(path.read_bytes())
            if mode in {"transparent", "extract"}:
                assert item["has_transparency"], "PNG has no nonopaque alpha; inspect prompt/backend."
            if mode == "mask":
                with Image.open(path) as result, Image.open(studio.store.path("assets", reference["id"])) as source:
                    assert result.convert("RGBA").getpixel((10, 10)) == source.convert("RGBA").getpixel((10, 10))
            print(mode, "PASS", item["id"])
    finally:
        studio.close()


if __name__ == "__main__":
    main()
