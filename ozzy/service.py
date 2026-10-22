import copy
import queue
import secrets
import threading
import time
import uuid

from PIL import Image

from .schema import Generation
from .storage import atomic_json, composite_mask


class Studio:
    def __init__(self, store, engine):
        self.store, self.engine = store, engine
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=16)
        self.jobs = {j["id"]: j for j in store.records("jobs", 10000)}
        self.cancels = {}
        self.closed = False
        for job in self.jobs.values():
            if job["status"] in {"queued", "running"}:
                job.update(status="interrupted", message="Oturum kapandı. Tarif ile yeniden başlatın.")
                self.save(job)
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()

    def save(self, job):
        atomic_json(self.store.path("jobs", job["id"], ".json"), job)

    def submit(self, request: Generation):
        for ident in request.references + ([request.mask] if request.mask else []):
            if not self.store.path("assets", ident).is_file():
                raise ValueError("Referans veya maske bulunamadı.")
        if request.mode == "mask":
            with Image.open(self.store.path("assets", request.references[0])) as base, Image.open(self.store.path("assets", request.mask)) as mask:
                if base.size != mask.size:
                    raise ValueError("Maske ile ilk referans aynı boyutta olmalı.")
                if mask.convert("L").getextrema()[1] == 0:
                    raise ValueError("Maske boş. Düzenlenecek bölgeyi beyazla işaretleyin.")
        if not self.engine.capabilities()["config_ready"]:
            raise ValueError("Model hazır değil. Notebook kurulumunu tamamlayın.")
        with self.lock:
            if self.closed or self.pending.full():
                raise ValueError("İş kuyruğu şu anda dolu veya kapalı.")
            ident = uuid.uuid4().hex
            job = {"id": ident, "status": "queued", "created": time.time(),
                   "request": request.model_dump(), "outputs": [], "message": "Sırada", "completed": 0,
                   "seed": secrets.randbelow(2147483640) if request.seed == -1 else request.seed,
                   "backend": self.engine.capabilities()}
            self.jobs[ident] = job
            self.cancels[ident] = threading.Event()
            self.save(job)
            self.pending.put_nowait(ident)
            return copy.deepcopy(job)

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs:
                raise ValueError("İş bulunamadı.")
            return copy.deepcopy(self.jobs[ident])

    def list_jobs(self, limit=100):
        with self.lock:
            return copy.deepcopy(sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)[:max(1, min(limit, 100))])

    def cancel(self, ident):
        with self.lock:
            job = self.get(ident)
            if job["status"] in {"queued", "running"}:
                self.cancels[ident].set()
                self.jobs[ident]["message"] = "İptal isteniyor"
            return self.get(ident)

    def update(self, ident, **values):
        with self.lock:
            self.jobs[ident].update(values)
            self.save(self.jobs[ident])

    def work(self):
        while True:
            ident = self.pending.get()
            if ident is None:
                self.pending.task_done()
                return
            job = self.get(ident)
            cancel = self.cancels[ident]
            try:
                if cancel.is_set():
                    raise InterruptedError()
                self.update(ident, status="running", message="Model yükleniyor", started=time.time())
                request = Generation.model_validate(job["request"])
                directory = self.store.root / "jobs" / ident
                directory.mkdir(exist_ok=True)
                for index in range(request.count):
                    if cancel.is_set():
                        raise InterruptedError()
                    output_id = uuid.uuid4().hex
                    output = self.store.path("outputs", output_id)
                    seed = (job["seed"] + index) % 2147483648
                    def progress(message):
                        with self.lock:
                            self.jobs[ident]["message"] = message
                    self.engine.run(request, self.store, output, seed, directory, cancel, progress)
                    if cancel.is_set():
                        output.unlink(missing_ok=True)
                        raise InterruptedError()
                    if request.mode == "mask" and request.preserve_unmasked:
                        composite_mask(self.store.path("assets", request.references[0]), output,
                                       self.store.path("assets", request.mask), request.mask_blur)
                    with Image.open(output) as image:
                        has_alpha = "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255
                        dimensions = image.size
                    item = {"id": output_id, "url": f"/api/outputs/{output_id}", "seed": seed,
                            "width": dimensions[0], "height": dimensions[1], "has_transparency": has_alpha}
                    atomic_json(self.store.path("outputs", output_id, ".json"),
                                {**item, "job_id": ident, "request": job["request"], "backend": job["backend"]})
                    with self.lock:
                        self.jobs[ident]["outputs"].append(item)
                    self.update(ident, completed=index + 1)
                self.update(ident, status="done", message="Tamamlandı", finished=time.time())
            except InterruptedError:
                self.update(ident, status="cancelled", message="İptal edildi", finished=time.time())
            except Exception as exc:
                self.update(ident, status="failed", message=str(exc)[-4000:], finished=time.time())
            finally:
                self.pending.task_done()

    def close(self):
        with self.lock:
            self.closed = True
            for event in self.cancels.values():
                event.set()
        self.pending.put(None)
        self.worker.join(timeout=20)

    def save_preset(self, name, settings):
        if not 1 <= len(name) <= 80:
            raise ValueError("Tarif adı 1–80 karakter olmalı.")
        value = {"id": uuid.uuid4().hex, "name": name, "request": Generation.model_validate(settings).model_dump()}
        with self.lock:
            atomic_json(self.store.path("presets", value["id"], ".json"), value)
        return value
