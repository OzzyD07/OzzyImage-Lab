import json
import math
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

MODEL_FILES = {'denoiser': 'qwen_image_2.1_bf16.safetensors',
               'encoder': 'qwen3vl_8b_bf16.safetensors', 'vae': 'qwen_image_2.1_vae_bf16.safetensors'}


def workflow(request, seed, uploaded, prefix):
    prompt = request.prompt
    if request.mode in {'transparent', 'extract'}:
        prompt = 'This is an RGBA format image with transparency. ' + prompt + '. The image has an alpha channel and a transparent background.'
    if request.mode == 'mask':
        prompt = 'Edit only the red highlighted region of image 1, remove the red annotation, preserve the rest. ' + prompt
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    graph = {
        '1': node('UNETLoader', unet_name=MODEL_FILES['denoiser'], weight_dtype='default'),
        '2': node('CLIPLoader', clip_name=MODEL_FILES['encoder'], type='qwen_image', device='default'),
        '3': node('VAELoader', vae_name=MODEL_FILES['vae']),
        '4': node('TextEncodeQwenImage21', clip=['2', 0], prompt=prompt, negative_prompt=request.negative_prompt, resolution=0),
        '5': node('EmptyLatentImage', width=request.width, height=request.height, batch_size=1),
        '6': node('QwenImage21Cache', model=['1', 0], device='gpu' if request.prefix_cache else 'off', dtype='default'),
        '7': node('KSampler', model=['6', 0], positive=['4', 0], negative=['4', 1],
                  latent_image=['4', 2] if uploaded else ['5', 0], seed=seed, steps=request.steps,
                  cfg=request.cfg, sampler_name='euler', scheduler='simple', denoise=1.0),
        '8': node('VAEDecode', samples=['7', 0], vae=['3', 0]),
        '9': node('SaveImage', images=['8', 0], filename_prefix=prefix),
    }
    if request.vae_tiling:
        graph['8'] = node('VAEDecodeTiled', samples=['7', 0], vae=['3', 0], tile_size=512,
                          overlap=64, temporal_size=64, temporal_overlap=8)
    if uploaded:
        graph['4']['inputs']['vae'] = ['3', 0]
    for index, filename in enumerate(uploaded, 1):
        load, rgba = str(100 + index), str(200 + index)
        graph[load] = node('LoadImage', image=filename)
        graph[rgba] = node('JoinImageWithAlpha', image=[load, 0], alpha=[load, 1])
        graph['4']['inputs'][f'images.image_{index}'] = [rgba, 0]
    return graph


class Engine:
    def __init__(self, config_path=None, demo=False, transport=None):
        self.demo = demo
        self.config = json.loads(Path(config_path).read_text()) if config_path else {}
        self.transport = transport

    def client(self):
        return httpx.Client(base_url=self.config.get('comfy_url', 'http://127.0.0.1:8188'), timeout=30, transport=self.transport)

    def capabilities(self):
        ready = self.demo
        if not ready and self.config.get('backend') == 'comfyui' and self.config.get('gpu_only') is True:
            try:
                with self.client() as client:
                    ready = client.get('/system_stats', timeout=2).is_success
            except httpx.HTTPError:
                pass
        return {'backend': 'demo' if self.demo else 'comfyui', 'model': 'Qwen-Image-2.1 BF16',
                'max_references': 10, 'modes': ['generate', 'edit', 'mask', 'transparent', 'extract'],
                'mask_method': 'reference annotation + optional pixel compositing',
                'native_inpaint': False, 'lora': False, 'controlnet': False, 'config_ready': ready,
                'profile': 'a100-80', 'gpu_only': True, 'offload': False, 'precision': 'bf16',
                'engine_commit': self.config.get('engine_commit'), 'model_revision': self.config.get('model_revision'), 'demo': self.demo}

    def prepare_references(self, request, store, jobdir):
        paths = []
        for index, ident in enumerate(request.references):
            with Image.open(store.path('assets', ident)) as source:
                image = source.convert('RGBA')
            if index == 0 and request.mode == 'mask':
                with Image.open(store.path('assets', request.mask)) as selection:
                    mask = selection.convert('L').point(lambda v: int(v * .55))
                    image = Image.composite(Image.new('RGBA', image.size, (255, 40, 70, 255)), image, mask)
            if index == 0:
                size = (request.width, request.height)
            else:
                ratio = image.width / image.height
                area = request.width * request.height
                size = (max(32, round(math.sqrt(area * ratio) / 32) * 32), max(32, round(math.sqrt(area / ratio) / 32) * 32))
            path = jobdir / f'reference-{index}.png'
            image.resize(size, Image.Resampling.LANCZOS).save(path)
            paths.append(path)
        return paths

    def run(self, request, store, output, seed, jobdir, cancel, progress):
        if self.demo:
            for step in range(6):
                if cancel.wait(.08):
                    raise InterruptedError('İptal edildi')
                progress(f'Demo {step + 1}/6')
            image = Image.new('RGB', (request.width, request.height), '#111c25')
            draw = ImageDraw.Draw(image)
            for i in range(12):
                margin = int(min(image.size) * (.03 + i * .035))
                draw.ellipse((margin, margin, image.width-margin, image.height-margin), outline='#c5ef77', width=3)
            draw.text((40, 40), 'DEMO / No model inference', fill='white')
            image.save(output)
            return
        if request.offload:
            raise ValueError('80 GB GPU-only profilinde offload kullanılamaz.')
        ident = str(uuid.uuid4())
        uploaded = []
        with self.client() as client:
            for path in self.prepare_references(request, store, jobdir):
                if cancel.is_set():
                    raise InterruptedError('İptal edildi')
                with path.open('rb') as stream:
                    response = client.post('/upload/image', files={'image': (ident + '-' + path.name, stream, 'image/png')}, data={'type': 'input'})
                response.raise_for_status()
                item = response.json()
                uploaded.append('/'.join(filter(None, [item.get('subfolder'), item['name']])))
            graph = workflow(request, seed, uploaded, 'ozzy/' + output.stem)
            (jobdir / f'workflow-{output.stem}.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
            submitted = False
            finished = False
            try:
                submitted = True
                response = client.post('/prompt', json={'prompt': graph, 'prompt_id': ident, 'client_id': ident})
                if not response.is_success:
                    raise RuntimeError('ComfyUI workflow reddedildi: ' + response.text[-3000:])
                started = time.monotonic()
                while True:
                    if cancel.is_set():
                        raise InterruptedError('İptal edildi')
                    response = client.get('/history/' + ident)
                    response.raise_for_status()
                    history = response.json().get(ident)
                    if history:
                        status = history.get('status', {})
                        if status.get('status_str') == 'error':
                            raise RuntimeError('ComfyUI üretim hatası: ' + json.dumps(status.get('messages', []), ensure_ascii=False)[-3500:])
                        if status.get('completed'):
                            images = history.get('outputs', {}).get('9', {}).get('images', [])
                            if len(images) != 1 or images[0].get('type') != 'output':
                                raise RuntimeError('ComfyUI beklenen PNG çıktısını üretmedi.')
                            result = client.get('/view', params={k: images[0][k] for k in ('filename', 'subfolder', 'type')})
                            result.raise_for_status()
                            output.write_bytes(result.content)
                            with Image.open(output) as image:
                                image.verify()
                            finished = True
                            progress('ComfyUI tamamlandı; sonuç Drive’a kaydediliyor.')
                            return
                    progress(f'ComfyUI · BF16 · GPU-only · geçen süre {time.monotonic()-started:.0f} sn. Ayrıntılı motor günlüğü: comfy.log')
                    if time.monotonic() - started > 7200:
                        raise TimeoutError('ComfyUI işi iki saatlik süre sınırını aştı.')
                    cancel.wait(1)
            finally:
                if submitted and not finished:
                    for endpoint, body in [('/queue', {'delete': [ident]}), ('/interrupt', {'prompt_id': ident})]:
                        try:
                            client.post(endpoint, json=body).raise_for_status()
                        except httpx.HTTPError:
                            progress('ComfyUI iptal isteğine yanıt vermedi; notebook’tan motoru durdurun.')
