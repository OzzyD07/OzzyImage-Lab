"""Build the distributable Colab notebook without execution outputs."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.strip().splitlines(keepends=True), "execution_count": None, "outputs": []})


md("""
# OzzyImage Lab
### Qwen-Image-2.1 Q8 · Drive · Web Studio · MCP
Hücreleri sırayla çalıştırın. **Runtime → Change runtime type → A100** seçin.
Model ağırlıkları, referanslar, maskeler ve çıktılar Drive'da; derleme ve kod `/content` içinde kalır.

- Model: Q8_0 denoiser + Qwen3-VL Q8_0 encoder + vision projector + BF16 VAE.
- Başlangıç: 1024×1024, 20 adım, CFG 6, Euler, tek görüntü, CPU offload ve VAE tiling.
- 40 GB başlangıç profilidir; 2K/çoklu referans bellek kullanımı gerçek çalışmada ölçülmelidir.
- Bu notebook çalışma süresini uzatmak için Colab kısıtlarını aşmaz; oturum kapanınca ngrok da kapanır.
- Kod lisansı MIT; model ağırlıkları ayrı **Qwen Research** lisansına tabidir.

Colab Secrets bölümüne `NGROK_AUTHTOKEN` ve en az 24 karakterlik `OZZY_TOKEN` ekleyin.
Gerekirse `HF_TOKEN` ekleyin. Bu anahtarları hücrelere veya GitHub'a yazmayın.
""")
code('''
# @title 1 · Drive ve proje ayarları
from google.colab import drive, userdata
from pathlib import Path
import os, sys, subprocess, json, time, re
drive.mount('/content/drive')
REPO_URL = "https://github.com/YOUR_USERNAME/OzzyImage-Lab.git" # @param {type:"string"}
REVISION = "main" # @param {type:"string"}
PROFILE = "a100-40" # @param ["a100-40", "a100-80"]
ENCODER = "Q8_0" # @param ["Q8_0", "UD-Q4_K_XL"]
DATA = Path('/content/drive/MyDrive/OzzyImage-Lab')
REPO = Path('/content/OzzyImage-Lab')
RUNTIME = Path('/content/ozzy-runtime')
DATA.mkdir(parents=True, exist_ok=True)
def run(*args, **kwargs):
    return subprocess.run(list(map(str,args)), check=True, **kwargs)
def secret(name, required=True):
    try:
        return userdata.get(name)
    except Exception:
        if required:
            from getpass import getpass
            return getpass(name + ': ')
        return ''
os.environ['OZZY_TOKEN'] = secret('OZZY_TOKEN')
assert len(os.environ['OZZY_TOKEN']) >= 24, 'En az 24 karakterli erişim anahtarı gerekli.'
hf_token = secret('HF_TOKEN', required=False)
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
print('Drive hazır. Anahtarlar yalnızca oturum ortamında tutuluyor.')
''')
code('''
# @title 2 · Kaynak kodu al / kontrol et
assert 'YOUR_USERNAME' not in REPO_URL, 'Önce kendi GitHub depo URL adresinizi girin.'
assert re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\\.git)?', REPO_URL)
assert re.fullmatch(r'[A-Za-z0-9_./-]+', REVISION) and not REVISION.startswith('-')
if not (REPO / '.git').exists():
    run('git', 'clone', REPO_URL, REPO)
    run('git', '-C', REPO, 'fetch', 'origin', REVISION)
    run('git', '-C', REPO, 'checkout', '--detach', 'FETCH_HEAD')
else:
    run(sys.executable, REPO/'scripts/checkpoint.py', 'check', '--repo', REPO, '--data', DATA, '--revision', REVISION)
print('Kaynak sürümü:', subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip())
''')
code('''
# @title 3 · Python ve derleme bağımlılıkları
run('apt-get', 'update', '-qq')
run('apt-get', 'install', '-y', '-qq', 'build-essential', 'cmake', 'git', 'pkg-config')
run(sys.executable, '-m', 'pip', 'install', '-q', '-e', str(REPO)+'[test]')
run(sys.executable, '-m', 'pytest', '-q', cwd=REPO)
''')
code('''
# @title 4 · Q8 dosyalarını Drive'a indir ve CUDA motorunu derle
VERIFY_WEIGHTS = False # @param {type:"boolean"}
args = [sys.executable, REPO/'scripts/setup_colab.py', '--data', DATA, '--runtime', RUNTIME,
        '--profile', PROFILE, '--encoder', ENCODER]
if VERIFY_WEIGHTS:
    args.append('--verify')
run(*args)
print('İndirmeler devam ettirilebilir. Model revision ve SHA256 kayıtları model-lock.json içinde.')
''')
md("""
### 5 · Gerçek GPU kabul testi (ilk kurulumda önerilir)
Beş gerçek üretim çalıştırır; süre ve GPU kullanımı gerektirir. Üretim, referans, maske,
şeffaflık ve ayıklama yollarını sınar. Özellikle RGBA çıktıyı görsel olarak da inceleyin.
GPU testi bu proje hazırlanırken çalıştırılmamıştır; yerel testler motoru taklit ederek uygulama akışını sınar.
""")
code('''
RUN_GPU_ACCEPTANCE = False # @param {type:"boolean"}
if RUN_GPU_ACCEPTANCE:
    run(sys.executable, REPO/'scripts/smoke_gpu.py', '--config', RUNTIME/'config.json', '--data', DATA)
else:
    print('GPU kabul testi atlandı.')
''')
code('''
# @title 6 · Web arayüzü ve MCP sunucusunu aç
from pyngrok import ngrok
import requests
if 'server' in globals() and server.poll() is None:
    raise RuntimeError('Sunucu zaten çalışıyor. Önce durdurma hücresini çalıştırın.')
ngrok.set_auth_token(secret('NGROK_AUTHTOKEN'))
tunnel = ngrok.connect(addr=8000, proto='http', bind_tls=True)
public_url = tunnel.public_url
env = dict(os.environ, OZZY_DATA=str(DATA), OZZY_CONFIG=str(RUNTIME/'config.json'),
           OZZY_PUBLIC_HOST=public_url.split('://',1)[1])
env.pop('OZZY_DEMO', None)
RUNTIME.mkdir(parents=True, exist_ok=True)
server_log = (RUNTIME/'server.log').open('w')
server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'ozzy.app:create_app', '--factory',
    '--host', '127.0.0.1', '--port', '8000', '--proxy-headers', '--forwarded-allow-ips', '127.0.0.1'],
    cwd=REPO, env=env, stdout=server_log, stderr=subprocess.STDOUT)
try:
    for _ in range(90):
        if server.poll() is not None:
            raise RuntimeError((RUNTIME/'server.log').read_text()[-5000:])
        try:
            if requests.get('http://127.0.0.1:8000/health',timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise TimeoutError('Sunucu başlatılamadı; server.log dosyasını inceleyin.')
except Exception:
    server.terminate()
    server.wait(timeout=20)
    server_log.close()
    ngrok.disconnect(public_url)
    raise
print('Web Studio:', public_url)
print('MCP Streamable HTTP:', public_url + '/mcp/')
print('MCP başlığı: Authorization: Bearer <OZZY_TOKEN>')
print('Anahtarı arayüze girin. URL içinde anahtar bulunmaz.')
''')
code('''
# @title 7 · Sunucuyu ve tüneli durdur
if 'server' in globals() and server.poll() is None:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
if 'server_log' in globals():
    server_log.close()
if 'public_url' in globals():
    ngrok.disconnect(public_url)
print('Sunucu kapandı. Drive dosyaları korunuyor.')
''')
md("""
### Sürüm kontrol noktası
Önce sunucuyu durdurun. `check` yalnızca karşılaştırır. `update`, mevcut kaynak sürümünü
Drive'da Git bundle olarak saklar ve seçili revision'a geçer. Yerel değişiklik varsa işlem durur.
`rollback` için `checkpoints/` içindeki bundle dosyasının 40 karakterli SHA adını kullanın.
Model/çıktılar güncelleme sırasında değiştirilmez. Bu **kaynak kodu checkpoint'idir**;
yarım kalan diffusion adımları devam ettirilmez. Eski iş kayıtları `interrupted` olur.
Güncelleme/geri dönüşten sonra 3, 4 ve 6. hücreleri yeniden çalıştırın.
""")
code('''
# @title 8 · Güncelleme / geri dönüş (isteğe bağlı)
ACTION = 'check' # @param ["check", "update", "rollback"]
TARGET_REVISION = 'main' # @param {type:"string"}
assert 'server' not in globals() or server.poll() is not None, 'Önce sunucuyu durdurun.'
run(sys.executable, REPO/'scripts/checkpoint.py', ACTION, '--repo', REPO, '--data', DATA, '--revision', TARGET_REVISION)
''')
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"colab": {"name": "OzzyImage_Lab.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "accelerator": "GPU"}, "cells": cells}
for index, cell in enumerate(cells):
    cell['id'] = f'ozzy-{index:02d}'
Path('notebooks').mkdir(exist_ok=True)
Path('notebooks/OzzyImage_Lab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
