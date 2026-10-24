# @title 4 · Q8 indir / CUDA motorunu hazırla — canlı günlük
import queue, threading, signal
VERIFY_WEIGHTS = False # @param {type:"boolean"}
ENGINE_MODE = 'auto' # @param ["auto", "build"]
args = [sys.executable, '-u', REPO/'scripts/setup_colab.py', '--data', DATA, '--runtime', RUNTIME,
        '--profile', PROFILE, '--encoder', ENCODER, '--engine-mode', ENGINE_MODE]
if VERIFY_WEIGHTS:
    args.append('--verify')
RUNTIME.mkdir(parents=True, exist_ok=True)
setup_log = RUNTIME / 'setup.log'
setup_env = dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_DISABLE_PROGRESS_BARS='0',
                 HF_HUB_VERBOSITY='info', TQDM_MININTERVAL='2')
print('Kurulum başlıyor. Canlı çıktı aşağıda; günlük:', setup_log, flush=True)
print('Motor modu:', ENGINE_MODE, '· auto: uyumlu Drive önbelleğini kullan, yoksa derle.', flush=True)
print('Sessiz aşamalarda 15 saniyede bir geçen süre gösterilir; bu mesaj ilerleme garantisi değildir.', flush=True)
started = time.monotonic()
process = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
    bufsize=1, env=setup_env, start_new_session=True)
messages = queue.Queue()
def read_setup_output():
    try:
        for line in process.stdout:
            messages.put(line)
    finally:
        messages.put(None)
reader = threading.Thread(target=read_setup_output, daemon=True)
reader.start()
try:
    with setup_log.open('w', encoding='utf-8') as log:
        while True:
            try:
                line = messages.get(timeout=15)
            except queue.Empty:
                print(f'[Bekleniyor] Geçen süre: {(time.monotonic()-started)/60:.1f} dk · henüz yeni motor/indirme çıktısı yok.', flush=True)
                continue
            if line is None:
                break
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f'Kurulum başarısız (çıkış kodu {returncode}). Yukarıdaki son hata satırlarını inceleyin. Günlük: {setup_log}')
except BaseException:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    raise
finally:
    reader.join(timeout=2)
    process.stdout.close()
print('Tamamlandı. Model revision ve SHA256 kayıtları Drive/model-lock.json içinde.', flush=True)
