# OzzyImage Lab

**Qwen-Image-2.1 BF16 + ComfyUI · 80 GB GPU-only · Colab · Drive · Web Studio · MCP**

Kişisel görsel üretim ve düzenleme stüdyosu. Özel web arayüzü ve MCP aynı kalıcı iş kuyruğunu kullanır;
üretim, arka planda sürekli çalışan ComfyUI motoruna API workflow olarak gönderilir.

## Kurulum

1. Proje dosyalarını GitHub deponuza yükleyin ve `notebooks/OzzyImage_Lab.ipynb` dosyasını Colab'da açın.
2. **80 GB sınıfı NVIDIA GPU** sağlayan çalışma zamanı kullanın. Notebook GPU belleğini ve BF16 desteğini denetler;
   40 GB GPU ile devam etmez. Colab hesabınızın 80 GB GPU tahsis edeceği garanti edilmez.
3. Colab Secrets: `NGROK_AUTHTOKEN`, en az 24 karakterlik rastgele `OZZY_TOKEN`, gerekiyorsa `HF_TOKEN`.
4. Notebook'taki depo URL'sini ve `REVISION` alanını kontrol edin. Kendi commit SHA'nızla sürüm sabitleyebilirsiniz.
5. Sırasıyla 1–4. hücreleri çalıştırın. 4. hücre ComfyUI'yi ve model dosyalarını kurar; **C++/CUDA motor derlemesi yoktur**.
6. **4B** hücresi ComfyUI'yi başlatır ve gereken düğümlerle BF16 dosyalarını doğrular.
7. İlk kurulumda isteğe bağlı **5. hücre GPU kabul testini** çalıştırın.
8. **6. hücre** web arayüzünü ve MCP'yi ngrok üzerinden açar. Linke girip `OZZY_TOKEN` ile oturum açın.
9. **7. hücre** stüdyoyu, ComfyUI'yi ve tüneli durdurur.

## Model ve bellek politikası

Temel model [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).
ComfyUI yükleyicileri için **Comfy-Org tarafından yeniden paketlenmiş BF16 ağırlıklar** kullanılır;
Qwen deposundaki Diffusers shard'ları doğrudan indirilmez. Bu dosyalar GGUF, INT8 veya FP8 değildir.
[Dağıtım açıklaması ve dosyalar](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

| Bileşen | Dosya |
| --- | --- |
| Görüntü modeli | `diffusion_models/qwen_image_2.1_bf16.safetensors` |
| Metin/görsel kodlayıcı | `text_encoders/qwen3vl_8b_bf16.safetensors` |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` |

Drive yolu: `MyDrive/OzzyImage-Lab/models/Qwen-Image-2.1-BF16/`.
En az 40 GB boş Drive alanı ayırın, galeri için ayrıca alan bırakın. Önceki GGUF indirmeleri silinmez;
bu yüzden geçişte her iki model grubunun disk kullanımını hesaba katın.

ComfyUI şu seçeneklerle çalışır:

```text
--gpu-only --bf16-unet --bf16-text-enc --bf16-vae
--disable-async-offload --disable-dynamic-vram --disable-fast-disk
```

Prefix KV cache `device=gpu`, `dtype=default` kullanır; istenirse tamamen kapatılabilir.
CPU offload UI'da sunulmaz; API/MCP `offload=true` isteğini reddeder. Yükleme sırasında dosya okuma
ve normal CPU işlemleri vardır; GPU-only, model bileşenlerini çıkarım sırasında CPU'ya taşıyan offload'u kapatır.
Bellek taşarsa iş hata verir, otomatik quantization veya CPU fallback yapılmaz. VAE tiling ayrı bir seçenektir;
GPU üzerinde bloklar halinde decode yapar, CPU offload değildir.

Başlangıç: **1024×1024, 40 adım, CFG 1, Euler / simple, batch 1**.
2K presetler arayüzde mevcuttur; varsayılan çözünürlük GPU ölçümü yapılana kadar 1024'tür.
1–8 varyasyon aynı anda değil sırayla üretilir. CFG 1'de negatif prompt etkisizdir.
80 GB sınırsız değildir; çok sayıda referans ve 4K boyutlar bellek sınırına ulaşabilir.

## Özellikler

- Metinden üretim, referanslı düzenleme, RGBA üretimi, konu ayıklama.
- En fazla 10 sıralı referans; şeffaf referansların alpha kanalı ComfyUI'ye aktarılır.
- Fırça, silgi, geri alma, PNG maske yükleme, kenar yumuşatma, maske dışını koruma.
- Seed, CFG, adım, boyut, sıralı varyasyon, GPU prefix cache ve VAE tiling.
- Kalıcı galeri, PNG + JSON tarifler, çıktıdan yeni referans oluşturma.
- ComfyUI API workflow'unu galeri veya MCP üzerinden indirme.
- Tek GPU iş kuyruğu, hedef işe özel iptal, hata kayıtları ve geçmiş.
- Drive'da kaynak checkpoint'leri, model revision ve SHA256 kilidi.

**Referans boyutları:** İlk referans seçilen çıktı genişlik/yüksekliğine ölçeklenir;
en-boy oranını kaynağa uygun seçin. Diğer referanslar yaklaşık aynı piksel alanına, oranları korunarak ölçeklenir.
Maske dışını koruma açıksa nihai birleşik çıktı özgün ilk referans boyutundadır.

**Maskeleme yöntemi:** Seçilen alan kırmızı işaretli referans olarak modele gönderilir.
`preserve_unmasked=true` sonucunda beyaz maskeli alan model çıktısından, siyah alan özgün görselden alınır.
Bu, native mask-conditioned inpainting değildir. `mask_blur=0` maske dışı pikselleri birebir korur;
yumuşatma sınır çevresindeki pikselleri de harmanlar.

Şeffaf üretim/ayıklama, modelin RGBA prompt biçimini kullanır. Galeri, saydam pikseller varsa RGBA etiketi gösterir.
LoRA/ControlNet veya başka model ailesi workflow'ları bu stüdyoda henüz açılmamıştır;
ComfyUI'ye geçiş bu özelliklerin otomatik olarak eklendiği anlamına gelmez.

## Canlı günlük ve doğrulama

4. hücre dosya adını, indirme/önbellek durumunu, SHA256 yüzdesini ve kurulum aşamasını canlı gösterir.
Çıktı olmayan dönemlerde 15 saniyede bir süre bildirir; bu bir ilerleme garantisi değildir.
`/content/ozzy-runtime/setup.log` kurulum, `comfy.log` motor, `server.log` stüdyo günlüğüdür.
Web arayüzü üretim boyunca geçen süreyi, tamamlanmayı ve ComfyUI hatalarını gösterir; adım yüzdesi göstermez.

`model-lock-bf16.json` ilk indirmede kaynak revision'ı sabitler. Dosyalar yeniden kullanılır;
yarım indirmeler Hugging Face istemcisi tarafından devam ettirilebilir. `VERIFY_WEIGHTS` yeniden SHA256 hesaplar.
ComfyUI commit'i: `7cebca52227c63b4180ceed448cbd7ad22903b6d`.
Python bağımlılıklarının kurulu sürümleri Drive'daki `comfy-python-environment.txt` dosyasına kaydedilir.
ComfyUI ayrı venv'de Colab'ın CUDA PyTorch paketlerini devralır; torch/torchvision sürümleri kurulumda korunur.

**Doğrulama sınırı:** CPU testleri, sahte ComfyUI HTTP yanıtları ve demo motoruyla uygulanmıştır.
Gerçek 80 GB GPU üzerinde BF16 yükleme, hız, bellek sınırı ve görüntü kalitesi henüz doğrulanmamıştır.
5. hücre üretim, referans, maske, RGBA ve ayıklamayı gerçek modelle çalıştırır ve GPU bellek kullanımını yazdırır.

## MCP ve API

Adres: `https://<ngrok-host>/mcp/` — Streamable HTTP.
Her istekte `Authorization: Bearer <OZZY_TOKEN>` başlığı kullanılır.

13 araç: `capabilities`, `upload_image`, `generate_image`, `get_job`, `list_jobs`, `cancel_job`,
`get_image`, `get_recipe`, `get_workflow`, `reuse_output`, `save_preset`, `list_presets`, `delete_preset`.

```json
{"settings":{"mode":"generate","prompt":"A ceramic mug, soft daylight","width":1024,"height":1024,"steps":40,"cfg":1,"seed":42,"offload":false}}
```

`upload_image` data-URL öneki olmayan base64 alır. Dönen kimlikler `references` ve `mask` alanlarında kullanılır.
`generate_image` hemen iş kimliği döndürür; `get_job` ile takip edin. `get_workflow` gerçek üretimin
ComfyUI API JSON grafiğini döndürür; demo ve eski sd-cli çıktıları için mevcut değildir.
Web API karşılığı `/api/outputs/{id}/workflow`.
Örnek istemci: `examples/mcp_client.py`. OAuth zorunlu istemciler bu sürümde desteklenmez.

ComfyUI yalnızca `127.0.0.1:8188` üzerinde dinler; ham ComfyUI API'si ngrok üzerinden açılmaz.
Dış erişim, mevcut anahtarlı özel stüdyo ve MCP üzerinden yapılır. ComfyUI düğüm ekranının ayrıca
internete açılması bu kurulumun parçası değildir. Tek Uvicorn worker kullanın.

## Önceki sürümden geçiş ve güncelleme

1. Eski kurulum/üretimi durdurun. Çalışan sd-cli işlemi varsa onu da sonlandırın.
2. Güncel kaynakları GitHub'a yükleyin; yeni notebook'u açın veya güncelleme hücresini kullanın.
3. **3, 4, 4B, 6** hücrelerini yeniden çalıştırın. Yalnızca notebook'u değiştirmek yeterli değildir.
4. Mevcut Drive galerinize ve referanslarınıza erişebilirsiniz. Eski GGUF dosyaları ve önbellekler korunur fakat kullanılmaz.
5. Eski tarif UI'dan yüklenebilir; üretim artık BF16/ComfyUI olduğundan aynı seed ile aynı sonuç beklenmemelidir.
   Eski doğrudan API tariflerindeki `offload=true` alanını kaldırın veya false yapın; başlangıç için CFG 1/40 adım kullanın.

8. hücre `check` ile uzak sürümü karşılaştırır, `update` ile mevcut kaynak kodunu Drive'a Git bundle olarak
kaydedip hedef sürüme geçer, `rollback` ile bundle SHA'sına döner. Yerel değişiklik varsa işlem durur.
Her iki sunucu da önce durdurulmalıdır. Kaynak checkpoint'i VM/VRAM snapshot'ı değildir.
Oturum kesilince yarım diffusion devam etmez; iş `interrupted` olur, tamamlanmış çıktılar korunur.

## Geliştirme

```bash
pip install -e '.[test]'
pytest -q
```

GPU olmadan arayüz testi için `OZZY_DEMO=1`, `OZZY_TOKEN=<uzun-test-anahtarı>` ortam değişkenlerini ayarlayıp:

```bash
python -m uvicorn ozzy.app:create_app --factory --host 127.0.0.1 --port 8000
```

Demo çıktıları model üretimi değildir. Gerçek modda `OZZY_DEMO` kaldırılır ve `OZZY_CONFIG`/`OZZY_DATA` ayarlanır.
`ozzy/engine.py` workflow adaptörü, `ozzy/comfy_runtime.py` GPU-only başlangıcı,
`scripts/setup_colab.py` indirme ve ComfyUI kurulumu, `scripts/make_notebook.py` notebook kaynağıdır.

Beş işlem için örnek API grafikleri ve kullanım notları [workflows](workflows/README.md) klasöründedir.

Kod MIT lisanslıdır. Modelin Qwen Research lisansı ve ComfyUI'nin kendi lisansı ayrıdır;
model ağırlıkları ve ComfyUI kaynakları bu projeyle dağıtılmaz.
