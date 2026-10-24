# ComfyUI API workflow örnekleri

Bu JSON dosyaları `/prompt` API'sinin `prompt` alanına gönderilen node grafiğidir;
ComfyUI editörünün konum ve bağlantı düzeni içeren UI workflow formatı değildir.
BF16 modellerin kurulması ve Qwen-Image-2.1 node'larını içeren sabitlenmiş ComfyUI sürümü gerekir.

- `generate.api.json`: metinden görsel.
- `edit.api.json`: referansla düzenleme.
- `mask.api.json`: işaretlenmiş referansla bölgesel düzenleme.
- `transparent.api.json`: şeffaf görsel isteği.
- `extract.api.json`: referanstaki nesneyi ayıklama isteği.

Referanslı örneklerde `reference-1.png` bir yer tutucudur. Görseli önce ComfyUI'ye
yükleyip LoadImage node'undaki dosya adını yanıtla değiştirin. Seed, prompt, boyut ve
SaveImage önekini kendi işinize göre düzenleyin.

Maskeleme örneği native inpaint conditioning kullanmaz: Studio önce referansın seçili
bölgesini işaretler, model çıktısını aldıktan sonra maskeye göre kaynakla birleştirir.
Bu hazırlama ve birleştirme işlemleri API grafiğinin dışında uygulamada yapılır.

Her gerçek üretimin tam grafiği Drive'da iş klasörüne kaydedilir; galeriden veya
`get_workflow` MCP aracıyla indirilebilir. Grafiği başka bir kuruluma taşırken kullanılan
referans görsellerini de taşımanız gerekir; JSON görsel verilerini içermez.
