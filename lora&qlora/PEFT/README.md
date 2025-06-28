-----
# Model Seçim Aşaması
https://llmselector.vercel.app sitesi ve Translation Leaderboard üzerinde en iyi 7B parametreli model seçilmiştir.

# `unified_training.py`

Bu script, LoRA, QLoRA, Prefix Tuning ve Soft Prompting dahil olmak üzere çeşitli **Parameter-Efficient Fine-Tuning (PEFT)** tekniklerini kullanarak büyük dil modellerini **fine-tune** etmek için birleşik bir framework sağlar. Eğitim sürecini kolaylaştırmak için `transformers` ve `peft` kütüphanelerinden yararlanır.

## Temel Özellikler:

  - **Esnek Model Yükleme**: <mcsymbol name="Config.QUANT_4BIT" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol> ayarına bağlı olarak 4-bit **quantization** (QLoRA için) ile veya **quantization** olmadan **base model**'lerin yüklenmesini destekler.
  - **Çoklu PEFT Teknikleri**: <mcsymbol name="Config.TRAINING_TECHNIQUE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'e bağlı olarak LoRA, QLoRA, Prefix Tuning veya Soft Prompting konfigürasyonlarını dinamik olarak uygular.
  - **Veri Hazırlama**: **Dataset**'leri yüklemek ve **tokenize** etmek için <mcfile name="data_preparation.py" path="data_preparation.py"></mcfile> ile entegre olur.
  - **Eğitim Döngüsü**: AdamW **optimizer**, **learning rate scheduler** ve ilerleme takibi ile standart bir eğitim döngüsü uygular.
  - **Değerlendirme**: **Loss** ve BLEU **score**'u hesaplayan bir değerlendirme döngüsü içerir.
  - **Model Kaydetme**: **Fine-tune** edilmiş modeli belirtilen bir **output directory**'ye kaydeder.

## Fonksiyonlar:

### <mcsymbol name="get_tokenized_dataset_and_tokenizer" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

<mcfile name="data_preparation.py" path="data_preparation.py"></mcfile> dosyasındaki <mcsymbol name="load_and_prepare_dataset" filename="data_preparation.py" path="data_preparation.py" startline="1" type="function"></mcsymbol> fonksiyonunu kullanarak **dataset**'i yükler ve hazırlar. **Tokenized dataset**'i ve **tokenizer**'ı döndürür.

### <mcsymbol name="load_base_model" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

Önceden eğitilmiş **causal language model**'i <mcsymbol name="Config.MODEL_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'den yükler. <mcsymbol name="Config.QUANT_4BIT" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol> etkinse 4-bit **quantization**'ı yönetir. Prefix Tuning ve Soft Prompting için, gerekirse **token embedding**'lerini yeniden boyutlandırır.

### <mcsymbol name="get_peft_config" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

<mcsymbol name="Config.TRAINING_TECHNIQUE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'e göre uygun **PEFT configuration**'ını (LoraConfig, PrefixTuningConfig veya PromptTuningConfig) döndürür.

### <mcsymbol name="train_loop" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

Eğitim sürecini yönetir. **Epoch**'lar boyunca yineleme yapar, **forward** ve **backward pass**'ler yapar, model parametrelerini günceller ve eğitim **loss**'u ile **perplexity**'yi hesaplar. Ayrıca periyodik değerlendirme için <mcsymbol name="evaluate_loop" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>'u çağırır.

### <mcsymbol name="evaluate_loop" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

**Evaluation dataset**'inde modeli değerlendirir. Tahminleri ve referansları **decode** ederek toplam **loss**'u ve BLEU **score**'u hesaplar.

### <mcsymbol name="train_model" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>

Tüm eğitim iş akışını düzenler:

1.  **Tokenized dataset**'i ve **tokenizer**'ı alır.
2.  **Base model**'i yükler.
3.  LoRA/QLoRA kullanılıyorsa modeli **k-bit training** için hazırlar.
4.  **PEFT configuration**'ını uygular.
5.  Eğitim ve değerlendirme için `DataLoader`'ları ayarlar.
6.  Eğitimi başlatmak için <mcsymbol name="train_loop" filename="unified_training.py" path="unified_training.py" startline="1" type="function"></mcsymbol>'u çağırır.
7.  **Fine-tune** edilmiş modeli <mcsymbol name="Config.OUTPUT_DIR" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'a kaydeder.

## Kullanım:

Eğitim script'ini çalıştırmak için, <mcfile name="config.py" path="config.py"></mcfile> dosyanızı istediğiniz <mcsymbol name="Config.MODEL_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>, <mcsymbol name="Config.TRAINING_TECHNIQUE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol> ve diğer **hyperparameter**'larla yapılandırdığınızdan emin olun. Ardından, script'i çalıştırmanız yeterlidir:

```bash
python unified_training.py
```

Script, yapılandırmanıza göre veri yükleme, model hazırlama, eğitim ve değerlendirmeyi otomatik olarak halledecektir.

-----

# Yapılandırma Dosyası (`config.py`)

Bu dosya, birleşik eğitim script'i için tüm yapılandırılabilir parametreleri tanımlar. Model, PEFT teknikleri, eğitim **hyperparameter**'ları, **dataset** ve **hardware** ile ilgili çeşitli ayarları merkezileştirir.

## Parametreler:

### Model ve PEFT Parametreleri

  - <mcsymbol name="Config.MODEL_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: **Fine-tuning** için kullanılacak **base model**'i belirtir (örn. "gpt2").
  - <mcsymbol name="Config.LORA_R" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: LoRA güncelleme **matrix**'lerinin **rank**'ı.
  - <mcsymbol name="Config.LORA_ALPHA" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: LoRA güncellemeleri için **scaling factor**.
  - <mcsymbol name="Config.LORA_DROPOUT" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: LoRA **layer**'ları için **dropout probability**.
  - <mcsymbol name="Config.QUANT_4BIT" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: 4-bit **quantization** (QLoRA için) kullanılıp kullanılmayacağını belirten **boolean**.
  - <mcsymbol name="Config.BNB_4BIT_COMPUTE_DTYPE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: 4-bit modunda **computation**'lar için **data type** (örn. `torch.bfloat16`).
  - <mcsymbol name="Config.BNB_4BIT_QUANT_TYPE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: **Quantization type** (örn. "nf4").
  - <mcsymbol name="Config.BNB_4BIT_USE_DOUBLE_QUANT" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: İç içe **quantization** kullanılıp kullanılmayacağını belirten **boolean**.
  - <mcsymbol name="Config.TRAINING_TECHNIQUE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Kullanılacak PEFT tekniğini belirtir ("lora", "qlora", "prefix\_tuning", "soft\_prompting").
  - <mcsymbol name="Config.NUM_VIRTUAL_TOKENS" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Prefix Tuning ve Soft Prompting için sanal token sayısı.
  - <mcsymbol name="Config.PREFIX_PROJECTION" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Prefix Tuning için, **prefix**'i **project** edip etmeyeceğini belirten **boolean**.

### Eğitim Hyperparameter'ları

  - <mcsymbol name="Config.BATCH_SIZE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Her eğitim **batch**'inde işlenen örnek sayısı.
  - <mcsymbol name="Config.GRADIENT_ACCUMULATION_STEPS" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Ağırlıkları güncellemeden önceki **forward pass** sayısı.
  - <mcsymbol name="Config.LEARNING_RATE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: **Gradient descent** için **step size**.
  - <mcsymbol name="Config.NUM_EPOCHS" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: **Dataset** üzerinde yapılan tam geçiş sayısı.
  - <mcsymbol name="Config.MAX_SEQ_LENGTH" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Giriş **sequence**'larının maksimum uzunluğu.

### Dataset ve Çıktı Konfigürasyonu

  - <mcsymbol name="Config.DATASET_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Kullanılacak HuggingFace **dataset**'inin adı.
  - <mcsymbol name="Config.DATASET_CONFIG_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: **Dataset** için **configuration name** (örn. "en-tr").
  - <mcsymbol name="Config.OUTPUT_DIR" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Model **checkpoint**'lerini ve **log**'larını kaydetmek için **directory**.

### Eğitim İzleme Parametreleri

  - <mcsymbol name="Config.LOG_STEPS" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Eğitim metriklerinin ne sıklıkla **log**'a kaydedileceği.
  - <mcsymbol name="Config.SAVE_STEPS" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Model **checkpoint**'lerinin ne sıklıkla kaydedileceği.

### Donanım Konfigürasyonu

  - <mcsymbol name="Config.FP16" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: 16-bit **floating-point precision** kullanılıp kullanılmayacağını belirten **boolean**.
  - <mcsymbol name="Config.BF16" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: BFloat16 **precision** kullanılıp kullanılmayacağını belirten **boolean**.
  - <mcsymbol name="Config.DEVICE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>: Eğitim için kullanılacak **device**'ı belirtir (örn. Apple Silicon GPU için "mps", NVIDIA GPU için "cuda", "cpu").

-----

# Veri Hazırlama Script'i (`data_preparation.py`)

Bu script, eğitim için **dataset**'lerin yüklenmesi, **tokenization**'ı ve hazırlanmasını yönetir. Metin verilerini dil modelleri için uygun bir formata dönüştürmek amacıyla `datasets` ve `transformers` kütüphanelerini kullanır.

## Temel Özellikler:

  - **Dataset Yükleme**: HuggingFace'den **dataset**'leri yükler, belirli **configuration**'ları destekler.
  - **Tokenization**: Metni **numerical input ID**'lerine dönüştürmek için önceden eğitilmiş bir **tokenizer** kullanır, **padding** ve **truncation**'ı yönetir.
  - **Label Hazırlama**: **Causal language modeling** için **label**'ları, **loss function** tarafından göz ardı edilen **padding token**'larını tokenizer default padding token ile değiştirerek hazırlar.
  - **Dataset Bölme**: **Dataset**'in **train**, **validation** ve **test split**'lerini seçer ve hazırlar.

## Fonksiyon:

### <mcsymbol name="load_and_prepare_dataset" filename="data_preparation.py" path="data_preparation.py" startline="1" type="function"></mcsymbol>

Bu fonksiyon temel veri hazırlama adımlarını gerçekleştirir:

1.  **Tokenizer'ı Başlatır**: Sağlanan `model_name`'e göre `AutoTokenizer`'ı yükler ve **padding token**'ını ayarlar.
2.  **Dataset'i Yükler**: <mcsymbol name="Config.DATASET_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol> ve isteğe bağlı bir `config_name` kullanarak **dataset**'i HuggingFace'den getirir.
3.  **`tokenize_function`'ı Tanımlar**: İç içe bir fonksiyon olup şunları yapar:
      - **Dataset**'ten İngilizce (`en`) ve Türkçe (`tr`) çevirilerini çıkarır.
      - Başlatılan **tokenizer**'ı kullanarak **input**'ları ve **target**'ları **tokenize** eder.
      - Eğitim sırasında doğru **loss calculation** için **label**'lardaki `pad_token_id`'yi tokenizer padding token ile değiştirir.
4.  **Dataset Split'lerini Hazırlar**: `train`, `validation` ve `test split`'leriyle bir `DatasetDict` oluşturur. Şu anda, eğitim setini mevcutsa 1000 (mevcut memory bu kadar kaldırıyor) örnekle sınırlar.
5.  **Tokenization'ı Uygular**: `tokenize_function`'ı **dataset** boyunca eşler, orijinal "translation" sütununu kaldırır.

## Kullanım:

Bu script öncelikli olarak <mcfile name="unified_training.py" path="unified_training.py"></mcfile> tarafından model eğitimi için gerekli veriyi ve <mcfile name="config.py" path="config.py"></mcfile> dosyasında tanımlanan hiperparametrelerle birlikte kullanılır.

-----

# Çıkarım Script'i (`inference.py`)

Bu script, **fine-tune** edilmiş bir PEFT modeli yüklemek ve metin üretimi gerçekleştirmek için işlevsellikler sağlar. Modelin kolayca test edilmesi için etkileşimli bir **command-line interface** içerir.

## Temel Özellikler:

  - **Model Yükleme**: Bir **base model**'i yükler ve ardından **inference** için üzerine bir PEFT **adapter**'ı (**fine-tune** edilmiş model) uygular.
  - **Metin Üretimi**: Çeşitli **decoding strategy**'leri (örn. **temperature**, **top-k**, **top-p sampling**) kullanarak verilen bir **prompt**'a göre metin üretir.
  - **Etkileşimli Mod**: Etkileşimli olarak **prompt**'lar sağlamak ve üretilen metni almak için bir **command-line interface** sunar.

## Fonksiyonlar:

### <mcsymbol name="generate_text" filename="inference.py" path="inference.py" startline="1" type="function"></mcsymbol>

Yüklenen PEFT modelini kullanarak metin üretir. Bir `prompt`, `model` ve `tokenizer` nesneleri ile üretilen çıktının uzunluğunu kontrol etmek için isteğe bağlı bir `max_new_tokens` parametresi alır. Üretim, daha çeşitli çıktılar için `temperature`, `top_k` ve `top_p` ile **sampling** kullanır.

### <mcsymbol name="load_model_for_inference" filename="inference.py" path="inference.py" startline="1" type="function"></mcsymbol>

**Inference** için gerekli bileşenleri yükler:

1.  **Tokenizer**: <mcsymbol name="Config.MODEL_NAME" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'e göre `AutoTokenizer`'ı yükler.
2.  **Base Model**: <mcsymbol name="Config" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'den uygun `torch_dtype` ve `device_map` ayarlarıyla `AutoModelForCausalLM`'yi yükler.
3.  **PEFT Adapter**: <mcsymbol name="Config.OUTPUT_DIR" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'dan **fine-tune** edilmiş PEFT **adapter**'ını yükler ve **base model** ile birleştirir.
4.  Modeli **evaluation mode**'una (`model.eval()`) ayarlar ve belirtilen <mcsymbol name="Config.DEVICE" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol>'a taşır.

### <mcsymbol name="interactive_generation" filename="inference.py" path="inference.py" startline="1" type="function"></mcsymbol>

Terminalde **prompt**'lar girebileceğiniz etkileşimli bir döngü sağlar. Üretilen metin daha sonra konsola yazdırılır. Etkileşimli oturumdan çıkmak için 'exit' yazabilirsiniz.

## Kullanım:

**Inference script**'ini kullanmak için, <mcsymbol name="Config.OUTPUT_DIR" filename="config.py" path="config.py" startline="1" type="class"></mcsymbol> tarafından belirtilen dizinde (genellikle <mcfile name="unified_training.py" path="unified_training.py"></mcfile> tarafından oluşturulur) **fine-tune** edilmiş bir modelinizin kaydedilmiş olduğundan emin olun. Ardından, script'i çalıştırın:

```bash
python inference.py
```

Script, modeli yükleyecek ve sizden metin girmenizi isteyen etkileşimli bir moda girecektir.


# Training Ciktilari
## LoRa
[Epoch 2] Training Started
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [10:06<00:00,  2.42s/it]
[Epoch 2] Train Loss: 3.9015, Perplexity: 49.48
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:32<00:00,  1.03s/it]
[Epoch 2] Eval Loss: 4.0557, Perplexity: 57.72, BLEU: 0.0280

[Epoch 3] Training Started
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [10:25<00:00,  2.50s/it]
[Epoch 3] Train Loss: 3.8821, Perplexity: 48.52
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:46<00:00,  1.05s/it]
[Epoch 3] Eval Loss: 4.0671, Perplexity: 58.38, BLEU: 0.0280

[Epoch 4] Training Started
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [10:27<00:00,  2.51s/it]
[Epoch 4] Train Loss: 3.8223, Perplexity: 45.71
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [09:33<00:00,  1.15s/it]
[Epoch 4] Eval Loss: 4.0676, Perplexity: 58.42, BLEU: 0.0291

[Epoch 5] Training Started
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [11:07<00:00,  2.67s/it]
[Epoch 5] Train Loss: 3.7501, Perplexity: 42.53
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:35<00:00,  1.03s/it]
[Epoch 5] Eval Loss: 4.0630, Perplexity: 58.15, BLEU: 0.0289

[Epoch 6] Training Started
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [10:35<00:00,  2.54s/it]
[Epoch 6] Train Loss: 3.7219, Perplexity: 41.34
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [09:04<00:00,  1.09s/it]
[Epoch 6] Eval Loss: 4.0710, Perplexity: 58.62, BLEU: 0.0310
## QLoRa
[Epoch 0] Training Started
Training:   0%|                                                                                                                                                                                                 | 0/250 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [08:45<00:00,  2.10s/it]
[Epoch 0] Train Loss: 4.1798, Perplexity: 65.35
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:44<00:00,  1.08it/s]
[Epoch 0] Eval Loss: 4.0902, Perplexity: 59.75, BLEU: 0.0257

[Epoch 1] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:29<00:00,  2.28s/it]
[Epoch 1] Train Loss: 3.9627, Perplexity: 52.60
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:02<00:00,  1.04it/s]
[Epoch 1] Eval Loss: 4.0757, Perplexity: 58.89, BLEU: 0.0274

[Epoch 2] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:26<00:00,  2.27s/it]
[Epoch 2] Train Loss: 3.9079, Perplexity: 49.79
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:37<00:00,  1.09it/s]
[Epoch 2] Eval Loss: 4.0612, Perplexity: 58.05, BLEU: 0.0276

[Epoch 3] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:26<00:00,  2.27s/it]
[Epoch 3] Train Loss: 3.8630, Perplexity: 47.61
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:48<00:00,  1.07it/s]
[Epoch 3] Eval Loss: 4.0645, Perplexity: 58.24, BLEU: 0.0298

[Epoch 4] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:25<00:00,  2.26s/it]
[Epoch 4] Train Loss: 3.8353, Perplexity: 46.31
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:43<00:00,  1.08it/s]
[Epoch 4] Eval Loss: 4.0718, Perplexity: 58.66, BLEU: 0.0305

[Epoch 5] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:17<00:00,  2.23s/it]
[Epoch 5] Train Loss: 3.7625, Perplexity: 43.05
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:42<00:00,  1.08it/s]
[Epoch 5] Eval Loss: 4.0676, Perplexity: 58.41, BLEU: 0.0312

[Epoch 6] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:02<00:00,  2.17s/it]
[Epoch 6] Train Loss: 3.7686, Perplexity: 43.32
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:21<00:00,  1.13it/s]
[Epoch 6] Eval Loss: 4.0822, Perplexity: 59.28, BLEU: 0.0290

[Epoch 7] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:14<00:00,  2.22s/it]
[Epoch 7] Train Loss: 3.7044, Perplexity: 40.63
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:21<00:00,  1.13it/s]
[Epoch 7] Eval Loss: 4.0897, Perplexity: 59.72, BLEU: 0.0317

[Epoch 8] Training Started
Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:40<00:00,  2.32s/it]
[Epoch 8] Train Loss: 3.6583, Perplexity: 38.80
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [07:25<00:00,  1.12it/s]
[Epoch 8] Eval Loss: 4.1093, Perplexity: 60.90, BLEU: 0.0326
## Prefix Tuning

[Epoch 0] Training Started
Training:   0%|                                                                        | 0/250 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [05:56<00:00,  1.43s/it]
[Epoch 0] Train Loss: 4.1791, Perplexity: 65.31
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [04:45<00:00,  1.75it/s]
[Epoch 0] Eval Loss: 4.1121, Perplexity: 61.08, BLEU: 0.0258

[Epoch 1] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [05:53<00:00,  1.41s/it]
[Epoch 1] Train Loss: 3.8700, Perplexity: 47.94
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [04:40<00:00,  1.78it/s]
[Epoch 1] Eval Loss: 4.1335, Perplexity: 62.40, BLEU: 0.0052

[Epoch 2] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [05:53<00:00,  1.41s/it]
[Epoch 2] Train Loss: 3.7272, Perplexity: 41.56
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [04:40<00:00,  1.78it/s]
[Epoch 2] Eval Loss: 4.1190, Perplexity: 61.50, BLEU: 0.0020

[Epoch 3] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [06:18<00:00,  1.51s/it]
[Epoch 3] Train Loss: 3.6041, Perplexity: 36.75
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [04:18<00:00,  1.93it/s]
[Epoch 3] Eval Loss: 4.0942, Perplexity: 59.99, BLEU: 0.0294

[Epoch 4] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [05:42<00:00,  1.37s/it]
[Epoch 4] Train Loss: 4.8400, Perplexity: 126.46
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [04:11<00:00,  1.99it/s]
[Epoch 4] Eval Loss: 6.3230, Perplexity: 557.22, BLEU: 0.0002

[Epoch 5] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [06:31<00:00,  1.57s/it]
[Epoch 5] Train Loss: 3.9731, Perplexity: 53.15
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [06:00<00:00,  1.39it/s]
[Epoch 5] Eval Loss: 4.1080, Perplexity: 60.83, BLEU: 0.0112

[Epoch 6] Training Started
Training: 100%|██████████████████████████████████████████████████████████████| 250/250 [07:40<00:00,  1.84s/it]
[Epoch 6] Train Loss: 3.6465, Perplexity: 38.34
Evaluating: 100%|████████████████████████████████████████████████████████████| 500/500 [06:38<00:00,  1.25it/s]
[Epoch 6] Eval Loss: 4.1395, Perplexity: 62.77, BLEU: 0.0031


## Yorum

Dataset boyutu ve batch_size'dan dolayı eğitim oldukça dengesiz ilerlemektedir. Mevcut belleğim (MLX ile M4 Air 16 GB) daha fazlasını kaldırmadığı için eğitim yetersiz kalmaktadır. Mevcut eğitimi GPT-2 ile de yaptığım takdirde, mevcut BLEU metriği çok daha aşağılarda seyretmektedir.

