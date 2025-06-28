

# BERT Mimarisi: BPE Tokenizer ve Maskeli Dil Modeli (MLM) Görevi

Bu doküman, BERT (Bidirectional Encoder Representations from Transformers) modelinin temel bileşenlerini, Byte Pair Encoding (BPE) tabanlı tokenizer'ı ve Maskeli Dil Modeli (MLM) görevini ayrıntılı bir şekilde açıklamaktadır.

## Bölüm 1: BPE (Byte Pair Encoding) Tokenizer

BPE, metin verilerini alt kelime birimlerine (subword units) ayırmak için kullanılan bir veri sıkıştırma ve tokenizasyon algoritmasıdır. Temel amacı, sık tekrarlanan karakter dizilerini birleştirerek daha küçük ve anlamlı token'lar oluşturmaktır.

### 1.1. İstatistik Toplama (`Get Stats`)

Bu adım, ham metin veya önceden token'lara ayrılmış metin içerisindeki ardışık token çiftlerinin (bigramların) frekanslarını sayar. Amaç, veri kümesinde en sık birlikte görünen token çiftlerini belirlemektir. Bu bilgi, birleştirilecek bir sonraki en iyi çifti seçmek için kullanılır.

### 1.2. Tokenizer Eğitimi (`Train Function`)

Tokenizer eğitim fonksiyonu, bir kelime dağarcığı (vocabulary) oluşturmayı veya mevcut bir kelime dağarcığını genişletmeyi hedefler. Süreç aşağıdaki adımları içerir:

1.  **Başlangıç Kelime Dağarcığı:** Genellikle karakterlerden oluşan bir temel kelime dağarcığı ile başlanır.
2.  **İteratif Birleştirme:**
    *   Metin korpusundaki her kelime, karakterlerine veya mevcut token'larına ayrılır.
    *   `Get Stats` fonksiyonu kullanılarak en sık geçen token çifti bulunur.
    *   Bu en sık geçen çift, yeni bir token olarak kelime dağarcığına eklenir ve bu çiftin geçtiği tüm yerlerde yeni token ile değiştirilir.
    *   Bu işlem, hedeflenen kelime dağarcığı boyutuna (`vocab_size`) ulaşılana kadar veya birleştirilecek anlamlı çift kalmayana kadar tekrarlanır.

Bu sürecin sonunda, sık tekrarlanan ve potansiyel olarak anlamsal birimler taşıyan karakter dizileri (örneğin, "er", "ing", "tion") tek bir token olarak temsil edilir. Eğer mevcut bir kelime dağarcığı (`vocab.json`) varsa, eğitim süreci bu dağarcığı genişletebilir; aksi takdirde sıfırdan bir dağarcık oluşturulur.

### 1.3. Tokenizasyon İşlemi (`Tokenize`)

Eğitilmiş BPE tokenizer, yeni metinleri token'larına ayırmak için kullanılır:

1.  **Kelime Ön İşleme:** Girdi metni genellikle kelimelere ayrılır.
2.  **Karakterlere Ayırma:** Her kelime başlangıçta karakterlerine ayrılır (veya biliniyorsa mevcut alt kelime token'larına).
3.  **İteratif Birleştirme Uygulaması:**
    *   Kelime içindeki ardışık token çiftleri incelenir.
    *   Eğer bir çift, eğitim sırasında öğrenilen birleştirmeler (`self.merges` içinde) arasında yer alıyorsa, bu çift birleştirilerek tek bir token oluşturulur.
    *   Bu işlem, kelime içinde daha fazla birleştirme yapılamayana kadar tekrarlanır. Birleştirme politikası genellikle en sık veya en erken öğrenilen birleştirmelere öncelik verir. Örneğin, "Mert" kelimesi için adımlar şu şekilde olabilir:
        1.  Başlangıç: `M`, `e`, `r`, `t`
        2.  Eğer "Me" ve "rt" birleştirmeleri öğrenilmişse: `Me`, `rt`
        3.  Eğer "Mert" birleştirmesi öğrenilmişse (veya "Me" + "rt" -> "Mert"): `Mert`
4.  **Bilinmeyen Token'lar:** Eğer bir karakter veya karakter grubu kelime dağarcığında bulunmuyorsa, genellikle özel bir `[UNK]` (unknown) token'ı ile temsil edilir.

Sonuç olarak, her kelime bir veya daha fazla alt kelime token'ından oluşan bir diziye dönüştürülür. Bu token dizisi, modelin girdisi olarak kullanılır.

## Bölüm 2: BERT Mimarisi

BERT, Transformer mimarisinin kodlayıcı (encoder) katmanlarını temel alır. Temel amacı, metindeki her bir token için bağlamsal olarak zengin bir temsil (embedding) öğrenmektir.

### 2.1. BertEmbeddings

Bu katman, girdi token ID'lerini alır ve her bir token için üç tür embedding'i birleştirerek başlangıç temsillerini oluşturur:

1.  **Token Embeddings:** Her bir token'ın kelime dağarcığındaki ID'sine karşılık gelen öğrenilebilir vektörlerdir. Boyutu `[vocab_size, hidden_size]` şeklindedir.
2.  **Position Embeddings:** Token'ların dizideki pozisyonunu kodlayan öğrenilebilir vektörlerdir. BERT, mutlak pozisyon embedding'leri kullanır. Boyutu `[max_position_embeddings, hidden_size]` şeklindedir.
3.  **Token Type Embeddings (Segment Embeddings):** Özellikle cümle çifti görevlerinde (örneğin, Soru Cevaplama, Doğal Dil Çıkarımı) hangi token'ın hangi cümleye ait olduğunu belirtmek için kullanılır. Genellikle iki segment (A ve B) için öğrenilebilir vektörlerdir. Boyutu `[token_type_vocab_size, hidden_size]` (genellikle `[2, hidden_size]`) şeklindedir.

Bu üç embedding toplanır. Girdi dizisinin uzunluğu `max_position_embeddings` değerini aşarsa kırpılır (trim), kısa kalırsa özel bir `[PAD]` token'ı ile doldurulur (padding). Son olarak, bir Layer Normalization ve Dropout katmanından geçirilir.

### 2.2. BertSelfAttention (Çok Başlı Öz-Dikkat Mekanizması)

Öz-dikkat, bir dizideki her token'ın diğer tüm token'larla ilişkisini ağırlıklandırarak bağlamsal temsiller üretmesini sağlar.

*   **`__init__` Fonksiyonu:**
    *   `num_attention_heads` (dikkat başlığı sayısı) tanımlanır. `hidden_size`, başlık sayısına tam bölünmelidir (`attention_head_size = hidden_size / num_attention_heads`).
    *   Sorgu (Query - Q), Anahtar (Key - K) ve Değer (Value - V) için lineer projeksiyon katmanları (`nn.Linear(hidden_size, all_head_size)`) başlatılır. `all_head_size` genellikle `hidden_size` ile aynıdır.

*   **`transpose_for_scores` Fonksiyonu:**
    Bu yardımcı fonksiyon, Q, K, V matrislerini çoklu dikkat başlıklarına uygun hale getirmek için yeniden şekillendirir. Girdi `[batch_size, seq_length, hidden_size]` boyutundan `[batch_size, num_attention_heads, seq_length, attention_head_size]` boyutuna dönüştürülür.

*   **Dikkat Hesaplaması:**
    1.  Girdi temsilleri (önceki katmanın çıktısı), Q, K, V lineer katmanlarından geçirilir.
    2.  Bu Q, K, V çıktıları, `transpose_for_scores` ile başlık formatına getirilir.
    3.  **Ölçeklenmiş Nokta Çarpımı Dikkat (Scaled Dot-Product Attention):** Her başlık için bağımsız olarak hesaplanır:
        `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(attention_head_size) ) * V`
        *   `Q * K^T`: Sorgu ve anahtar matrislerinin çarpımı, token'lar arası benzerlik skorlarını verir.
        *   `/ sqrt(attention_head_size)`: Ölçekleme faktörü, gradyanların stabil kalmasına yardımcı olur.
        *   `softmax`: Benzerlik skorlarını olasılık dağılımına (dikkat ağırlıkları) dönüştürür.
        *   Bu ağırlıklar V (değer) matrisi ile çarpılarak her token için ağırlıklı bir temsil elde edilir.
    4.  İsteğe bağlı olarak dikkat maskesi (`attention_mask`) uygulanabilir (örneğin, padding token'larına dikkat etmemek için).
    5.  Başlıklardan gelen dikkat çıktıları birleştirilir (`concat`) ve bir lineer projeksiyondan (`nn.Linear(all_head_size, hidden_size)`) geçirilir.

### 2.3. BertSelfOutput

Bu katman, `BertSelfAttention` mekanizmasının çıktısını alır, bir lineer katmandan geçirir, ardından bir Dropout ve Layer Normalization uygular. Ayrıca, dikkat mekanizmasının girdisiyle bir artık bağlantı (residual connection) oluşturur:
`output = LayerNorm(attention_output + input_tensor)`

### 2.4. BertAttention

Bu modül, `BertSelfAttention` ve `BertSelfOutput` katmanlarını bir araya getiren bir sarmalayıcıdır. Bir öz-dikkat bloğunu temsil eder.

### 2.5. BertIntermediate

Bu katman, BERT katmanındaki pozisyon bazlı ileri beslemeli ağın (feed-forward network - FFN) ilk kısmıdır. Genellikle `BertAttention` çıktısının boyutunu daha büyük bir ara boyuta (`intermediate_size`) genişleten bir lineer katman ve ardından bir aktivasyon fonksiyonu (genellikle GELU) içerir.
`output = activation_function(nn.Linear(hidden_size, intermediate_size)(input_tensor))`

### 2.6. BertOutput

Bu katman, FFN'nin ikinci kısmıdır. `BertIntermediate` çıktısını alır, boyutu tekrar orijinal `hidden_size`'a düşüren bir lineer katmandan geçirir. Ardından Dropout ve Layer Normalization uygulanır. `BertIntermediate` girdisiyle bir artık bağlantı oluşturur:
`output = LayerNorm(nn.Linear(intermediate_size, hidden_size)(intermediate_output) + intermediate_input)`

### 2.7. BertLayer

Bir `BertLayer`, tam bir Transformer kodlayıcı bloğunu temsil eder. Sırasıyla şunları içerir:
1.  `BertAttention` (çok başlı öz-dikkat ve ilişkili çıktı katmanı)
2.  `BertIntermediate` (FFN'nin genişletme kısmı)
3.  `BertOutput` (FFN'nin daraltma ve artık bağlantı kısmı)

`BertIntermediate`'ten gelen sinyal ile `BertOutput` sinyali (aslında `BertIntermediate`'in girdisi) toplanarak artık bağlantı oluşturulur.

### 2.8. BertEncoder

*   **`__init__` Fonksiyonu:** Belirlenen sayıda (`num_hidden_layers`) `BertLayer` örneğini başlatır ve bir `nn.ModuleList` içinde saklar.

*   **İleri Yayılım (`forward`):**
    1.  Girdi embedding'leri (`hidden_states`) alır.
    2.  İsteğe bağlı olarak `attention_mask` ve `head_mask` alır.
    3.  Girdiyi sıralı olarak her bir `BertLayer`'dan geçirir. Her katmanın çıktısı, bir sonraki katmanın girdisi olur.
    4.  Eğer istenirse (`output_hidden_states=True`), tüm katmanların çıktılarını (`all_hidden_states`) biriktirir.
    5.  Eğer istenirse (`output_attentions=True`), tüm katmanlardaki dikkat ağırlıklarını (`all_attentions`) biriktirir.
    6.  Son katmanın çıktısı (`sequence_output`), tüm gizli durumlar ve tüm dikkat ağırlıkları bir demet (tuple) olarak döndürülür: `(sequence_output, all_hidden_states, all_attentions)`.
    `sequence_output`, her bir girdi token'ı için son katmandan elde edilen bağlamsal temsilleri içerir.

### 2.9. BertPooler

`BertPooler` katmanı, genellikle sınıflandırma görevleri için tüm girdi dizisinin birleşik bir temsilini elde etmek amacıyla kullanılır. Tipik olarak, girdi dizisinin ilk token'ının ([CLS] token'ı) son katmandaki gizli durumunu (`sequence_output[:, 0]`) alır, bir lineer katmandan geçirir ve bir Tanh aktivasyon fonksiyonu uygular. Eğer model sadece dil modelleme gibi görevler için kullanılıyorsa veya farklı bir havuzlama stratejisi tercih ediliyorsa bu katman ihmal edilebilir veya `None` olarak ayarlanabilir.

### 2.10. BertModel

Bu, BERT mimarisinin ana sarmalayıcı sınıfıdır.

*   **`__init__` Fonksiyonu:**
    *   `BertEmbeddings` ve `BertEncoder` katmanlarını başlatır.
    *   Gerekli tüm lineer katmanlar, embedding katmanları ve Layer Normalization katmanları için ağırlıkları (weights) ve sapmaları (biases) ilklendirir (genellikle belirli bir ortalama ve standart sapma ile normal dağılımdan örneklenerek).

*   **İleri Yayılım (`forward`):**
    1.  Girdi olarak `input_ids` (token ID'leri) veya `inputs_embeds` (doğrudan embedding'ler) alabilir.
    2.  `attention_mask` (padding token'larını maskelemek için) ve `token_type_ids` (segment bilgisi için) gibi ek girdileri işler. Eğer bunlar sağlanmazsa, varsayılan olarak uygun maskeler ve ID'ler oluşturulur.
    3.  `head_mask` (belirli dikkat başlıklarını maskelemek için) varsa, boyutlarını düzenler ve `BertEncoder`'a iletir.
    4.  `BertEmbeddings` katmanını kullanarak başlangıç embedding'lerini oluşturur.
    5.  Bu embedding'leri ve diğer maskeleri `BertEncoder`'a verir.
    6.  `BertEncoder`'dan dönen `(sequence_output, pooled_output, all_hidden_states, all_attentions)` demetini alır. `pooled_output` genellikle `BertPooler`'ın çıktısıdır (eğer varsa). `sequence_output`, `BertEncoder`'ın son katmanının token başına çıktılarıdır.

### 2.11. BertLMPredictionHead

Bu başlık, Maskeli Dil Modelleme (MLM) görevi için kullanılır. `BertEncoder`'dan gelen son katman gizli durumlarını (`sequence_output`) alır ve her bir token pozisyonu için kelime dağarcığı üzerinde bir olasılık dağılımı tahmin eder.

Genellikle şu adımları içerir:
1.  Bir lineer dönüşüm katmanı (`transform`): `hidden_size` -> `hidden_size`.
2.  Bir aktivasyon fonksiyonu (genellikle GELU).
3.  Bir Layer Normalization katmanı.
4.  Çıkış lineer katmanı (`decoder`): `hidden_size` -> `vocab_size`. Bu katmanın ağırlıkları genellikle token embedding matrisi (`bert.embeddings.word_embeddings.weight`) ile paylaşılır (tied weights), bu da modelin daha verimli olmasını sağlar.

Çıktısı, `[batch_size, sequence_length, vocab_size]` boyutunda bir logit matrisidir.

### 2.12. BertForMaskedLM (Maskeli Dil Modeli için BERT)

Bu, MLM görevi için tam bir BERT modelini oluşturan bir sarmalayıcıdır.

*   **`__init__` Fonksiyonu:**
    *   Bir `BertModel` örneği (`self.bert`) başlatır.
    *   Bir `BertLMPredictionHead` örneği (`self.cls` veya benzeri bir isimle) başlatır.
    *   Tüm ağırlıkları ve sapmaları ilklendirir.

*   **İleri Yayılım (`forward`):**
    1.  Girdileri (`input_ids`, `attention_mask`, `token_type_ids` vb.) alır.
    2.  Bu girdileri `self.bert` modeline vererek `sequence_output` (son katman gizli durumları) ve diğer çıktıları elde eder.
    3.  `sequence_output`'u `BertLMPredictionHead`'e vererek her token pozisyonu için kelime dağarcığı üzerinde logit tahminleri (`prediction_scores`) alır.
    4.  Eğitim sırasında, eğer `masked_lm_labels` sağlanırsa, bu logit'ler ve etiketler kullanılarak bir kayıp fonksiyonu (genellikle Çapraz Entropi Kaybı - Cross-Entropy Loss) hesaplanır. Kayıp sadece maskelenmiş token pozisyonları için hesaplanır.
    5.  Çıktı olarak `(loss, prediction_scores, all_hidden_states, all_attentions)` gibi bir demet döndürür.

## Bölüm 3: MLM (Masked Language Model) Veri Kümesi Hazırlığı

MLM görevi için veri kümesi hazırlığı, BERT'in ön eğitimi (pre-training) için kritik bir adımdır.

1.  **Maksimum Uzunluk (`max_length`):** Modelin işleyebileceği maksimum token dizisi uzunluğu belirlenir (örneğin, 512).

2.  **Tokenizasyon:** Girdi metinleri, daha önce eğitilmiş BPE tokenizer kullanılarak token dizilerine dönüştürülür.

3.  **Maskeleme Stratejisi (`mask_tokens` fonksiyonu):**
    Token dizisinden rastgele bir yüzde (genellikle %15) token seçilir ve bu token'lar aşağıdaki kurallara göre değiştirilir:
    *   **%80 olasılıkla:** Seçilen token, özel bir `[MASK]` token'ı ile değiştirilir.
    *   **%10 olasılıkla:** Seçilen token, kelime dağarcığından rastgele başka bir token ile değiştirilir.
    *   **%10 olasılıkla:** Seçilen token değiştirilmez (orijinal haliyle bırakılır).
    Bu strateji, modelin sadece `[MASK]` token'larını değil, aynı zamanda bozulmuş girdileri de düzeltebilmesini ve bağlamı daha iyi anlamasını sağlar.

4.  **Özel Token'ların Eklenmesi:** Her dizinin başına genellikle bir `[CLS]` (sınıflandırma) token'ı ve cümleleri ayırmak için (eğer varsa) `[SEP]` (ayırıcı) token'ı eklenir.

5.  **Padding ve Trimming:**
    *   Token dizisi `max_length`'ten uzunsa, sondan kırpılır.
    *   Token dizisi `max_length`'ten kısaysa, sonuna özel bir `[PAD]` token'ı eklenerek `max_length`'e kadar doldurulur.
    *   Bir `attention_mask` oluşturulur: gerçek token'lar için 1, `[PAD]` token'ları için 0 değerini alır.

6.  **Etiketlerin (Labels) Oluşturulması:**
    Modelin tahmin etmesi gereken token'lar için etiketler oluşturulur. Sadece maskelenmiş veya değiştirilmiş pozisyonlardaki orijinal token ID'leri etiket olarak kullanılır. Diğer pozisyonlar için etiketler genellikle -100 gibi bir değere ayarlanır, böylece kayıp hesaplamasına dahil edilmezler.

Bu adımlar sonucunda modelin eğitimi için `input_ids`, `attention_mask`, `token_type_ids` (gerekliyse) ve `labels` (maskelenmiş token'lar için) tensörleri hazırlanmış olur.



## Training Stage 

Epoch 3/3, Batch 750/1191, Loss: 10.8566, Avg Loss: 14.3355, Batch Acc: 0.0702, Avg Acc: 0.0670, LR: 6.35e-06
Epoch 3/3, Batch 760/1191, Loss: 13.4623, Avg Loss: 14.3201, Batch Acc: 0.0702, Avg Acc: 0.0672, LR: 6.21e-06
Epoch 3/3, Batch 770/1191, Loss: 8.8180, Avg Loss: 14.3096, Batch Acc: 0.0851, Avg Acc: 0.0673, LR: 6.06e-06
Epoch 3/3, Batch 780/1191, Loss: 17.5170, Avg Loss: 14.2937, Batch Acc: 0.0577, Avg Acc: 0.0677, LR: 5.92e-06
Epoch 3/3, Batch 790/1191, Loss: 14.9524, Avg Loss: 14.2912, Batch Acc: 0.0357, Avg Acc: 0.0674, LR: 5.77e-06
Epoch 3/3, Batch 800/1191, Loss: 13.3556, Avg Loss: 14.2847, Batch Acc: 0.0667, Avg Acc: 0.0675, LR: 5.63e-06
Epoch 3/3, Batch 810/1191, Loss: 14.8332, Avg Loss: 14.2637, Batch Acc: 0.1111, Avg Acc: 0.0678, LR: 5.49e-06
Epoch 3/3, Batch 820/1191, Loss: 12.6167, Avg Loss: 14.2420, Batch Acc: 0.0909, Avg Acc: 0.0681, LR: 5.34e-06
Epoch 3/3, Batch 830/1191, Loss: 17.0111, Avg Loss: 14.2580, Batch Acc: 0.0465, Avg Acc: 0.0679, LR: 5.20e-06
Epoch 3/3, Batch 840/1191, Loss: 16.0633, Avg Loss: 14.2594, Batch Acc: 0.0889, Avg Acc: 0.0680, LR: 5.05e-06
Epoch 3/3, Batch 850/1191, Loss: 12.8066, Avg Loss: 14.2432, Batch Acc: 0.1639, Avg Acc: 0.0681, LR: 4.91e-06
Epoch 3/3, Batch 860/1191, Loss: 16.2759, Avg Loss: 14.2349, Batch Acc: 0.0357, Avg Acc: 0.0685, LR: 4.77e-06
Epoch 3/3, Batch 870/1191, Loss: 9.4121, Avg Loss: 14.2367, Batch Acc: 0.0952, Avg Acc: 0.0684, LR: 4.62e-06
Epoch 3/3, Batch 880/1191, Loss: 10.9391, Avg Loss: 14.2132, Batch Acc: 0.1290, Avg Acc: 0.0686, LR: 4.48e-06
Epoch 3/3, Batch 890/1191, Loss: 13.9651, Avg Loss: 14.2094, Batch Acc: 0.1286, Avg Acc: 0.0687, LR: 4.33e-06
Epoch 3/3, Batch 900/1191, Loss: 16.3899, Avg Loss: 14.1994, Batch Acc: 0.1224, Avg Acc: 0.0688, LR: 4.19e-06
Epoch 3/3, Batch 910/1191, Loss: 14.4480, Avg Loss: 14.1802, Batch Acc: 0.0784, Avg Acc: 0.0689, LR: 4.05e-06
Epoch 3/3, Batch 920/1191, Loss: 10.9332, Avg Loss: 14.1598, Batch Acc: 0.0946, Avg Acc: 0.0691, LR: 3.90e-06






