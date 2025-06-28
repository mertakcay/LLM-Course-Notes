
## Tokenizer

### Get Stats
Raw veya token ile replace edilmiş metin grubu ikili olarak count edilmesini sağlar. Bu fonksiyonun amacı en çok geçen 2 linin ortaya çıkarılmasını sağlamaktır.
### Train Function
Bu fonksiyonun amacı eğer mevcut bir vocab json ı varsa token dict extend edebilmek yoksa sıfırdan bir token vocab dict i sinin yaratılmasını sağlamaktır. Buradaki amaç her bir metinde sık geçen pair gruplarının token haline sayısal olarak ifade edebilmektir.

#### Line -55
Burada kadar mevcut train verisindeki metinleri line by line parçalamaktır ve mevcut token lar varsa bunu eklemektir.
#### Line 55-77
Burada kadar mevcut çıkartılmış token uzayını raw texte uygulayarak stepwise hale getirmeyi sağlamaktır. Extend edilmek istenen vocab size dan mevcut vocab düşülerek ne kadar extend edileceğini belirlemektedir.

#### Line 77-94
Burada replace edilmiş raw_text ile her bir word getirilmiştir. Bunun üzerine her bir word ün içerisindeki pairlar ortaya çıkartılıp kümül dict in içerisinde saklanmıştır ki uzun vadede burası secilebilen bir hale gelebilsin. Buradan gelen dict ile maximum bulunan pair grubu vocab a eklenerek extend edilmiştir.

### Tokenize

#### Line - 155
Okunan her bir word ü ele almıştır. Buradan sonra while döngüsüne girerek dönüştürebileceği token bulduğu kadar replace etmeye devam eder. 

#### Line 183-185
Burada sepesifik olarak en başta tüm charları veya char gruplarını token çevrimini yapar, eğer bilinmeyen bir token var ise UNK special token ataması yapar.

#### Line 192-210
Kelime gruplarına ayırdıktan sonra her bir char ı tokena çevirir. Bu token lar içerisinde pairlarını bulmaya çalışır eğer bulduğu token pairi self.merges içerisinde yani geçmişte train  edilmiş olan dict te var ise bunu bir temp_new_ids içerisinde tutar ta ki yeni bir ikili bulunamayasıya kadar.
Burada dikkat edilmesi gereken nokta en başta minimum pair ile işlemi yapmasıdır yani ;
1- M, e, r, t
2- Me, rt
3- Mert 
şekline bir merging policye sahiptir.
Burada bulabildiğini çevirmiş olup en son all_token_ids denilen listenin içerisinde tutulur ve bu ilgili word grubun token temsilini ifade eder.

##  Bert Mimarisi

### BertEmbeddings

Mevcut tokenlari input olarak alir. Eger exceed token varsa trimler, yoksa padding yapar. token types ile maskelenen var ise bunu bu array e ekler. Buradan gelen tokenlari normalize edip dropouttan gecirir.

__Boyutlar__
word_embeddings = [vocab_size, hidden_size] 
position_embeddings = [position_size, hidden_size]
token_type_embeddings = [token_types, hidden_size]

### BertSelfAttention

### __init__ fonksiyonu

Sequence Lenghti head e mod alir ve fazladan token kalmamasini saglar. Buradan gelen her bir segment sequence headlere paylastirilir ki uzun contextteki problemler ortaya cikmasin.
Q,K,V linear layerlari init edilir.

### __transpose_for_scores__ fonksiyonu
Headlerden gelmis segmented sequence matrisinin head formatina donusumunu saglar.

### Line 115 - 125
Q,K,V linear nn e feed edip gecirir. Buradan gecen deger segmentlere ayrilmis seqtir. Q,K,V carpimlari yapilir ve attention problari elde edilir. Elde edilen problarin olasilik toplaminin 1 olmasi icin softmaxten gecer.  Eger head mask varsa bu uygulanir. Buradan gelmis segmented(head seq lenght) flat edilir.

### BertSelfOutput(Aka Selfattention Output)
Düz bir MLP layerından farksızdır. Non-linearity katılıp normalizasyon eklenmiştir.

### Bert Attention
SelfAttention ve embedding için bir wrapper fonksiyonudur. Mevcut mimaride olan Q,K,V ve position bilgisinin toplandığı alana tekabül etmektedir.
### Bert Intermediate 
Hidden Space i daha derin bir uzaya maplemek için kullanılan bir MLP dir. Gelu tercih edilir.

### Bert Output
Derin uzaya maplenmis MLP yi tekrar hidden size a ceker. Buradan gelen sinyali normalize edip dropout ekler ki genelesstirme yetenegi artsin. Ek olarak residual baglantisi vardir.

### Bert Layer
BertAttention, BertIntermediate ve BertOutput blocklari icin wrapper fonksiyonudur. Intermediateten gelen sinyal ile output sinyali toplanarak residual olusturulmustur. attention_outputs[1: ] olmasinin sebebi batch size boyutunu devre disi birakmaktir.

### Bert Encoder

__init__ function 
Burada kac tane hidden layer belirlendiyse o kadar Bert Layeri initilize edilir. 

### Line 219-236
Onceki steplerden tanimlanmamis all_hidden_space/attentions yoksa yaratilir. Her bir layerin hidden space i all hidden space ile birlestirilir, ayni durum attention icin varsa oda mevcut degiskenle birlestirilir ve tuple olarak return edilir. Bu cikti aslinda mevcut tokenlarin position ve NN den embed edilmis space i temsil ederi. Bunun uzerine kurulacak olan MLMHead ile surec devam edilecektir.

### Bert Pooler
Medium yazisinda gordum ancak emin degilim. First token i alip iceriye feed ediyor amacini tam anlamadim. Bundan dolayi none a cektim.
### BertModel
Onceden yaratilmis olan BertEmbeddings ve Encoder init edilir. Her bir lineear, Embedding ve LayerNorm katmanlari icin weightleri yaratilir.

### Line 287 - 
Input embedings veya input ids alinir, Burasi geldikten sonra her bir seq_lenght obegi ve batch boyutlari cikartilir. Predefined bir attention_mask veya token_type tanimi varsa bunlar eklenir yoksa default yaratim islemi gerceklesir. Buna ek olarak head_mask tanimi varsa bunlarin boyutlari duzenlenir. Tek boyutlu verdiyse hidden_dim_mask, sonrasina seq_lenght, hidden_dim eklemeleri yapilir. Iki boyutluysa yani her bir hidden_dim_mask tanimi farkliysa zaten verildigi icin sadece sagina seq_lenght ve hidden_dim eklemeleri yapilir. Eger herhangi bir maske kullanilmayacaksa None eklenerek sifirlanmis olur. Burada [0] ile ulasmamizin sebebi encoder [hidden_states, all_hidden_states, all_attentions] donduruyor bizim ihtiyacimiz olan sadece hidden_state yani embed edilmis bilgi oldugu icin buna erisim sagliyoruz.

### BertLMPredictionHead
Embedding space de sadece mapping yapiyor. gelu ile non-linearity i sagliyor.


### BertForMaskedLM - Wrapper Function

Simdiye kadar tum layerlarda kullanilacak olan weight ve biaslari init ediyor. self.bert ile latent space olusturuyor daha sonra yazmis oldugumuz BERTLMPrediction head ile [batch_size, sequence_lenght, vocab_size] boyutunda bir tahmin matrisi olusturuyor.

## MLM Dataset
Tanım olarak hardcoded bir max_lenght seçimi yapılır
### Line 52-55
Geçmişte tanımlanmış olan BPE tokenizer ile tokenize edilmiş olur. Buradan gelen tokenlara random şekilde mask veya random replacemant  uygulanması gerekmektedir. Bu işlemde *mask_token* fonksiyonu ile sağlanmaktadır.

### Line 55-85
Her bir token grubu [-100] ile CLS tokenini temsil eder. Bunun trainingde kullanilmsi icin egitim asamasini gectikten sonra burasi disarida birakilir.
Arasında ise ilgili mask ve replacemant a göre ekleme çıkarmalar yapılır. Buna ek olarak eğer exceed token varsa trimlenme işlemi, yetersiz token var ise padding işlemi uygulanmış olur. Prob degeri ile maskeleme veya replacemant islemi yapilir.


## Generation Layer

### Line -37
Her bir maska kadar olan cümle obeklerini her iterasyonda tamamlamak için segmente ayırırız. 

### Line 37-44
Predefined bir mask tokeni varsa bunu feedlemek amacıyla kullanılız.

### Line 44-
Mask keşfini tamamlamam ihtimaline karşı buffer iterasyon tanımlanır. Buradaki iterasyon sayısı tum masking adımını kapsamaktadır. Ek olarak her bir masking aşamasını tamamlamak için decoding layerını yaratılır. Buradaki decoding layerı greed search algoritmasını kullanır. Ek olarak temperature ile generation probları düzenlenmiş olur. Top k yani en çok açılacak olan olasılık ağacı dinamik olarak seçilebilir. Elde edilen token uzayı softmaxten geçirerek toplamı 1 olarak şekilde düzenlenmiş olur. Line 76 da olan multinomial sampling kullanılarak top token seçilir ve current_token_ids yapısına eklenmiş olur. Tüm bu adımlar masking + 5 kadar ilerler ve tokenler elde edilmiş olur. Elde edilen tokenler tokenizerde yapmış olduğumuz reverse tokenize yani decoding(tree decoding ile karışmamalı) tekrar string öbeğine dönüştürülüp yazdırılır.

### Generation Stage 
Filling masks for prompt: 'Mert Akçay [MASK] üniversite [MASK].' (temp=0.0, top_k=10)
segment text Mert Akçay 
latest 56
segment text  üniversite 
latest 56
segment text .
[55, 15290, 81, 83, 1290, 285, 88, 56, 155, 72, 154, 81, 306, 204, 56, 11]
Filled text: MertAkçaynüniversiter.

Tahmin edebileceğimiz gibi ssonuç oldukça kötü.
### Training Stage 

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








