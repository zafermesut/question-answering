Size tokenizasyon ve projelerdeki tokenizasyon kodlarını adım adım açıklayacağım.

## Tokenizasyon Nedir?

Tokenizasyon, metni daha küçük parçalara (token'lara) ayırma işlemidir. Bu parçalar kelimeler, alt kelimeler veya karakterler olabilir. Tokenizasyon, NLP modellerinin metni anlayabilmesi için gerekli ilk adımdır çünkü modeller sayısal verilerle çalışır.

## BERT Projesindeki Tokenizasyon

```python:train/bert_train.ipynb
# Tokenizer yükleme
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizasyon işlemi
inputs = tokenizer(
    questions,
    context,
    max_length=512,
    truncation="only_second",
    stride=128,
    return_offsets_mapping=True,
    padding="max_length",
)
```

BERT tokenizasyon adımları:

1. `AutoTokenizer.from_pretrained()`: BERT'in önceden eğitilmiş tokenizer'ını yükler
2. Tokenizer iki girdi alır: sorular ve bağlam metinleri
3. Parametreler:
   - `max_length=512`: Maximum token uzunluğu
   - `truncation="only_second"`: Sadece bağlam metnini kesme
   - `stride=128`: Örtüşen pencere boyutu
   - `return_offsets_mapping=True`: Token konumlarını döndürme
   - `padding="max_length"`: Tüm örnekleri aynı uzunluğa getirme

## T5 Projesindeki Tokenizasyon

```python:train/t5_train.ipynb
# Tokenizer yükleme
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Preprocess fonksiyonu
def preprocess_squad(examples):
    # Girdileri formatlama
    inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]

    # Girdi tokenizasyonu
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Çıktı tokenizasyonu
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([ans["text"][0] if ans["text"] else "" for ans in examples["answers"]],
                       max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

T5 tokenizasyon adımları:

1. `T5Tokenizer.from_pretrained()`: T5'in önceden eğitilmiş tokenizer'ını yükler
2. Preprocess fonksiyonu içinde:
   - Soru ve bağlam birleştirilerek özel bir format oluşturulur
   - Girdi metinleri tokenize edilir
   - Cevaplar (labels) ayrıca tokenize edilir
3. Parametreler:
   - `max_length=512`: Girdi için maximum token uzunluğu
   - `max_length=128`: Cevaplar için maximum token uzunluğu
   - `truncation=True`: Uzun metinleri kesme
   - `padding="max_length"`: Sabit uzunluğa padding

## Temel Farklar:

1. BERT tek yönlü tokenizasyon yaparken (girdi), T5 hem girdi hem çıktı için tokenizasyon yapar
2. T5'te özel bir format kullanılır ("question: ... context: ...")
3. BERT offset mapping kullanırken T5 buna ihtiyaç duymaz
4. T5'te cevaplar için ayrı bir tokenizasyon işlemi yapılır

Bu tokenizasyon işlemleri, modellerin metni anlayabilmesi ve işleyebilmesi için gerekli ön işleme adımlarıdır.
