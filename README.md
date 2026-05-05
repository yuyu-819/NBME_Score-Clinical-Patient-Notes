# NBME - Score Clinical Patient Notes

本專案目前已完成 preprocessing。接下來的主線是：讀取 preprocess 後的
<mark>`train_preprocessed_5fold.pkl`</mark>，建立 tokenization 與 label，訓練模型，做 validation 後處理，最後輸出 <mark>`submission.csv`</mark>。

## 資料檔案

原始 Kaggle 資料放在：

- <mark>`data/nbme-score-clinical-patient-notes/train.csv`</mark>
- <mark>`data/nbme-score-clinical-patient-notes/test.csv`</mark>
- <mark>`data/nbme-score-clinical-patient-notes/features.csv`</mark>
- <mark>`data/nbme-score-clinical-patient-notes/patient_notes.csv`</mark>
- <mark>`data/nbme-score-clinical-patient-notes/sample_submission.csv`</mark>

preprocess 後的訓練資料：

- <mark>`data/nbme-score-clinical-patient-notes/processed/train_preprocessed_5fold.pkl`</mark>

## Preprocess 產出欄位

<mark>`train_preprocessed_5fold.pkl`</mark> 目前包含以下欄位：

| 欄位 | 用途 |
| --- | --- |
| <mark>`id`</mark> | 每筆資料的唯一 ID。validation decode、錯誤分析、submission 對回資料時使用。 |
| <mark>`case_num`</mark> | 病例類別。主要用於分 fold、檢查 case 分布，也可做 case-level 分析。 |
| <mark>`pn_num`</mark> | patient note 編號。分 fold 時避免同一份 note 同時出現在 train / valid。 |
| <mark>`feature_num`</mark> | feature 編號。與 <mark>`case_num`</mark> 一起決定要找的 clinical feature。 |
| <mark>`pn_history`</mark> | patient note 原文。模型要從這段文字中找答案 span。 |
| <mark>`feature_text`</mark> | 要尋找的 clinical feature。tokenization 時會和 <mark>`pn_history`</mark> 一起作為模型輸入。 |
| <mark>`annotation_text`</mark> | 標註文字內容。主要用於 debug / 檢查，不是訓練 label 的主要來源。 |
| <mark>`char_spans`</mark> | 真實答案在 <mark>`pn_history`</mark> 裡的 character span，例如 <mark>`[(203, 217)]`</mark>。建立 token label 與 validation scoring 時使用。 |
| <mark>`fold`</mark> | 5-fold split 編號。訓練時用 <mark>`fold != k`</mark>，驗證時用 <mark>`fold == k`</mark>。 |

訓練最核心會用到：

```text
pn_history
feature_text
char_spans
fold
id
```

其中 <mark>`annotation_text`</mark> 通常不直接餵給模型；真正的 label 來源是 <mark>`char_spans`</mark>。

## 使用之模型

使用 <mark>`token classification`</mark>。

因為 NBME 的答案可能有多段、不連續 span，例如：

```text
203 217;300 315
```

<mark>`token classification`</mark> 可以直接預測每個 token 是否屬於答案，比 <mark>`start/end prediction`</mark> 更直覺。    <mark>`start/end prediction`</mark> 比較像 QA 任務，天然較適合單一連續 span。

不同 backbone：

- <mark>`microsoft/deberta-v3-base`</mark>
- <mark>`microsoft/deberta-v3-large`</mark>
- <mark>`roberta-large`</mark>

## Tokenization

tokenization 使用：

```text
feature_text + pn_history
```

-> <mark>`feature_text`</mark> :找尋目標， <mark>`pn_history`</mark> :內容

範例：

```python
encoded = tokenizer(
    feature_text,
    pn_history,
    truncation="only_second",
    max_length=512,
    padding="max_length",
    return_offsets_mapping=True,
)
```

tokenizer 產生：

| 欄位 | 用途 |
| --- | --- |
| <mark>`input_ids`</mark> | 模型輸入 token ids。訓練時會餵給模型。 |
| <mark>`attention_mask`</mark> | 告訴模型哪些位置是有效 token，哪些是 padding。訓練時會餵給模型。 |
| <mark>`token_type_ids`</mark> | 部分模型會有，用來區分第一段與第二段。DeBERTa / RoBERTa 類模型不一定需要。 |
| <mark>`offset_mapping`</mark> | 每個 token 對應回原始文字的 character range。建立 label 與 validation decode 時使用。 |

## Label Building

label building 使用：

```text
char_spans + offset_mapping
```

規則：

```text
special token / padding / feature_text token -> label = -100
pn_history token 且與任一 char_span 重疊 -> label = 1
pn_history token 但沒有與 char_span 重疊 -> label = 0
```

<mark>`-100`</mark> 代表 loss 計算時忽略該 token。

例子：

```text
char_spans: [(203, 217)]

token: chest     offset: (203, 208) -> label = 1
token: pressure  offset: (209, 217) -> label = 1
token: denies    offset: (400, 406) -> label = 0
```

注：<mark>`offset_mapping`</mark> 是在 tokenization 時產生。訓練時模型不需要直接吃 <mark>`offset_mapping`</mark>，但 validation 後處理一定會用到，所以 dataset 需要保留。

## Training

每個 fold 的切法：

```python
train_df = df[df["fold"] != k]
valid_df = df[df["fold"] == k]
```

模型訓練時主要輸入：

```text
input_ids
attention_mask
labels
```

如果模型需要，也可以包含：

```text
token_type_ids
```

不直接餵給模型，但需要保留給 validation / debug 的欄位：

```text
id
pn_history
feature_text
char_spans
offset_mapping
case_num
pn_num
feature_num
```

## Validation 後處理

validation inference 後，模型會輸出每個 token 的 logits。   
流程：

```text
logits
↓
softmax / sigmoid
↓
token probabilities
↓
threshold
↓
predicted token labels
↓
offset_mapping
↓
predicted character spans
```

常見 threshold：

```text
0.5
```

但實際應該在 validation 上調整。

decode 時需要做：

- 把預測為答案的 token 用 <mark>`offset_mapping`</mark> 轉回 character span。
- 相鄰或中間只隔空白的 token 合併成同一段 span。
- 多個不連續答案用分號串起來。
- 沒有預測答案時輸出空字串。

輸出格式範例：

```text
203 217
203 217;300 315
```

validation scoring 建議使用 character-level F1 (micro F1)：

```text
TP = 預測且正確的 character
FP = 預測但不該預測的 character
FN = 沒預測到但應該預測的 character
```

## Test Inference 與 Submission

test 資料需要先 merge：

```text
test.csv
+ patient_notes.csv
+ features.csv
```

得到和訓練 inference 一致的欄位：

```text
id
case_num
pn_num
feature_num
pn_history
feature_text
```

test 沒有：

```text
annotation_text
char_spans
fold
```

所以 test 只做：

```text
tokenization
↓
model inference
↓
probabilities
↓
threshold
↓
offset_mapping decode
↓
location string
```

submission 格式：

```csv
id,location
00016_000,203 217
00016_001,
00016_002,100 110;150 165
```

最後輸出：

- <mark>`submission.csv`</mark>

## Ensemble

ensemble 在 token probability 階段平均：

```text
model_1 probabilities
model_2 probabilities
model_3 probabilities
↓
average probabilities
↓
threshold
↓
decode to character spans
```

平均 token probabilities 保留更多 token-level 資訊通常比起各模型先 decode 成 span 後再投票更穩。

## 整體流程

```text
train_preprocessed_5fold.pkl
│
├─ 欄位：
│  id
│  case_num
│  pn_num
│  feature_num
│  pn_history
│  feature_text
│  annotation_text
│  char_spans
│  fold
│
↓
選擇 backbone
DeBERTa / RoBERTa / ClinicalBERT
│
↓
選擇任務方法
token classification
│
↓
根據 fold 切 train / valid
train = fold != k
valid = fold == k
│
↓
Tokenization
使用 feature_text + pn_history
產生 input_ids、attention_mask、offset_mapping
│
↓
Label Building
使用 char_spans + offset_mapping
產生 token-level labels
│
↓
Model Training
input_ids + attention_mask → model
token-level labels → calculate loss
│
↓
Validation Inference
輸出 token probabilities
│
↓
Decoding
token probabilities → predicted token spans
│
↓
Character Mapping
predicted token spans + offset_mapping → predicted location
│
↓
Post-processing
修正空白、標點、span 邊界、多段 span
│
↓
Validation Score / Threshold Tuning
│
↓
Test Inference
│
↓
Submission
id + predicted character-level location
```
