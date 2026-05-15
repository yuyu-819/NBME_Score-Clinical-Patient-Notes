# NBME - Score Clinical Patient Notes 🩺 

競賽目標：從醫學病歷中，自動找出對應的臨床關鍵特徵

## 📁 資料檔案

原始 Kaggle 資料：

- `train.csv` / `test.csv` / `features.csv` / `patient_notes.csv` / `sample_submission.csv`

整理後之訓練集：

- `train_preprocessed_5fold.pkl`

整理後之測試集：

- `test_preprocessed.pkl`

## 🧹 Preprocess 產出欄位

`train_preprocessed_5fold.pkl` 目前包含以下欄位：

| 欄位 | 用途 |
| --- | --- |
| `id` | 每筆資料的唯一 ID。validation decode、錯誤分析、submission 對回資料時使用。 |
| `case_num` | 病例類別。主要用於分 fold、檢查 case 分布，也可做 case-level 分析。 |
| `pn_num` | patient note 編號。分 fold 時避免同一份 note 同時出現在 train / valid。 |
| `feature_num` | feature 編號。與 `case_num` 一起決定要找的 clinical feature。 |
| `pn_history` | patient note 原文。模型要從這段文字中找答案 span。 |
| `feature_text` | 要尋找的 clinical feature。tokenization 時會和 `pn_history` 一起作為模型輸入。 |
| `annotation_text` | 標註文字內容。主要用於 debug / 檢查，不是訓練 label 的主要來源。 |
| `char_spans` | 真實答案在 `pn_history` 裡的 character span，例如 `[(203, 217)]`。建立 token label 與 validation scoring 時使用。 |
| `fold` | 5-fold split 編號。訓練時用 `fold != k`，驗證時用 `fold == k`。 |

訓練核心會用到：

```text
pn_history
feature_text
char_spans
fold
id
```

## 🤖 使用之模型

使用 `token classification`( 可以直接預測每個 token 是否屬於答案 `start/end prediction` 較適合單一連續 span )

因為 NBME 的答案可能有多段、不連續 span，例如：

```text
203 217;300 315
```

不同 backbone：

- `DeBERTa-V3`
- `Bio-Clinical BERT`
- `PubMedBERT`

## 🔤 Tokenization

tokenization 使用：
`feature_text` + `pn_history`
(`feature_text` : 找尋目標， `pn_history` : 內容)

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
| `input_ids` | 模型輸入 token ids。訓練時會餵給模型。 |
| `attention_mask` | 告訴模型哪些位置是有效 token，哪些是 padding。訓練時會餵給模型。 |
| `token_type_ids` | 部分模型會有，用來區分第一段與第二段。DeBERTa / RoBERTa 類模型不一定需要。 |
| `offset_mapping` | 每個 token 對應回原始文字的 character range。建立 label 與 validation decode 時使用。 |

## 🏷️ Label Building
token-level labels ( 訓練時的 ground truth label )

label building 使用：
`char_spans` + `offset_mapping`

規則：

```text
special token / padding / feature_text token -> label = -100（計算loss時呼略）
pn_history token 且與任一 char_span 重疊 -> label = 1
pn_history token 但沒有與 char_span 重疊 -> label = 0
```

例子：

```text
char_spans: [(203, 217)]

token: chest     offset: (203, 208) -> label = 1
token: pressure  offset: (209, 217) -> label = 1
token: denies    offset: (400, 406) -> label = 0
```

注：`offset_mapping` 是在 tokenization 時產生。訓練時模型不需要直接吃 `offset_mapping`，但 validation 後處理一定會用到，所以 dataset 需要保留。

## 🏋️ Training

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
(`token_type_ids`看選的模型需不需要)

## ✅ Validation 後處理

流程：

```text
logits
↓
softmax / sigmoid
↓
token probabilities
↓
threshold
↓   ①
predicted token labels
↓
offset_mapping
↓   ②
predicted character spans
```
validation inference 後，模型會輸出每個 token 的 logits 
threshold 在 validation 上調整 常見 threshold：0.5

*後續優化*  
① 在 validation 上找threshold最佳值
② offset_mapping decode 設計（是否修正空白、使否須去掉span、多段span是否合併）

decode 時需要做：

- 把預測為答案的 token 用 `offset_mapping` 轉回 `character span`。
- 相鄰或中間只隔空白的 token 合併成同一段 span。
- 多個不連續答案用分號串起來。
- 沒有預測答案時輸出空字串。

輸出格式範例：

```text
203 217
203 217;300 315
```

validation scoring 使用 character-level F1 (micro F1)：

```text
TP = 預測且正確的 character
FP = 預測但不該預測的 character
FN = 沒預測到但應該預測的 character
```

## 🧪 Test Inference 與 Submission

test 流程：

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

最後輸出：`submission.csv`

## 🔀 Ensemble

ensemble 在 token probability 階段平均：    
(平均 token probabilities 保留更多 token-level 資訊通常比起各模型先 decode 成 span 後再投票更穩)

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



## 🗺️ 整體流程

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
DeBERTa-V3 / Bio-Clinical BERT / PubMedBERT
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
