from transformers import AutoTokenizer

# モデル名にゃ
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# トークナイザーの読み込みにゃ
tokenizer = AutoTokenizer.from_pretrained(model_name)

# テストする日本語文にゃ
sentence = input("トークン数を数えたい文を入力してにゃ：")

# トークン化（input_ids に変換）して数えるにゃ
tokens = tokenizer(sentence, return_tensors="pt")
token_ids = tokens["input_ids"][0]
token_count = len(token_ids)

# 結果表示にゃ
print(f"文：{sentence}")
print(f"トークン数：{token_count}")
print(f"トークンID：{token_ids.tolist()}")