from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece as spm

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# 英译中
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# 编写程序调用翻译模型
def translate(text):
    # text = "我爱你"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# 测试
print(translate("我爱你"))
print(translate("你好"))
print(translate("你好，我是小明"))
print(translate("你好，我是小明，我来自中国"))
print(translate("你好，我是小明，我来自中国，我喜欢吃火锅"))

# 输出
# ['I love you']
# ['Hello']
# ['Hello, I am Xiao Ming']
# ['Hello, I am Xiao Ming, I come from China']
# ['Hello, I am Xiao Ming, I come from China, I like to eat hot pot']



