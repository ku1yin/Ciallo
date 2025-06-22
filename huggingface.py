from transformers import pipeline

# 加载模型
sentiment_analysis = pipeline("sentiment-analysis", model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")

# 进行情感分析
result = sentiment_analysis("我喜欢你")
print(result)