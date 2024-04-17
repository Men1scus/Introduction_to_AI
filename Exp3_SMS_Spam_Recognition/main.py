import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------- 停用词库路径，若有变化请修改 -------------
# Reference: https://github.com/goto456/stopwords/tree/master

# stopwords_path = r'stopwords/baidu_stopwords.txt'
# stopwords_path = r'stopwords/cn_stopwords.txt'
# stopwords_path = r'stopwords/hit_stopwords.txt'
stopwords_path = r'stopwords/scu_stopwords.txt'

# ---------------------------------------------------

def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表，如 ['嘿', '很', '乎', '会', '或']
    """
    stopwords = []
    # ----------- 请完成读取停用词的代码 ------------

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    
    #----------------------------------------------
    
    return stopwords

# 读取停用词
stopwords = read_stopwords(stopwords_path)

# ----------------- 导入相关的库 -----------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,  PowerTransformer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression

# pipline_list用于传给Pipline作为参数
pipeline_list = [
    # --------------------------- 需要完成的代码 ------------------------------
    
    # ========================== 以下代码仅供参考 =============================
# Vectorizer
    # ('cv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    # ('cv', CountVectorizer(ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    # ('cv', CountVectorizer(ngram_range=(1,3), max_features=5000, token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    # ('cv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
	# ('hv',  HashingVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)), # ValueError: Negative values in data passed to MultinomialNB (input X)
    ('tv',  TfidfVectorizer(ngram_range=(1,2), max_df=0.25, token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
	# ('tv',  TfidfVectorizer(ngram_range=(1,3), token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
	# ('tv',  TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
	
# Scaler
    # ('ss',StandardScaler(with_mean=False)),
    # ('mms',MinMaxScaler()), # TypeError: MinMaxScaler does not support sparse input. Consider using MaxAbsScaler instead.
    ('mas', MaxAbsScaler()),
    # ('rs',RobustScaler(with_centering=False)), # 运行效率极低，耗时20分钟仍然未出结果，遂放弃
    # ('pt',PowerTransformer()), # TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.

# Classifier
    # ('classifier', BernoulliNB())
    # ('classifier', MultinomialNB())
	# ('classifier', MultinomialNB(alpha=0.99))
    ('classifier', ComplementNB(alpha=0.25))
    # ('classifier', ComplementNB(alpha=0.99))
	# ('classifier', LogisticRegression())
	
    # ========================================================================
    
    # ------------------------------------------------------------------------
]

# 加载训练好的模型
from sklearn.externals import joblib
# import joblib
# ------- pipeline 保存的路径，若有变化请修改 --------
pipeline_path = 'results/pipeline.model'
# --------------------------------------------------
pipeline = joblib.load(pipeline_path)

def predict(message):
    """
    预测短信短信的类别和每个类别的概率
    param: message: 经过jieba分词的短信，如"医生 拿 着 我 的 报告单 说 ： 幸亏 你 来 的 早 啊"
    return: label: 整数类型，短信的类别，0 代表正常，1 代表恶意
            proba: 列表类型，短信属于每个类别的概率，如[0.3, 0.7]，认为短信属于 0 的概率为 0.3，属于 1 的概率为 0.7
    """
    label = pipeline.predict([message])[0]
    proba = list(pipeline.predict_proba([message])[0])
    
    return label, proba