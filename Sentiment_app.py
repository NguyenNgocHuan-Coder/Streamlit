import streamlit as st
import pickle
import re, regex
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from scipy.sparse import csr_matrix, hstack

# ========== Load mô hình và vectorizer từ .pkl ==========
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== Load các dictionary từ file txt ==========
def load_dict_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return dict(line.split('\t') for line in lines if '\t' in line)

def load_list_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

emoji_dict = load_dict_from_txt("emojicon.txt")
teen_dict = load_dict_from_txt("teencode.txt")
wrong_lst = load_list_from_txt("wrong-word.txt")
stopwords_lst = load_list_from_txt("vietnamese-stopwords.txt")
positive_words = load_list_from_txt("positive_VN.txt")
negative_words = load_list_from_txt("negative_VN.txt")
positive_emojis = load_list_from_txt("positive_emoji.txt")
negative_emojis = load_list_from_txt("negative_emoji.txt")

# ========== Tiền xử lý ==========
def covert_unicode(txt):
    dicchar = {
        "à":"à", "á":"á", "ả":"ả", "ã":"ã", "ạ":"ạ", "â":"â", "ầ":"ầ", "ấ":"ấ", "ẩ":"ẩ", "ẫ":"ẫ", "ậ":"ậ",
        "ă":"ă", "ằ":"ằ", "ắ":"ắ", "ẳ":"ẳ", "ẵ":"ẵ", "ặ":"ặ", "è":"è", "é":"é", "ẻ":"ẻ", "ẽ":"ẽ", "ẹ":"ẹ",
        "ê":"ê", "ề":"ề", "ế":"ế", "ể":"ể", "ễ":"ễ", "ệ":"ệ", "ì":"ì", "í":"í", "ỉ":"ỉ", "ĩ":"ĩ", "ị":"ị",
        "ò":"ò", "ó":"ó", "ỏ":"ỏ", "õ":"õ", "ọ":"ọ", "ô":"ô", "ồ":"ồ", "ố":"ố", "ổ":"ổ", "ỗ":"ỗ", "ộ":"ộ",
        "ơ":"ơ", "ờ":"ờ", "ớ":"ớ", "ở":"ở", "ỡ":"ỡ", "ợ":"ợ", "ù":"ù", "ú":"ú", "ủ":"ủ", "ũ":"ũ", "ụ":"ụ",
        "ư":"ư", "ừ":"ừ", "ứ":"ứ", "ử":"ử", "ữ":"ữ", "ự":"ự", "ỳ":"ỳ", "ý":"ý", "ỷ":"ỷ", "ỹ":"ỹ", "ỵ":"ỵ"
    }
    return ''.join(dicchar.get(c, c) for c in txt)

def normalize_repeated_characters(text):
    return re.sub(r'(.)\1+', r'\1', text)

def process_text(text):
    document = text.lower().replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        sentence = ''.join(emoji_dict.get(c, c) for c in sentence)
        sentence = ' '.join(teen_dict.get(w, w) for w in sentence.split())
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        sentence = ' '.join(w for w in sentence.split() if w not in wrong_lst)
        new_sentence += sentence + '. '
    return regex.sub(r'\s+', ' ', new_sentence).strip()

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        words = word_tokenize(sentence, format='text')
        tagged = pos_tag(words)
        filtered = ' '.join(w for w, t in tagged if t in lst_word_type)
        new_document += filtered + ' '
    return regex.sub(r'\s+', ' ', new_document).strip()

def remove_stopword(text):
    return regex.sub(r'\s+', ' ', ' '.join(w for w in text.split() if w not in stopwords_lst)).strip()

def count_sentiment_items(text):
    text = str(text).lower()
    pos_word = sum(1 for word in positive_words if word in text)
    pos_emoji = sum(text.count(emoji) for emoji in positive_emojis)
    neg_word = sum(1 for word in negative_words if word in text)
    neg_emoji = sum(text.count(emoji) for emoji in negative_emojis)
    return pos_word, neg_word, pos_emoji, neg_emoji

# ========== Dự đoán ==========
def predict_sentiment(text_input):
    text = covert_unicode(text_input)
    text = normalize_repeated_characters(text)
    text = process_text(text)
    text = process_postag_thesea(text)
    text = remove_stopword(text)

    tfidf_vector = vectorizer.transform([text])
    pos_word, neg_word, pos_emoji, neg_emoji = count_sentiment_items(text_input)
    numeric_features = scaler.transform([[pos_word, neg_word, pos_emoji, neg_emoji]])
    binary_feature = csr_matrix([[1]])
    final_features = hstack([tfidf_vector, csr_matrix(numeric_features), binary_feature])
    y_pred = model_lr.predict(final_features)[0]
    label = le.inverse_transform([y_pred])[0]
    return label

# ========== Giao diện Streamlit ==========
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📢 Ứng dụng phân tích cảm xúc review công ty")

input_text = st.text_area("✍️ Nhập câu đánh giá của bạn:", height=150)

if st.button("🚀 Dự đoán cảm xúc"):
    if not input_text.strip():
        st.warning("⛔ Vui lòng nhập nội dung review!")
    else:
        with st.spinner("🔍 Đang xử lý..."):
            result = predict_sentiment(input_text)
        st.success(f"✅ Kết quả dự đoán: **{result.upper()}**")
