import streamlit as st
import pickle
import re, regex
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from scipy.sparse import csr_matrix, hstack

# ========== Sidebar Menu ==========
st.sidebar.title("📚 Menu")
menu_choice = st.sidebar.radio("Chọn chức năng:", (
    "📌 Business Objective",
    "🏗️ Build Model",
    "💬 Sentiment Analysis",
    "🧩 Information Clustering"
))
# ========== Load mô hình và vectorizer từ .pkl ==========
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== Load dictionary ==========
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
    return txt.encode('utf-8').decode('utf-8')

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
        sentence = ' '.join(
            word[0] if word[1] in lst_word_type else ''
            for word in pos_tag(word_tokenize(sentence, format="text"))
        )
        new_document += sentence + ' '
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
def predict_sentiment(text_input, recommend_num):
    text = covert_unicode(text_input)
    text = normalize_repeated_characters(text)
    text = process_text(text)
    text = process_postag_thesea(text)
    text = remove_stopword(text)

    tfidf_vector = vectorizer.transform([text])
    pos_word, neg_word, pos_emoji, neg_emoji = count_sentiment_items(text_input)
    numeric_features = scaler.transform([[pos_word, neg_word, pos_emoji, neg_emoji]])
    recommend_feature = csr_matrix([[recommend_num]])

    final_features = hstack([tfidf_vector, csr_matrix(numeric_features), recommend_feature])
    y_pred = model_lr.predict(final_features)[0]
    label = le.inverse_transform([y_pred])[0]
    return label


# ========== Các Trang Ứng Dụng ==========
if menu_choice == "📌 Business Objective":
    st.title("📌 Business Objective: Sentiment Analysis and Information Clustering")
    st.markdown("""
    #### Mục tiêu của đồ án:
    
    - **Sentiment Analysis**: Xây dựng mô hình phân loại cảm xúc từ các đánh giá của nhân viên/ứng viên về công ty trên ITviec (Tích cực / Trung tính / Tiêu cực). Giúp công ty nắm bắt được tâm lý người lao động.

    - **Information Clustering**: Phân cụm các đánh giá để xác định đặc điểm nổi bật của từng nhóm công ty, từ đó đề xuất các cải tiến để giữ chân nhân viên và nâng cao trải nghiệm ứng viên.

    #### Ứng dụng:
    
    - Hệ thống đánh giá nội bộ cho các công ty
    - Công cụ gợi ý cải thiện môi trường làm việc
    - Tự động phân tích hàng loạt đánh giá từ nền tảng tuyển dụng
    """)

elif menu_choice == "🏗️ Build Model":
    st.title("🏗️ Build Model")
    st.write("### Sentiment Analysis")
    st.write("##### 1. Data EDA")
    st.image("Sentiment_EDA.JPG")
    st.image("Clustering_EDA.JPG")
    st.write("##### 2. Visualize")
    st.image("sentiment_distributed_data.JPG")
    st.write("##### 3. Build model and Evaluation")
    st.write("###### Chọn 3 model Logistic Regression , Random Forest , Decision Tree")
    st.write("###### Đánh giá kết quả dựa trên Presicion , ReCall , F1-Score , Accuracy")
    st.image("sentiment_evaluation.JPG")
    st.write("###### Confusion Matrix")
    st.image("Confusion Matrix.JPG")
    st.markdown("Chọn mô hình <span style='color: red; font-weight: bold; text-decoration: underline'>Logistic Regression</span> là tối ưu nhất.",
    unsafe_allow_html=True)

elif menu_choice == "💬 Sentiment Analysis":
    st.title("💬 Ứng dụng phân tích cảm xúc review công ty")

    input_text = st.text_area("✍️ Nhập câu đánh giá của bạn:", height=150)
    recommend_input = st.checkbox("✅ Bạn có recommend công ty này không?", value=True)
    recommend_num = 1 if recommend_input else 0

    if st.button("🚀 Dự đoán cảm xúc"):
        if not input_text.strip():
            st.warning("⛔ Vui lòng nhập nội dung review!")
        else:
            with st.spinner("🔍 Đang xử lý..."):
                result = predict_sentiment(input_text, recommend_num)
            st.success(f"✅ Kết quả dự đoán: **{result.upper()}**")

elif menu_choice == "🧩 Information Clustering":
    st.title("🧩 Information Clustering")
    st.info("🛠️ Phân cụm đánh giá công ty sẽ được cập nhật sau")

