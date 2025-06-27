import streamlit as st
import pickle
import re, regex
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# ========== Sidebar Menu ==========
st.sidebar.title("📚 Menu")
menu_choice = st.sidebar.radio("Chọn chức năng:", (
    "📌 Business Objective",
    "🏗️ Build Model",
    "💬 Sentiment Analysis",
    "🧩 Information Clustering"
))
# ========== Thông tin tác giả ==========
st.sidebar.markdown("""---""")
st.sidebar.markdown("""
**🎓 Tác giả đồ án:**

- Nguyễn Ngọc Huân  
  ✉️ *nguyenngochuan992@gmail.com*

- Nguyễn Thị Hoa Thắng  
  ✉️ *thangnth0511@gmail.com*
""")
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
correct_dict = load_list_from_txt("phrase_corrections.txt")
english_dict = load_list_from_txt("english-vnmese.txt")
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
    st.write("##### 2. Visualization")
    st.image("sentiment_distributed_data.JPG")
    st.write("##### 3. Build model and Evaluation")
    st.write("###### - Huấn luyện mô hình phân loại Logistic Regression , Random Forest , Decision Tree")
    st.write("###### - Đánh giá kết quả dựa trên Presicion , ReCall , F1-Score , Accuracy")
    st.image("sentiment_evaluation.JPG")
    st.write("###### Confusion Matrix")
    st.image("Confusion Matrix.JPG")
    st.markdown("Chọn mô hình <span style='color: red; font-weight: bold; text-decoration: underline'>Logistic Regression</span> là tối ưu nhất.",
    unsafe_allow_html=True)
    st.write("### Information Clustering")
    st.write("##### 1. Data EDA")
    st.image("Clustering_EDA.JPG")
    st.write("##### 2. Visualization")
    st.image("Cluster_wordcloud.JPG")
    st.write("##### 3. Build model and Evaluation")
    st.write("###### - Huấn luyện mô hình phân cụm với các thuật toán KMeans, AgglomerativeClustering, SpectralClustering, Birch")
    st.write("###### - Đánh giá kết quả dựa trên Sihouette score")
    st.image("k_evaluation.JPG")
    st.write("###### Trực quan hoá Elbow theo Sihouette score")
    st.image("ellbow.JPG")
    st.write("###### Trực quan hoá Elbow theo Sihouette score")
    st.image("Cluster_distributed.JPG")
    st.markdown(" Kết luận : Chọn mô hình <span style='color: red; font-weight: bold; text-decoration: underline'>KMeans</span> với k=4 là mô hình tối ưu nhất vì:",unsafe_allow_html=True)
    st.markdown(""" 
    - Silhouette Score ≈ 0.75 cao nhất với k=4, rất ổn định.
    - Các điểm còn lại giảm nhẹ nhưng vẫn khá cao → ổn định tốt.
    - Biểu đồ phân cụm (LDA + KMeans): Nhóm dữ liệu được chia rõ ràng, trực quan.
    - Ranh giới giữa các cụm rõ ràng, gần như không có điểm chồng lấn.
    """)
                
    st.write("##### 4. Interpreting and Visualizing Cluster Analysis Results")
    st.write("###### ✅ Chủ đề #1:Bất cập trong đãi ngộ & điều kiện làm việc. Cụm này nhấn mạnh về các yếu tố về lương và phúc lợi , đặc  biệt có đề cập đến vấn đề bất cập là lương_chậm và công nghệ cũ.")
    st.write("###### 🔑 Key words: chính_sách_làm_thêm_giờ, chế_độ_đãi_ngộ, chế_độ_phúc_lợi, giờ_giấc_thoải_mái, lương_chậm, lương_thưởng, sức_khoẻ, văn_phòng_đẹp, công_ty_lớn, đồng_nghiệp_thân_thiện,môi_trường_làm_việc_thân_thiện.")
    st.image("wordcloud_0.JPG")
    st.write("######  ✅ Chủ đề #2: Môi trường & văn hóa doanh nghiệp .Tập trung vào môi trường làm việc, văn hóa công ty, và cơ sở vật chất hỗ trợ nhân viên, đi kèm một số yếu tố về chính sách và lương")
    st.write("###### 🔑 Key words: môi_trường_làm_việc_tốt, môi_trường_làm_việc_thoải_mái, văn_hoá_công_ty, môi_trường_làm_việc_năng_động, văn_hoá_công_ty, đồng_nghiệp_thân_thiện, công_ty_lớn, văn_phòng_đẹp ,bãi_đậu_xe_rộng_rãi, lương_thưởng, chính_sách_làm_thêm_giờ.")
    st.image("wordcloud_1.JPG")
    st.write("######  ✅ Chủ đề #3: Đãi ngộ & cơ hội phát triển . Gần giống cụm 0 nhưng nhấn mạnh thêm vào yếu tố phúc lợi, dự án lớn và mức lương tốt → thể hiện sự quan tâm đến giá trị công việc & đãi ngộ.")
    st.write("###### 🔑 Key words: dự_án_lớn, lương_tốt, lương_ổn,môi_trường_làm_việc_thoải_mái, môi_trường_làm_việc_thân_thiện, chế_độ_phúc_lợi, đồng_nghiệp_thân_thiện, văn_phòng_rộng")
    st.image("wordcloud_2.JPG")
    st.write("######  ✅ Chủ đề #4: Trải nghiệm làm việc tích cực . Cụm này thể hiện rõ yếu tố trải nghiệm làm việc hàng ngày: linh hoạt, văn phòng đẹp, đồng nghiệp vui vẻ, văn hóa tích cực.")
    st.write("###### 🔑 Key words: văn_phòng_đẹp, văn_phòng_rộng_rãi, phong_cảnh_đẹp, chính_sách_làm_thêm_giờ, lương_thưởng, đồng_nghiệp_thân_thiện, môi_trường_làm_việc_tốt, công_ty_lớn, bãi_đậu_xe_rộng_rãi, môi_trường_làm_việc_năng_động, môi_trường_làm_việc_thoải_mái, văn_hóa_công_ty.")
    st.image("wordcloud_3.JPG")
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
    
    try:
        #LOAD EMOJICON
        file = open('emojicon.txt', 'r', encoding="utf8")
        emoji_lst = file.read().split('\n')
        emoji_dict1 = {}
        for line in emoji_lst:
            key, value = line.split('\t')
            emoji_dict1[key] = str(value)
        file.close()
        #################
        #LOAD TEENCODE
        file = open('teencode.txt', 'r', encoding="utf8")
        teen_lst = file.read().split('\n')
        teen_dict1 = {}
        for line in teen_lst:
            key, value = line.split('\t')
            teen_dict1[key] = str(value)
        file.close()

        ###############
        #LOAD TRANSLATE ENGLISH -> VNMESE
        file = open('english-vnmese.txt', 'r', encoding="utf8")
        english_lst = file.read().split('\n')
        english_dict1 = {}
        for line in english_lst:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                key = parts[0]
                value = '\t'.join(parts[1:])  # nếu value có chứa dấu tab thì vẫn giữ nguyên
            english_dict1[key] = value
        file.close()

        ################
        #LOAD wrong words
        file = open('wrong-word.txt', 'r', encoding="utf8")
        wrong_lst1 = file.read().split('\n')
        file.close()

        #################
        #LOAD STOPWORDS
        file = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
        stopwords_lst1 = file.read().split('\n')
        file.close()

        #################
        ##LOAD PHRASE_CORRECTION
        file = open('phrase_corrections.txt', 'r', encoding="utf8")
        correct_lst = file.read().split('\n')
        correct_dict1 = {}
        for line in correct_lst:
            key, value = line.split(':')
            correct_dict1[key] = str(value)
        file.close()
        df = pd.read_excel("Reviews.xlsx", engine="openpyxl")
        df["Review"] = df["What I liked"].fillna("") + " " + df["Suggestions for improvement"].fillna("")
        df = df[["Company Name", "Review"]].dropna()

        # Select box chọn công ty
        company_list_all = sorted(df["Company Name"].dropna().unique())
        selected_company = st.selectbox("🔎 Chọn công ty để phân tích:", company_list_all)

        df = df[df["Company Name"] == selected_company]
        def apply_phrase_correction(sentence, correct_dict1):
            for phrase, corrected in correct_dict1.items():
                # Dùng regex để thay thế cụm từ chính xác (có phân cách bằng dấu cách)
                pattern = r'\b' + regex.escape(phrase) + r'\b'
                sentence = regex.sub(pattern, corrected, sentence)
            return sentence        
        def process_text(text, emoji_dict1, teen_dict1, english_dict1, correct_dict1, wrong_lst1,stopwords_lst1):
            #Chuyển văn bản thành chữ thường
            document = text.lower()
            document = document.replace("’",'')
            document = regex.sub(r'\.+', ".", document)
            new_sentence = ''
            for sentence in sent_tokenize(document):
                #CONVERT EMOJICON
                sentence = ''.join(emoji_dict1[word] + ' ' if word in emoji_dict1 else word for word in list(sentence))

                #CONVERT TEENCODE
                sentence = ' '.join(teen_dict1[word] if word in teen_dict1 else word for word in sentence.split())

                #CONVERT ENGLISH TO VIETNAMESE
                sentence = ' '.join(english_dict1[word] if word in english_dict1 else word for word in sentence.split())

                #DEL Punctuation & Numbers (chỉ giữ từ tiếng Việt, kể cả có dấu)
                pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
                sentence = ' '.join(regex.findall(pattern, sentence))

                #CONVERT PHRASE CORRECTION
                sentence = apply_phrase_correction(sentence, correct_dict1)

                #DEL wrong words
                # sentence = ' '.join(word for word in sentence.split() if word not in wrong_lst1)

                #DEL stop words
                # sentence = ' '.join(word for word in sentence.split() if word not in stopwords_lst1)

                new_sentence = new_sentence + sentence + '. '

            document = new_sentence
            ###### DEL excess blank space
            document = regex.sub(r'\s+', ' ', document).strip()

            return document

        # Tiền xử lý văn bản
        df["Cleaned"] = df['Review'].apply(lambda text: process_text(text, emoji_dict1, teen_dict1, english_dict1, correct_dict1, wrong_lst1,stopwords_lst1))
        #Tách từ
        def work_tokenize(text):
            tokens = word_tokenize(text, format='text')
            return tokens
        df["Cleaned"] = df["Cleaned"].apply(lambda text: work_tokenize(text)) 
        #Nối từ phủ định với từ liền sau nó :
        def merge_negation_words(text):
            pattern = r"\b(không|không_có|chưa|chưa_có|khó|ít|ít_khi|hiếm|thiếu)\s+(\p{L}+)"
            return regex.sub(pattern, r"\1_\2", text)
        df["Cleaned"] = df["Cleaned"].apply(lambda text: merge_negation_words(text))
        def remove_stopwords_and_dedup(text):
            # Tách từ
            words = text.split()

            # Loại bỏ stopwords
            filtered = [word for word in words if word not in stopwords_lst1]

            # Loại bỏ từ/cụm từ trùng nhau liền kề
            deduped = []
            prev_word = None
            for word in filtered:
                if word != prev_word:
                    deduped.append(word)
                prev_word = word

            return " ".join(deduped)   
        df["Cleaned"] = df["Cleaned"].apply(lambda text: remove_stopwords_and_dedup(text))
        def postag_merge(text):
            # Gán nhãn từ loại
            tagged = pos_tag(text)

            # Gộp: danh từ + (tính từ | động từ), hoặc động từ + tính từ
            merged_words = []
            skip = False
            for i in range(len(tagged)):
                if skip:
                    skip = False
                    continue

                word, tag = tagged[i]

                if i + 1 < len(tagged):
                    next_word, next_tag = tagged[i + 1]

                    # Nối danh từ với tính từ hoặc động từ
                    if tag == 'N' and next_tag in {'A', 'V'}:
                        merged_words.append(f"{word}_{next_word}")
                        skip = True
                    # Nối động từ với tính từ
                    # elif tag == 'V' and next_tag == 'A':
                    #     merged_words.append(f"{word}_{next_word}")
                    #     skip = True
                    # else:
                    #     merged_words.append(word)
                else:
                    merged_words.append(word)

            return " ".join(merged_words)
        df["Cleaned"] = df["Cleaned"].apply(lambda text: postag_merge(text))
        df["Cleaned"] = df["Cleaned"].apply(lambda text: apply_phrase_correction(text, correct_dict1))    
        # Vector hóa văn bản
        vectorizer_cluster = CountVectorizer(max_df=0.95, min_df=20)
        X_vec = vectorizer_cluster.fit_transform(df["Cleaned"])

        # Phân cụm với KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_vec)

        # Từ khóa đặc trưng theo cụm
        keywords = vectorizer_cluster.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_keywords = [", ".join([keywords[i] for i in order_centroids[c][:10]]) for c in range(5)]
        df["Top Keywords"] = df["Cluster"].map({i: kw for i, kw in enumerate(cluster_keywords)})

        cluster_id = df["Cluster"].iloc[0]
        top_keywords = df["Top Keywords"].iloc[0]

        st.markdown(f"✅ **Công ty thuộc cụm số:** `{cluster_id}`")
        st.markdown(f"🔑 **Từ khóa đặc trưng của cụm:** {top_keywords}")
        st.markdown(f"📝 Số lượng đánh giá: {df.shape[0]}")

    except Exception as e:
        st.error(f"Lỗi đọc hoặc xử lý dữ liệu: {e}")
