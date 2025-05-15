import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Đọc dữ liệu từ file CSV
file_path = "train.csv"  # Đảm bảo file train.csv nằm trong thư mục dự án của PyCharm
try:
    df = pd.read_csv(file_path)
    print(f"Dữ liệu có {df.shape[0]} hàng và {df.shape[1]} cột.\n")

    # Cấu hình hiển thị trong PyCharm
    pd.set_option('display.max_columns', None)  # Hiển thị tất cả các cột
    pd.set_option('display.width', 1000)  # Điều chỉnh chiều rộng hiển thị
    pd.set_option('display.max_rows', 20)  # Giới hạn số dòng hiển thị

    # Hiển thị 5 dòng đầu và cuối
    print("📌 5 dòng đầu của dữ liệu:")
    print(df.head(), "\n")

    print("📌 5 dòng cuối của dữ liệu:")
    print(df.tail(), "\n")

    # Hiển thị thông tin tổng quan về DataFrame
    print("📊 Thông tin chi tiết về dữ liệu:")
    df.info()

    # Chuyển đổi cột "age" sang kiểu số nguyên
    if "age" in df.columns:
        df["age"] = df["age"].astype(int)
        print("\n✅ Đã chuyển đổi cột 'age' sang kiểu số nguyên!")

    # Hiển thị 2 dòng đầu sau khi chuyển đổi kiểu dữ liệu
    print("\n📌 2 dòng đầu sau khi chỉnh sửa kiểu dữ liệu:")
    print(df.head(2), "\n")

    # Duyệt qua các cột và in ra các giá trị duy nhất của các cột không phải là số
    numerical_features = {"ID", "age", "result"}  # Dùng set để tìm nhanh hơn
    print("📊 Các cột có dữ liệu phân loại (Categorical Features):\n")

    for col in df.columns:
        if col not in numerical_features:
            unique_values = df[col].unique()
            print(f"🔹 Cột: {col}")
            print(unique_values)
            print("-" * 50)
    # Xóa các cột "ID" và "age_desc" nếu chúng tồn tại trong DataFrame
    columns_to_drop = ["ID", "age_desc"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Hiển thị thông tin sau khi xóa cột
    print(f"\n✅ Đã xóa các cột {columns_to_drop} (nếu có).")
    print(f"📊 Kích thước DataFrame sau khi xóa cột: {df.shape}\n")

    # Hiển thị 2 dòng đầu tiên sau khi xóa cột
    print("📌 2 dòng đầu của dữ liệu sau khi xóa cột:")
    print(df.head(2), "\n")

    # Hiển thị danh sách tên cột hiện tại
    print("📜 Danh sách các cột còn lại trong DataFrame:")
    print(df.columns.tolist())
    # Kiểm tra xem cột 'contry_of_res' có tồn tại không
    column_name = "contry_of_res"
    if column_name in df.columns:
        print(f"📌 Các giá trị duy nhất trong cột '{column_name}' trước khi thay đổi:")
        print(df[column_name].unique(), "\n")

        # Định nghĩa dictionary để chuẩn hóa tên quốc gia
        mapping = {
            "Viet Nam": "Vietnam",
            "AmericanSamoa": "United States",
            "Hong Kong": "China"
        }

        # Thay thế giá trị trong cột 'contry_of_res' theo mapping
        df[column_name] = df[column_name].replace(mapping)

        # Hiển thị các giá trị duy nhất sau khi thay thế
        print(f"✅ Đã cập nhật tên quốc gia trong cột '{column_name}'.")
        print("📌 Các giá trị duy nhất sau khi thay đổi:")
        print(df[column_name].unique(), "\n")
    else:
        print(f"⚠️ Lỗi: Cột '{column_name}' không tồn tại trong DataFrame!")
    # Kiểm tra phân phối của cột "Class/ASD"
    if "Class/ASD" in df.columns:
        print("\n📊 Phân phối của cột 'Class/ASD':")
        print(df["Class/ASD"].value_counts(), "\n")
    else:
        print("⚠️ Cột 'Class/ASD' không tồn tại trong DataFrame!")

    # Hiển thị kích thước DataFrame
    print(f"📏 Kích thước của DataFrame: {df.shape}\n")

    # Hiển thị danh sách các cột
    print("📜 Danh sách cột trong DataFrame:")
    print(df.columns.tolist(), "\n")

    # Hiển thị 2 dòng đầu của DataFrame
    print("📌 2 dòng đầu của dữ liệu:")
    print(df.head(2), "\n")

    # Thống kê mô tả dữ liệu
    print("📊 Mô tả dữ liệu:")
    print(df.describe(), "\n")

    # Thiết lập giao diện đồ thị
    sns.set_theme(style="darkgrid")

    # Biểu đồ phân phối độ tuổi
    if "age" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["age"], kde=True)
        plt.title("Phân phối độ tuổi")

        # Tính trung bình và trung vị
        age_mean = df["age"].mean()
        age_median = df["age"].median()

        print(f"📌 Độ tuổi trung bình: {age_mean:.2f}")
        print(f"📌 Độ tuổi trung vị: {age_median:.2f}")

        # Thêm đường dọc cho giá trị trung bình và trung vị
        plt.axvline(age_mean, color="red", linestyle="--", label="Mean")
        plt.axvline(age_median, color="green", linestyle="-", label="Median")

        plt.legend()
        plt.show()
    else:
        print("⚠️ Cột 'age' không tồn tại trong DataFrame!")

    # Biểu đồ phân phối kết quả "result"
    if "result" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["result"], kde=True)
        plt.title("Phân phối kết quả (result)")

        # Tính trung bình và trung vị
        result_mean = df["result"].mean()
        result_median = df["result"].median()

        print(f"📌 Giá trị trung bình của 'result': {result_mean:.2f}")
        print(f"📌 Giá trị trung vị của 'result': {result_median:.2f}")

        # Thêm đường dọc cho giá trị trung bình và trung vị
        plt.axvline(result_mean, color="red", linestyle="--", label="Mean")
        plt.axvline(result_median, color="green", linestyle="-", label="Median")

        plt.legend()
        plt.show()
    else:
        print("⚠️ Cột 'result' không tồn tại trong DataFrame!")
    # Biểu đồ Box Plot cho "age"
    if "age" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["age"])
        plt.title("Box Plot for Age")
        plt.xlabel("Age")
        plt.show()
    else:
        print("⚠️ Cột 'age' không tồn tại trong DataFrame!")

    # Biểu đồ Box Plot cho "result"
    if "result" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["result"])
        plt.title("Box Plot for Result")
        plt.xlabel("Result")
        plt.show()
    else:
        print("⚠️ Cột 'result' không tồn tại trong DataFrame!")

    # Xác định Outliers bằng phương pháp IQR cho cột "age"
    if "age" in df.columns:
        Q1 = df["age"].quantile(0.25)
        Q3 = df["age"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        age_outliers = df[(df["age"] < lower_bound) | (df["age"] > upper_bound)]
        print(f"📌 Số lượng outliers trong 'age': {len(age_outliers)}")
    else:
        print("⚠️ Cột 'age' không tồn tại trong DataFrame!")

    # Xác định Outliers bằng phương pháp IQR cho cột "result"
    if "result" in df.columns:
        Q1 = df["result"].quantile(0.25)
        Q3 = df["result"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        result_outliers = df[(df["result"] < lower_bound) | (df["result"] > upper_bound)]
        print(f"📌 Số lượng outliers trong 'result': {len(result_outliers)}")
    else:
        print("⚠️ Cột 'result' không tồn tại trong DataFrame!")

    # Hiển thị danh sách cột
    print("\n📜 Danh sách cột trong DataFrame:")
    print(df.columns.tolist(), "\n")
    # Danh sách các cột phân loại (Categorical Columns)
    categorical_columns = [
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
        'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'gender',
        'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before',
        'relation'
    ]

    # Vẽ biểu đồ đếm (Count Plot) cho từng cột
    for col in categorical_columns:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df[col], hue=df[col], palette="pastel", legend=False)
            plt.title(f"Count Plot for {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"⚠️ Cột '{col}' không tồn tại trong DataFrame!")
    # Biểu đồ Count Plot cho cột target "Class/ASD"
    if "Class/ASD" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df["Class/ASD"], hue=df["Class/ASD"], palette="pastel", legend=False)
        plt.title("Count Plot for Class/ASD")
        plt.xlabel("Class/ASD")
        plt.ylabel("Count")
        plt.show()

        # Hiển thị số lượng giá trị trong cột "Class/ASD"
        print("\n📌 Phân phối dữ liệu trong cột 'Class/ASD':")
        print(df["Class/ASD"].value_counts(), "\n")
    else:
        print("⚠️ Cột 'Class/ASD' không tồn tại trong DataFrame!")

    # Chuẩn hóa giá trị trong cột "ethnicity"
    if "ethnicity" in df.columns:
        df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
        print("\n📜 Giá trị duy nhất trong cột 'ethnicity' sau khi thay thế:")
        print(df["ethnicity"].unique(), "\n")
    else:
        print("⚠️ Cột 'ethnicity' không tồn tại trong DataFrame!")

    # Chuẩn hóa giá trị trong cột "relation"
    print("📜 Gía trị trong cột relation: "+df["relation"].unique())
    if "relation" in df.columns:
        df["relation"] = df["relation"].replace(
            {"?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"}
        )
        print("\n📜 Giá trị duy nhất trong cột 'relation' sau khi thay thế:")
        print(df["relation"].unique(), "\n")
    else:
        print("⚠️ Cột 'relation' không tồn tại trong DataFrame!")

    # Hiển thị 5 dòng đầu tiên của DataFrame
    print("\n📋 Xem trước dữ liệu sau khi cập nhật:")
    print(df.head(), "\n")
    # Xác định các cột có kiểu dữ liệu "object" (chuỗi ký tự)
    object_columns = df.select_dtypes(include=["object"]).columns
    print("\n📝 Các cột có kiểu dữ liệu 'object':")
    print(object_columns, "\n")

    # Kiểm tra xem có cột nào cần mã hóa không
    if len(object_columns) == 0:
        print("✅ Không có cột nào cần Label Encoding!")
    else:
        # Tạo dictionary để lưu LabelEncoders
        encoders = {}

        # Áp dụng Label Encoding cho từng cột
        for column in object_columns:
            print(f"🔄 Encoding cột: {column}")
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(
                df[column].astype(str))  # Chuyển đổi dữ liệu về dạng chuỗi để tránh lỗi NaN
            encoders[column] = label_encoder  # Lưu encoder để sử dụng lại sau này

        # Lưu các encoders vào file pickle
        with open("encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        print("\n💾 Encoders đã được lưu vào 'encoders.pkl'!")

    # Hiển thị 5 dòng đầu của DataFrame sau khi mã hóa
    print("\n📋 Xem trước dữ liệu sau khi Label Encoding:")
    print(df.head(), "\n")
    # Hiển thị ma trận tương quan
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


    # Hàm thay thế outliers bằng giá trị trung vị (median)
    def replace_outliers_with_median(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        median = df[column].median()

        # Thay thế outliers bằng median
        df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
        print(f"✅ Đã thay thế outliers trong cột {column} bằng giá trị trung vị ({median})")
        return df


    # Áp dụng hàm cho cột "age" và "result"
    df = replace_outliers_with_median(df, "age")
    df = replace_outliers_with_median(df, "result")

    # Kiểm tra thông tin dữ liệu sau khi xử lý
    print(df.head(), "\n")
    print("\n📊 Kích thước DataFrame sau khi xử lý outliers:", df.shape)
    print("📝 Các cột trong DataFrame:", df.columns.tolist(), "\n")

    # Chia dữ liệu thành X (đặc trưng) và y (nhãn mục tiêu)
    X = df.drop(columns=["Class/ASD"])  # Biến độc lập
    y = df["Class/ASD"]  # Biến phụ thuộc (nhãn)

    # Hiển thị mẫu dữ liệu đầu ra
    print("\n🎯 X - Đặc trưng đầu vào:")
    print(X.head())

    print("\n✅ y - Nhãn mục tiêu:")
    print(y.head())

    # ✂️ Chia tập dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n📊 Kích thước tập train & test:")
    print(f"✅ y_train: {y_train.shape}, y_test: {y_test.shape}")

    # 🔍 Kiểm tra số lượng mẫu trong từng lớp trước khi dùng SMOTE
    print("\n🎯 Phân bố dữ liệu trước khi áp dụng SMOTE:")
    print(y_train.value_counts())
    print(y_test.value_counts())

    # 🆙 Áp dụng SMOTE để cân bằng dữ liệu
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\n📊 Kích thước tập train sau khi áp dụng SMOTE:", y_train_smote.shape)
    print("\n✅ Phân bố dữ liệu sau khi SMOTE:")
    print(y_train_smote.value_counts())

    # 🚀 Danh sách mô hình sẽ huấn luyện
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }

    # 📌 Lưu kết quả Cross Validation
    cv_scores = {}

    # 🏋️ Huấn luyện mô hình với 5-fold Cross Validation
    for model_name, model in models.items():
        print(f"\n🔹 Training {model_name} with default parameters...")
        scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")

        # Lưu kết quả vào dictionary
        cv_scores[model_name] = scores
        print(f"✅ {model_name} Cross-Validation Accuracy: {np.mean(scores):.2f}")
        print("-" * 50)

    # 🏆 Hiển thị kết quả Cross Validation của tất cả mô hình
    print("\n🎯 Cross-Validation Accuracy Scores:")
    for model, score in cv_scores.items():
        print(f"{model}: {np.mean(score):.2f} ± {np.std(score):.2f}")
    # 🚀 Khởi tạo mô hình
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    xgboost_classifier = XGBClassifier(random_state=42)

    # 📌 Thiết lập danh sách hyperparameter cho từng mô hình
    param_grid = {
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30, 50, 70],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "subsample": [0.5, 0.7, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0]
        }
    }

    # 🏆 Lưu kết quả tìm kiếm
    best_models = {}
    best_params = {}

    # 🏋️ Huấn luyện RandomizedSearchCV cho từng mô hình
    models = {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgboost_classifier
    }

    for model_name, model in models.items():
        print(f"\n🔹 Running RandomizedSearchCV for {model_name}...")

        # Khởi tạo RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid[model_name],
            n_iter=20,  # Số lần thử nghiệm
            cv=5,  # 5-Fold Cross Validation
            scoring="accuracy",
            random_state=42,
            n_jobs=-1  # Chạy song song trên nhiều CPU
        )

        # Huấn luyện mô hình
        random_search.fit(X_train_smote, y_train_smote)

        # Lưu mô hình & tham số tốt nhất
        best_models[model_name] = random_search.best_estimator_
        best_params[model_name] = random_search.best_params_

        # Hiển thị kết quả
        print(f"\n✅ {model_name} Best Score: {random_search.best_score_:.4f}")
        print(f"📌 Best Parameters: {random_search.best_params_}")

    # 🏆 Hiển thị tất cả kết quả sau khi tìm kiếm
    print("\n🎯 Best Models & Parameters:")
    for model, params in best_params.items():
        print(f"{model}: {params}")

    # 📌 Chọn mô hình có độ chính xác tốt nhất từ RandomizedSearchCV
    best_model_name = None
    best_model = None
    best_score = 0

    # 📌 Tìm model có điểm Cross-Validation cao nhất
    for model_name, model in best_models.items():
        score = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy").mean()

        if score > best_score:
            best_model_name = model_name
            best_model = model
            best_score = score

    # 🔥 Hiển thị model tốt nhất
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🎯 Best Cross-Validation Accuracy: {best_score:.4f}")
    print(f"📌 Best Parameters: {best_params[best_model_name]}")
    if best_model is not None:
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("✅ Best model saved successfully as best_model.pkl")
    else:
        print("❌ No best model found. Check your model selection process.")
    # Đánh giá mô hình trên dữ liệu kiểm tra
    y_test_pred = best_model.predict(X_test)

    # In các kết quả đánh giá
    print("Accuracy score:")
    print(accuracy_score(y_test, y_test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Dự đoán khi đưa vào dữ liệu mới
    def input_manual_data():
        """
        Hàm nhập dữ liệu thủ công từ bàn phím
        Trả về DataFrame chứa dữ liệu đã nhập (không bao gồm age_desc)
        """
        print("\n" + "=" * 50)
        print("NHẬP THÔNG TIN ĐỂ DỰ ĐOÁN AUTISM SPECTRUM DISORDER")
        print("=" * 50)

        # Tạo dictionary để lưu dữ liệu
        data = {
            'A1_Score': [], 'A2_Score': [], 'A3_Score': [], 'A4_Score': [], 'A5_Score': [],
            'A6_Score': [], 'A7_Score': [], 'A8_Score': [], 'A9_Score': [], 'A10_Score': [],
            'age': [], 'gender': [], 'ethnicity': [], 'jaundice': [], 'austim': [],
            'contry_of_res': [], 'used_app_before': [], 'result': [], 'relation': []
        }

        # Nhập thông tin cơ bản
        print("\n--- Thông tin cá nhân ---")
        data['age'].append(float(input("• Tuổi (vd: 25.5): ")))
        data['gender'].append(input("• Giới tính (m/f): ").lower())
        data['ethnicity'].append(input("• Dân tộc (vd: White-European, Asian, Middle Eastern, ?): "))
        data['jaundice'].append(input("• Tiền sử vàng da (yes/no): ").lower())
        data['austim'].append(input("• Tiền sử gia đình có autism (yes/no): ").lower())
        data['contry_of_res'].append(input("• Quốc gia cư trú (vd: Vietnam, United States): "))
        data['used_app_before'].append(input("• Đã từng dùng ứng dụng ASD trước đây (yes/no): ").lower())
        data['result'].append(float(input("• Điểm kết quả test (từ -15 đến 15, vd: 5.2): ")))
        data['relation'].append(input("• Mối quan hệ với bệnh nhân (vd: Self, Parent, ?): "))

        # Nhập điểm các câu hỏi A1-A10
        print("\n--- Điểm các câu hỏi (0 hoặc 1) ---")
        for i in range(1, 11):
            while True:
                try:
                    score = int(input(f"• A{i}_Score (0/1): "))
                    if score in [0, 1]:
                        data[f'A{i}_Score'].append(score)
                        break
                    else:
                        print("⚠️ Chỉ nhập 0 hoặc 1!")
                except ValueError:
                    print("⚠️ Vui lòng nhập số nguyên!")

        return pd.DataFrame(data)


    def preprocess_input_data(df, encoders):
        """
        Tiền xử lý dữ liệu nhập vào giống với dữ liệu train
        Đảm bảo feature names khớp với khi huấn luyện
        """
        # Chuẩn hóa giá trị
        if "contry_of_res" in df.columns:
            mapping = {
                "Viet Nam": "Vietnam",
                "AmericanSamoa": "United States",
                "Hong Kong": "China"
            }
            df["contry_of_res"] = df["contry_of_res"].replace(mapping)

        if "ethnicity" in df.columns:
            df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})

        if "relation" in df.columns:
            df["relation"] = df["relation"].replace({
                "?": "Others",
                "Relative": "Others",
                "Parent": "Others",
                "Health care professional": "Others"
            })

        # Đảm bảo chỉ giữ lại các cột cần thiết
        expected_features = [
            'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'austim',
            'contry_of_res', 'used_app_before', 'result', 'relation'
        ]

        # Loại bỏ các cột không có trong training data
        df = df[expected_features]

        # Label Encoding
        object_columns = df.select_dtypes(include=["object"]).columns
        for column in object_columns:
            if column in encoders:
                df[column] = encoders[column].transform(df[column].astype(str))

        return df


    def predict_asd():
        """
        Hàm chính để dự đoán ASD từ dữ liệu nhập tay
        """
        try:
            # Load model và encoder
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)

            # Nhập dữ liệu
            input_df = input_manual_data()

            # Tiền xử lý
            processed_df = preprocess_input_data(input_df, encoders)

            # Đảm bảo thứ tự cột giống khi huấn luyện
            if hasattr(model, 'feature_names_in_'):
                processed_df = processed_df[model.feature_names_in_]

            # Dự đoán
            prediction = model.predict(processed_df)
            proba = model.predict_proba(processed_df)[:, 1][0]

            # Hiển thị kết quả
            print("\n" + "=" * 50)
            print("KẾT QUẢ DỰ ĐOÁN")
            print("=" * 50)
            print(f"👉 Kết luận: {'CÓ nguy cơ mắc ASD' if prediction[0] == 1 else 'KHÔNG có nguy cơ mắc ASD'}")
            print(f"👉 Độ tin cậy: {proba * 100:.2f}%")
            print("=" * 50)

            # Giải thích thêm
            if prediction[0] == 1:
                print(
                    "\nLưu ý: Kết quả này chỉ mang tính tham khảo. Hãy đến gặp chuyên gia y tế để được chẩn đoán chính xác.")
            else:
                print(
                    "\nLưu ý: Dù kết quả là không có nguy cơ, nhưng nếu có biểu hiện bất thường vẫn nên tham khảo ý kiến bác sĩ.")

        except FileNotFoundError:
            print("\n⚠️ Lỗi: Không tìm thấy file model hoặc encoder! Hãy chắc chắn bạn đã huấn luyện mô hình trước.")
        except Exception as e:
            print(f"\n⚠️ Lỗi: {str(e)}")


    # Chạy chương trình
    if __name__ == "__main__":
        print("CHƯƠNG TRÌNH DỰ ĐOÁN NGUY CƠ TỰ KỶ (ASD)")
        print("----------------------------------------")

        while True:
            predict_asd()

            tiep_tuc = input("\nBạn muốn dự đoán tiếp không? (y/n): ").lower()
            if tiep_tuc != 'y':
                print("\nCảm ơn đã sử dụng chương trình!")
                break

except FileNotFoundError:
    print(f"⚠️ Lỗi: Không tìm thấy file {file_path}. Hãy kiểm tra lại đường dẫn!")
except ValueError as e:
    print(f"❌ Lỗi: Không thể chuyển đổi cột 'age' sang kiểu int. Kiểm tra dữ liệu có giá trị không hợp lệ!")
    print(f"Chi tiết lỗi: {e}")

