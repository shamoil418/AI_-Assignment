import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Comparison App",
    page_icon="📱",
    layout="wide"
)

# --- App Title and Description ---
st.title("📱 Teen Phone Addiction: ML Model Analyzer")
st.write(
    "This app analyzes the teen phone addiction dataset, compares various machine learning models, "
    "and generates a summary table with key performance and theoretical metrics."
)

# --- Data Loading and Caching (Modified for Deployment) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data directly from the file path."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df.drop(['ID', 'Name'], axis=1, errors='ignore')
    
    # Bin the Target Variable into 5 Categories based on the 1-10 scale
    bin_edges = [0, 2, 4, 6, 8, 10.1]
    labels = [0, 1, 2, 3, 4]
    class_names_str = ["Very Low", "Low", "Medium", "High", "Very High"]
    
    df['addiction_category'] = pd.cut(df['Addiction_Level'], bins=bin_edges, labels=labels, right=True, include_lowest=True)
    df.dropna(subset=['addiction_category'], inplace=True)
    df['addiction_category'] = df['addiction_category'].astype(int)

    X = df.drop(['Addiction_Level', 'addiction_category'], axis=1)
    y = df['addiction_category'].values

    categorical_features = ['Gender', 'Location', 'School_Grade', 'Phone_Usage_Purpose']
    numerical_features = [col for col in df.columns if col not in categorical_features and col not in ['Addiction_Level', 'addiction_category']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor, class_names_str, X

# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_options = [
        "Logistic Regression", "Decision Tree", "k-Nearest Neighbors (kNN)",
        "Support Vector Machine (SVM)", "Random Forest", "Neural Networks (NNs)", "K-means"
    ]
    selected_models = st.multiselect(
        "Choose ML models to compare",
        options=model_options,
        default=["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
    )

# --- Main App Body ---
if selected_models:
    st.header("📊 Results and Analysis")
    
    uploaded_file = st.file_uploader("📂 Upload teen addiction dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        (X_train, X_test, y_train, y_test), preprocessor, class_names, X_full = load_and_preprocess_data(uploaded_file)
    else:
        st.warning("⚠️ Please upload the dataset to continue.")
        st.stop()
    
    results_list = []

    # --- Supervised Models Evaluation ---
    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-Nearest Neighbors (kNN)": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC(random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Neural Networks (NNs)": MLPClassifier(max_iter=1000, random_state=42)
    }
    
    for name in selected_models:
        if name in supervised_models:
            with st.expander(f"▼ {name} Results", expanded=False):
                model = supervised_models[name]
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = pipeline.predict(X_test)
                prediction_time = time.time() - start_time

                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                y_prob = pipeline.predict_proba(X_test)
                auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                
                model_filename = f'temp_{name}.joblib'
                joblib.dump(pipeline, model_filename)
                memory_kb = os.path.getsize(model_filename) / 1024
                os.remove(model_filename)
                
                eval_metrics_str = (f"Precision: {report['weighted avg']['precision']:.2f}, Recall: {report['weighted avg']['recall']:.2f}, "
                                    f"F-score: {report['weighted avg']['f1-score']:.2f}, AUC-ROC: {auc_roc:.2f}")

                st.subheader("Performance Metrics")
                st.text(eval_metrics_str)
                st.text(f"Training Time: {training_time:.4f}s | Prediction Speed: {prediction_time:.4f}s | Memory Usage: {memory_kb:.2f} KB")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
                st.pyplot(fig)

                results_list.append({
                    'Algorithm': name, 'Training Time': training_time, 'Prediction Speed': prediction_time,
                    'Memory Usage': memory_kb, 'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)': eval_metrics_str
                })
    
    # --- K-Means Evaluation ---
    if "K-means" in selected_models:
        with st.expander("▼ K-means Clustering Results", expanded=False):
            X_processed = preprocessor.fit_transform(X_full)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            
            start_time = time.time()
            cluster_labels = kmeans.fit_predict(X_processed)
            training_time = time.time() - start_time
            silhouette = silhouette_score(X_processed, cluster_labels)

            st.metric("Silhouette Score", f"{silhouette:.4f}")
            st.info("K-means is unsupervised. A higher Silhouette Score (closer to 1) is better.")

            results_list.append({
                'Algorithm': 'K-means', 'Training Time': training_time, 'Prediction Speed': np.nan, 'Memory Usage': np.nan,
                'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)': f"Silhouette Score: {silhouette:.2f}"
            })
            
    # --- Final Summary Table ---
    if results_list:
        st.header("📜 Final Comparison Table")
        
        theoretical_data = {
            'Algorithm': ["Decision Tree", "Logistic Regression", "k-Nearest Neighbors (kNN)", "Support Vector Machine (SVM)", "Neural Networks (NNs)", "Random Forest", "K-means"],
            'Bias–Variance': ["Low Bias, High Variance", "High Bias, Low Variance", "Low Bias, High Variance", "Tunable", "Low Bias, High Variance", "Low Bias, Medium Variance", "N/A"],
            'Data Size Sensitivity': ["Moderate", "Low", "High", "High", "Low", "Low", "Low"],
            'Hyperparameter Sensitivity': ["High", "Low", "High", "Very High", "Very High", "Moderate", "High"],
            'Robustness': ["Moderate", "Low", "Low", "High", "Moderate", "Very High", "Low"],
            'Best-Suited Metrics': ["Accuracy, F1-Score", "Accuracy, AUC-ROC", "Accuracy, F1-Score", "Accuracy, F1-Score", "Accuracy, Log-Loss", "Accuracy, F1-Score", "Silhouette Score"],
            'Remarks': ["Easy to interpret", "Simple & fast", "Slow for prediction", "Powerful but slow to train", "Flexible 'black box'", "Robust & accurate", "Unsupervised grouping"]
        }
        
        results_df = pd.DataFrame(results_list)
        theoretical_df = pd.DataFrame(theoretical_data)
        
        final_df = pd.merge(theoretical_df, results_df, on='Algorithm', how='left')
        
        # Reorder columns to exactly match the desired format
        final_df = final_df[[
            'Algorithm', 'Bias–Variance', 'Data Size Sensitivity', 'Training Time', 'Prediction Speed',
            'Memory Usage', 'Hyperparameter Sensitivity', 'Robustness',
            'Evaluation Metrics (Precesion, Recall, Fscore, AUC-ROC)', 'Best-Suited Metrics', 'Remarks'
        ]]
        
        st.dataframe(final_df)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=convert_df_to_csv(final_df),
            file_name="ml_algorithm_comparison.csv",
            mime="text/csv",
        )

else:
    st.info("Please select one or more models from the sidebar to begin the analysis.")
