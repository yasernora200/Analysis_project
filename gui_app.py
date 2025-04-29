import streamlit as st
import pandas as pd
from nora import (
    load_data, detect_task_type, prepare_data, get_models,
    get_metrics, train_single_model, evaluate_model,
    save_model, load_model, predict_new_data
)

st.set_page_config(page_title="ML Modeling Tool", layout="wide")
st.title("🔍 Machine Learning Modeling Interface")

# Step 1: Upload data
uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Preview of your data", df.head())

    # Step 2: Select target column
    target_column = st.selectbox("🎯 Select the target column", df.columns)

    # Step 3: Detect task type
    task_type = detect_task_type(df, target_column)
    st.info(f"📌 Detected task type: **{task_type.upper()}**")

    # Step 4: Select model manually
    models = get_models(task_type)
    model_name = st.selectbox("🤖 Select a model", list(models.keys()))
    model = models[model_name]

    # Step 5: Select metric manually
    metrics = get_metrics(task_type)
    metric_name = st.selectbox("📏 Select a metric for evaluation", list(metrics.keys()))
    metric_func = metrics[metric_name]

    # Step 6: Train model
    if st.button("🚀 Train Selected Model"):
        (X_train, X_test, y_train, y_test), feature_columns = prepare_data(df, target_column)
        trained_model = train_single_model(model, X_train, y_train)

        score = evaluate_model(trained_model, X_test, y_test, metric_func)
        st.success(f"✅ {model_name} achieved a {metric_name} of: {score:.4f}")

        save_model(trained_model, feature_columns)
        st.info("💾 Model saved successfully and ready for prediction")

    # Step 7: Prediction with new data
    st.markdown("---")
    st.subheader("📡 Predict using saved model")
    new_data_file = st.file_uploader("Upload new data for prediction (without target column)", type=["csv"], key="predict")

    if new_data_file:
        new_df = pd.read_csv(new_data_file)
        loaded_model, feature_columns = load_model()
        predictions = predict_new_data(loaded_model, feature_columns, new_df)
        st.write("### 🔮 Predictions", pd.DataFrame(predictions, columns=["Prediction"]))
