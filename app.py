import streamlit as st
st.set_page_config(page_title="足球赔率预测器", layout="centered")

# 足球比赛赔率实时预测器（导入真实数据 + 可视化 + 保存预测 + 上传训练 + 下载记录 + 清空记录）
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
import requests
from io import StringIO

matplotlib.rcParams['font.family'] = 'SimHei'  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

# 自动采集中超数据示例
def fetch_csl_data():
    url = "https://raw.githubusercontent.com/sample-data/csl_sample.csv"
    try:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        df.to_csv("epl_sample.csv", index=False)
        return True
    except:
        return False

if st.sidebar.button("🔄 自动获取中超示例数据"):
    if fetch_csl_data():
        st.sidebar.success("✅ 中超数据采集成功并保存为 epl_sample.csv")
    else:
        st.sidebar.error("❌ 获取失败，请检查网络或链接是否有效")

model_source = "默认示例数据"
uploaded_file = st.file_uploader("📂 上传自定义比赛数据 CSV（可选，用于训练模型）", type=["csv"])
if uploaded_file:
    with open("epl_sample.csv", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ 上传成功，模型将基于该文件重新训练")
    model_source = uploaded_file.name

def load_or_train_model():
    try:
        model = joblib.load("trained_model.pkl")
    except:
        st.warning("未找到模型，正在从CSV重新训练...")
        try:
            df = pd.read_csv("epl_sample.csv")
            df = df.dropna()
            label_map = {'H': 0, 'D': 1, 'A': 2}
            df['result_label'] = df['result'].map(label_map)
            features = ['home_team_rank', 'away_team_rank',
                        'home_team_goals_avg', 'away_team_goals_avg',
                        'home_win_pct', 'away_win_pct']
            X = df[features]
            y = df['result_label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, "trained_model.pkl")
        except Exception as e:
            st.error("CSV 数据读取失败，请确保存在名为 'epl_sample.csv' 的文件并包含所需字段")
            raise e
    return model

def convert_prob_to_odds(prob_dict, return_rate=0.94):
    base_odds = {k: 1 / v if v > 0 else 100.0 for k, v in prob_dict.items()}
    total_inverse = sum(1 / v for v in prob_dict.values())
    adjusted_odds = {
        k: round(1 / ((1 / o) / total_inverse / return_rate), 2)
        for k, o in base_odds.items()
    }
    return adjusted_odds

def save_prediction(input_dict, pred_probs, odds, model_source):
    record = {
        **input_dict,
        **pred_probs,
        '主胜赔率': odds['主胜'],
        '平局赔率': odds['平局'],
        '客胜赔率': odds['客胜'],
        '模型来源': model_source,
        '预测时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    file = "prediction_records.csv"
    df = pd.DataFrame([record])
    if os.path.exists(file):
        df.to_csv(file, mode='a', header=False, index=False)
    else:
        df.to_csv(file, index=False)

st.title("⚽ 实时足球赔率预测（支持上传数据训练 + 可视化 + 保存 + 下载 + 清空）")

st.markdown(f"**当前模型数据源**：{model_source}")
st.markdown("请输入以下参数来预测比赛结果：")

col1, col2 = st.columns(2)
with col1:
    home_rank = st.slider("主队排名", 1, 20, 7)
    home_goals = st.slider("主队平均进球数", 0.0, 5.0, 1.8)
    home_win_pct = st.slider("主队胜率", 0.0, 1.0, 0.6)

with col2:
    away_rank = st.slider("客队排名", 1, 20, 8)
    away_goals = st.slider("客队平均进球数", 0.0, 5.0, 1.3)
    away_win_pct = st.slider("客队胜率", 0.0, 1.0, 0.4)

if st.button("🎯 预测结果与赔率"):
    model = load_or_train_model()
    input_dict = {
        'home_team_rank': home_rank,
        'away_team_rank': away_rank,
        'home_team_goals_avg': home_goals,
        'away_team_goals_avg': away_goals,
        'home_win_pct': home_win_pct,
        'away_win_pct': away_win_pct
    }
    input_df = pd.DataFrame([input_dict])
    proba = model.predict_proba(input_df)[0]
    pred_probs = {
        '主胜': round(proba[0], 4),
        '平局': round(proba[1], 4),
        '客胜': round(proba[2], 4)
    }
    odds = convert_prob_to_odds(pred_probs)
    save_prediction(input_dict, pred_probs, odds, model_source)

    st.markdown("---")
    st.subheader("📊 预测概率（饼图）")
    fig, ax = plt.subplots()
    ax.pie(pred_probs.values(), labels=pred_probs.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("💰 建议赔率")
    st.write(f"主胜赔率: {odds['主胜']}, 平局赔率: {odds['平局']}, 客胜赔率: {odds['客胜']}")

    st.success("✅ 本次预测已保存到 prediction_records.csv")

    st.caption("⚠️ 模型需配套真实数据使用。默认训练示例数据，支持上传CSV文件")

if os.path.exists("prediction_records.csv"):
    with open("prediction_records.csv", "rb") as f:
        st.download_button("📥 下载所有预测记录 CSV", f, "prediction_records.csv", "text/csv")
    if st.button("🗑️ 清空所有预测记录"):
        os.remove("prediction_records.csv")
        st.warning("✅ 所有记录已清空！")
