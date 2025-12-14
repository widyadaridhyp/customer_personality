import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

FASTAPI_URL = "http://fastapi:8000/predict"

st.title("🛍️ Customer Personality Analysis & Prediction")

tab1, tab2 = st.tabs(["📝 Single Prediction (Form)", "📂 Upload CSV + EDA + Bulk Prediction"])

# =====================================================================
# TAB 1 — FORM INPUT PREDICTION
# =====================================================================
with tab1:

    st.header("🔮 Customer Responses Prediction")

    st.subheader("Variable Description")
    st.markdown("""
    | Variable | Type | Categories / Levels | Description |
    |---------|-------|----------------------|-------------|
    | **income** | Numeric (int/float) | — | Annual customer revenue |
    | **recency** | Numeric (int) | — | Days since last purchase |
    | **numwebvisitsmonth** | Numeric (int) | — | Website visits per month |
    | **numwebpurchases** | Numeric (int) | — | Purchases via website |
    | **numstorepurchases** | Numeric (int) | — | Purchases in physical store |
    | **numcatalogpurchases** | Numeric (int) | — | Purchases via catalog |
    | **numdealspurchases** | Numeric (int) | — | Purchases during deals/promotions |
    | **mntwines** | Numeric (float/int) | — | Expenditure on wine |
    | **mntfruits** | Numeric (float/int) | — | Expenditure on fruits |
    | **mntmeatproducts** | Numeric (float/int) | — | Expenditure on meat |
    | **mntfishproducts** | Numeric (float/int) | — | Expenditure on fish |
    | **mntsweetproducts** | Numeric (float/int) | — | Expenditure on sweets |
    | **mntgoldprods** | Numeric (float/int) | — | Expenditure on gold products |
    | **education** | Categorical | Basic, Graduation, Master, PhD | Highest education level |
    | **marital_status** | Categorical | Married, Together, Divorced, Single | Customer marital status |
    """)

    st.subheader("Prediction Form")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Kolom 1
        with col1:
            income = st.number_input("Income", min_value=0, value=30000, help="Pendapatan tahunan pelanggan")
            recency = st.number_input("Recency", min_value=0, value=10, help="Hari sejak pembelian terakhir")
            numwebvisitsmonth = st.number_input("Web Visits / Month", min_value=0, value=5)

        # Kolom 2
        with col2:
            numwebpurchases = st.number_input("Web Purchases", min_value=0, value=2)
            numstorepurchases = st.number_input("Store Purchases", min_value=0, value=3)
            numcatalogpurchases = st.number_input("Catalog Purchases", min_value=0, value=1)
            numdealspurchases = st.number_input("Deals Purchases", min_value=0, value=1)

        # Kolom 3 — Spending
        with col3:
            mntwines = st.number_input("Wines Spending", min_value=0, value=100)
            mntfruits = st.number_input("Fruits Spending", min_value=0, value=20)
            mntmeatproducts = st.number_input("Meat Spending", min_value=0, value=50)
            mntfishproducts = st.number_input("Fish Spending", min_value=0, value=30)
            mntsweetproducts = st.number_input("Sweet Spending", min_value=0, value=10)
            mntgoldprods = st.number_input("Gold Products Spending", min_value=0, value=5)

        st.subheader("👤 Informasi Pelanggan")
        col4, col5 = st.columns(2)

        with col4:
            education = st.selectbox("Education", ["Basic", "Graduation", "Master", "PhD"])

        with col5:
            marital = st.selectbox("Marital Status", ["Married", "Together", "Divorced", "Single"])

        submit = st.form_submit_button("🎯 Prediksi Sekarang")

    if submit:
        df = pd.DataFrame([{
            "income": income,
            "recency": recency,
            "numwebvisitsmonth": numwebvisitsmonth,
            "numwebpurchases": numwebpurchases,
            "numstorepurchases": numstorepurchases,
            "numcatalogpurchases": numcatalogpurchases,
            "numdealspurchases": numdealspurchases,
            "mntwines": mntwines,
            "mntfruits": mntfruits,
            "mntmeatproducts": mntmeatproducts,
            "mntfishproducts": mntfishproducts,
            "mntsweetproducts": mntsweetproducts,
            "mntgoldprods": mntgoldprods,
            "education": education,
            "marital_status": marital
        }])

        response = requests.post(FASTAPI_URL, json={"data": df.to_dict(orient="records")})

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction'][0]}")
            st.info(f"Probability: {result['probability'][0]:.4f}")
        else:
            st.error("Terjadi error di backend FastAPI.")


# =====================================================================
# TAB 2 — UPLOAD CSV + EDA + BULK PREDICTION
# =====================================================================
with tab2:

    st.header("📂 Upload Dataset for Analysis and Predict Response Customer")

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success("File berhasil diupload!")

        # ==========================================================
        # EDA SECTION
        # ==========================================================
        st.subheader("📄 Preview Data")
        st.dataframe(df_raw.head())

        numeric_cols = [
            "income","recency","numwebvisitsmonth","numwebpurchases",
            "numstorepurchases","numcatalogpurchases","numdealspurchases",
            "mntwines","mntfruits","mntmeatproducts","mntfishproducts",
            "mntsweetproducts","mntgoldprods"
        ]

        st.subheader("📊 Descriptive Statistics")
        st.write(df_raw[numeric_cols].describe())

        st.subheader("📈 Scatterplot")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox('Choose first variable', numeric_cols, index=0)
            y_col = st.selectbox('Choose second variable', numeric_cols, index=0)
            
            fig = px.scatter(df_raw, x=x_col, y=y_col)
            st.plotly_chart(fig, key=f"scatter_{x_col}_{y_col}")

            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(df_raw, x=x_col, nbins=30, title=f"Distribusi {x_col}")
                st.plotly_chart(fig1, use_container_width=True, key=f"hist_x_{x_col}")
            with col2:
                fig2 = px.histogram(df_raw, x=y_col, nbins=30, title=f"Distribusi {y_col}")
                st.plotly_chart(fig2, use_container_width=True, key=f"hist_y_{y_col}")

        # Monetary untuk heatmap
        df_raw["monetary"] = (
            df_raw["mntwines"] + df_raw["mntfruits"] +
            df_raw["mntmeatproducts"] + df_raw["mntfishproducts"] +
            df_raw["mntsweetproducts"] + df_raw["mntgoldprods"]
        )

        corr_cols = [
            "income","recency","numwebvisitsmonth","numwebpurchases",
            "numstorepurchases","numcatalogpurchases",
            "numdealspurchases","monetary"
        ]

        st.subheader("🔥 Heatmap Korelasi")
        corr_matrix = df_raw[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(fig_corr)

        # ==========================================================
        # ADVANCED EDA VISUALIZATIONS
        # ==========================================================
        st.subheader("📌 Advanced EDA Visualizations")

        vis_type = st.selectbox(
            "Choose Visualization",
            [
                "Campaign Acceptance",
                "Categorical vs Numerical (Boxplot)",
                "Categorical vs Categorical (Stacked / Countplot)"
            ]
        )

        df_vis = df_raw.copy()

        # ===============================
        # 1. CAMPAIGN ACCEPTANCE
        # ===============================
        if vis_type == "Campaign Acceptance":

            st.markdown("### 🎯 Campaign Acceptance")

            campaign_cols = ["acceptedcmp1","acceptedcmp2","acceptedcmp3","acceptedcmp4","acceptedcmp5"]
            selected_cmp = st.selectbox("Choose Campaign Variable", campaign_cols)

            fig, ax = plt.subplots(figsize=(6,4))

            if "response" in df_vis.columns:
                sns.countplot(data=df_vis, x=selected_cmp, hue="response", palette="crest", ax=ax)
                ax.set_title(f"{selected_cmp} vs Response")
            else:
                sns.countplot(data=df_vis, x=selected_cmp, palette="crest", ax=ax)
                ax.set_title(f"Distribution of {selected_cmp}")

            st.pyplot(fig)


        # ===============================
        # 2. CATEGORICAL VS NUMERICAL
        # ===============================
        elif vis_type == "Categorical vs Numerical (Boxplot)":

            st.markdown("### 📦 Boxplot: Categorical vs Numerical")

            num_cols = [
                "income","recency","numwebvisitsmonth","numwebpurchases","numstorepurchases",
                "numcatalogpurchases","numdealspurchases","mntwines","mntfruits",
                "mntmeatproducts","mntfishproducts","mntsweetproducts","mntgoldprods"
            ]

            cat_cols = ["education", "marital_status"] + \
                    ["acceptedcmp1","acceptedcmp2","acceptedcmp3","acceptedcmp4","acceptedcmp5"]

            cat_sel = st.selectbox("Choose Categorical Variable", cat_cols)
            num_sel = st.selectbox("Choose Numerical Variable", num_cols)

            fig, ax = plt.subplots(figsize=(7,5))

            if "response" in df_vis.columns:
                sns.boxplot(data=df_vis, x="response", y=num_sel, palette="crest", ax=ax)
                ax.set_title(f"{num_sel} vs Response")
            else:
                sns.boxplot(data=df_vis, x=cat_sel, y=num_sel, palette="crest", ax=ax)
                ax.set_title(f"{num_sel} by {cat_sel}")

            st.pyplot(fig)


        # ===============================
        # 3. CATEGORICAL VS CATEGORICAL
        # ===============================
        elif vis_type == "Categorical vs Categorical (Stacked / Countplot)":

            st.markdown("### 📊 Categorical vs Categorical")

            cat_cols = ["education", "marital_status"] + \
                    ["acceptedcmp1","acceptedcmp2","acceptedcmp3","acceptedcmp4","acceptedcmp5"]

            cat_x = st.selectbox("Choose First Categorical Variable", cat_cols, key="cat_x")
            cat_y = st.selectbox(
                "Choose Second Categorical Variable",
                [c for c in cat_cols if c != cat_x],
                key="cat_y"
            )
            # ----------- CASE 1: dataset punya response -----------
            if "response" in df_vis.columns:
                st.markdown("### 🟦 Stacked Percentage Plot")

                ct = pd.crosstab(df_vis[cat_x], df_vis["response"])
                pct = ct.div(ct.sum(axis=1), axis=0) * 100

                fig, ax = plt.subplots(figsize=(8,6))
                pct.plot(kind="bar", stacked=True, colormap="crest", ax=ax)
                ax.set_ylabel("Percentage (%)")
                ax.set_title(f"{cat_x} vs Response")

                # Add labels
                for i in range(len(pct)):
                    cumulative = 0
                    for j in range(len(pct.columns)):
                        val = pct.iloc[i, j]
                        if val > 0:
                            ax.text(i, cumulative + val/2, f"{val:.1f}%", ha="center", fontsize=9)
                        cumulative += val

                st.pyplot(fig)

            # ----------- CASE 2: dataset TANPA response -----------
            else:
                st.markdown("### 🟧 Stacked Distribution (No Target)")

                ct = pd.crosstab(df_vis[cat_x], df_vis[cat_y])

                fig, ax = plt.subplots(figsize=(8,6))
                ct.plot(kind="bar", stacked=True, colormap="crest", ax=ax)

                ax.set_ylabel("Count")
                ax.set_title(f"{cat_x} vs {cat_y}")
                plt.xticks(rotation=0)

                st.pyplot(fig)

        # ==========================================================
        # PREDIKSI MASSAL
        # ==========================================================
        st.header("🤖 Bulk Prediction")

        if st.button("Predict All"):
            with st.spinner("Predicting..."):
                payload = {"data": df_raw.to_dict(orient="records")}
                response = requests.post(FASTAPI_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    df_raw["prediction"] = result["prediction"]
                    df_raw["probability"] = result["probability"]

                    st.success("Prediksi berhasil!")
                    st.dataframe(df_raw)

                    # Download hasil prediksi
                    csv = df_raw.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")
                else:
                    st.error("Gagal memanggil backend FastAPI.")
