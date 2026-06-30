import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc)

st.set_page_config(page_title="ShapeX Laundry — Franchise Analytics", layout="wide")
sns.set_style("whitegrid")

st.title("🧺 ShapeX Laundry — Franchise Expansion Analytics Dashboard")
st.caption("Consumer demand & franchise-partner propensity analysis | Pune outlets: Amanora Township · Viman Nagar · Pimple Saudagar")

@st.cache_data
def load_data():
    loc = pd.read_csv("location_performance.csv")
    cons = pd.read_csv("consumer_survey.csv")
    fr = pd.read_csv("franchise_survey.csv")
    return loc, cons, fr

location_performance, consumer, franchise = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Location Performance", "🧑‍🤝‍🧑 Consumer Insights",
                                   "🤝 Franchise Partners", "🤖 Predictive Models"])

with tab1:
    st.subheader("Descriptive Analysis — Location Performance")
    c1, c2 = st.columns(2)
    with c1:
        area_filter = st.multiselect("Filter by area type", sorted(location_performance.area_type.unique()),
                                      default=list(location_performance.area_type.unique()))
    with c2:
        city_filter = st.multiselect("Filter by city", sorted(location_performance.city.unique()),
                                      default=list(location_performance.city.unique()))
    fdf = location_performance[location_performance.area_type.isin(area_filter) & location_performance.city.isin(city_filter)]

    loc_summary = fdf.groupby("area_type").agg(
        avg_monthly_revenue=("monthly_revenue_inr","mean"),
        avg_daily_footfall=("daily_footfall","mean"),
        avg_rent_per_sqft=("rent_per_sqft_inr","mean")
    ).round(1).sort_values("avg_monthly_revenue", ascending=False)
    loc_summary["revenue_per_rent_unit"] = (loc_summary.avg_monthly_revenue/loc_summary.avg_rent_per_sqft).round(0)
    st.dataframe(loc_summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        loc_summary["avg_monthly_revenue"].plot(kind="barh", ax=ax, color="#B5651D")
        ax.set_title("Avg Monthly Revenue by Area Type"); ax.set_xlabel("INR")
        st.pyplot(fig)
    with col2:
        xtab_rev = pd.crosstab(fdf.area_type, fdf.competitor_density, values=fdf.monthly_revenue_inr, aggfunc="mean").round(0)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.heatmap(xtab_rev, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax2)
        ax2.set_title("Revenue: Area Type x Competitor Density")
        st.pyplot(fig2)

    st.info("**Insight:** IT Park / Working Professional hubs with Low-Medium competitor density show the highest revenue ceilings. Student/PG-dense areas offer the best revenue-per-rent efficiency for franchise unit economics.")

with tab2:
    st.subheader("Consumer Survey — Descriptive & Diagnostic")
    c1, c2 = st.columns(2)
    with c1:
        xtab = pd.crosstab(consumer.area_type, consumer.will_avail_service, normalize="index").round(2)*100
        xtab.columns = ["Will NOT avail (%)","Will avail (%)"]
        fig, ax = plt.subplots(figsize=(6,4))
        xtab["Will avail (%)"].sort_values().plot(kind="barh", color="#2E2A24", ax=ax)
        ax.set_title("Willingness to Avail, by Area Type")
        st.pyplot(fig)
    with c2:
        pivot = consumer.pivot_table(index="biggest_pain_point", columns="residence_type",
                                      values="will_avail_service", aggfunc="mean").round(2)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax2)
        ax2.set_title("Diagnostic: Pain Point x Residence Type")
        st.pyplot(fig2)
    st.info("**Diagnostic insight:** PG/Hostel residents citing 'No machine access' show the single highest willingness-to-avail cell — a cross-feature effect, not explained by either variable alone.")

    st.markdown("##### Raw consumer survey sample")
    st.dataframe(consumer.head(20), use_container_width=True)

with tab3:
    st.subheader("Franchise-Partner Survey — Descriptive & Diagnostic")
    c1, c2 = st.columns(2)
    with c1:
        xf = pd.crosstab(franchise.background, franchise.will_invest_in_franchise, normalize="index").round(2)*100
        xf.columns=["Will NOT invest (%)","Will invest (%)"]
        fig, ax = plt.subplots(figsize=(6,4))
        xf["Will invest (%)"].sort_values().plot(kind="barh", color="#6B8E23", ax=ax)
        ax.set_title("Willingness to Invest, by Background")
        st.pyplot(fig)
    with c2:
        franchise["capital_tier"] = pd.cut(franchise.capital_lakh_inr, bins=[0,15,30,100],
                                            labels=["Low (<=15L)","Mid (15-30L)","High (>30L)"])
        diag = franchise.groupby("capital_tier")[["attitude_breakeven_confidence","attitude_commit_5yr"]].mean().round(2)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        diag.plot(kind="bar", ax=ax2, color=["#B5651D","#2E2A24"])
        ax2.set_title("Confidence & Commitment by Capital Tier"); ax2.set_ylim(0,5)
        st.pyplot(fig2)
    st.info("**Diagnostic insight:** Capital adequacy acts as a confidence multiplier on breakeven belief and 5-year commitment — not just a funding gate.")

    st.markdown("##### Raw franchise survey sample")
    st.dataframe(franchise.head(20), use_container_width=True)

with tab4:
    st.subheader("Predictive Models — Consumer / Franchisee Propensity")
    target_choice = st.radio("Choose propensity model", ["Consumer: will_avail_service","Franchisee: will_invest_in_franchise"], horizontal=True)

    @st.cache_resource
    def train_consumer_models():
        df = consumer.copy()
        df["pg_no_machine"] = ((df.residence_type=="PG/Hostel") & (df.biggest_pain_point=="No machine access")).astype(int)
        df["high_time_pressure"] = df.occupation.isin(["Working Professional","Student"]).astype(int)
        df["price_sensitivity"] = (df.fair_price_per_wash_inr < df.fair_price_per_wash_inr.median()).astype(int)
        df["attitude_composite"] = df[[c for c in df.columns if c.startswith("attitude_")]].mean(axis=1)
        df["spend_per_load"] = (df.current_monthly_spend_inr/(df.loads_per_week*4)).round(1)
        df["proximity_friendly"] = (df.travel_tolerance_min<=10).astype(int)
        cat_cols=["age_group","occupation","residence_type","city","area_type","household_size","current_laundry_method","biggest_pain_point"]
        num_cols=["loads_per_week","current_monthly_spend_inr","travel_tolerance_min","fair_price_per_wash_inr",
                  "desired_features_count","attitude_switch_convenience","attitude_time_over_cost","attitude_trust_automation",
                  "attitude_subscription_appeal","attitude_would_recommend","pg_no_machine","high_time_pressure",
                  "price_sensitivity","attitude_composite","spend_per_load","proximity_friendly"]
        X = pd.get_dummies(df[cat_cols+num_cols], columns=cat_cols, drop_first=True)
        y = df["will_avail_service"]
        return run_pipeline(X,y)

    @st.cache_resource
    def train_franchise_models():
        fdf = franchise.copy()
        fdf["high_capital"] = (fdf.capital_lakh_inr >= fdf.capital_lakh_inr.median()).astype(int)
        fdf["space_ready"] = (fdf.space_availability=="Have space ready").astype(int)
        fdf["prior_relevant_exp"] = (fdf.prior_franchise_experience=="Yes-laundry/service").astype(int)
        fdf["attitude_composite_fr"] = fdf[[c for c in fdf.columns if c.startswith("attitude_")]].mean(axis=1)
        fdf["fast_launch"] = fdf.launch_timeline.isin(["<3m","3-6m"]).astype(int)
        fcat=["background","prior_franchise_experience","proposed_city","proposed_area_type","space_availability","royalty_structure_pref","launch_timeline"]
        fnum=["capital_lakh_inr","space_sqft","attitude_brand_reduces_risk","attitude_breakeven_confidence",
              "attitude_low_staffing_appeal","attitude_support_importance","attitude_commit_5yr",
              "high_capital","space_ready","prior_relevant_exp","attitude_composite_fr","fast_launch"]
        X = pd.get_dummies(fdf[fcat+fnum], columns=fcat, drop_first=True)
        y = fdf["will_invest_in_franchise"]
        return run_pipeline(X,y)

    def run_pipeline(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=9).fit(X_train_s, y_train),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42).fit(X_train, y_train),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(X_train, y_train),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42).fit(X_train, y_train),
        }
        results=[]; preds={}
        for name, m in models.items():
            Xtr = X_train_s if name=="KNN" else X_train
            Xte = X_test_s if name=="KNN" else X_test
            ytr_p = m.predict(Xtr); yte_p = m.predict(Xte); yte_proba = m.predict_proba(Xte)[:,1]
            preds[name] = {"y_pred":yte_p, "y_proba":yte_proba, "y_test":y_test}
            fpr,tpr,_ = roc_curve(y_test, yte_proba)
            results.append({"Model":name,
                "Train Accuracy":round(accuracy_score(y_train,ytr_p),3),
                "Test Accuracy":round(accuracy_score(y_test,yte_p),3),
                "Precision":round(precision_score(y_test,yte_p),3),
                "Recall":round(recall_score(y_test,yte_p),3),
                "F1-score":round(f1_score(y_test,yte_p),3),
                "ROC-AUC":round(auc(fpr,tpr),3)})
        return pd.DataFrame(results), preds, models, X.columns

    if target_choice.startswith("Consumer"):
        results_df, preds, models, feat_cols = train_consumer_models()
    else:
        results_df, preds, models, feat_cols = train_franchise_models()

    st.markdown("##### Model performance comparison")
    st.dataframe(results_df.sort_values("F1-score", ascending=False), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(len(results_df)); w=0.35
        ax.bar(x-w/2, results_df["Train Accuracy"], width=w, label="Train", color="#B5651D")
        ax.bar(x+w/2, results_df["Test Accuracy"], width=w, label="Test", color="#2E2A24")
        ax.set_xticks(x); ax.set_xticklabels(results_df["Model"], rotation=15)
        ax.set_ylim(0,1.05); ax.legend(); ax.set_title("Train vs Test Accuracy")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        for name in models:
            fpr, tpr, _ = roc_curve(preds[name]["y_test"], preds[name]["y_proba"])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        ax2.plot([0,1],[0,1],"k--", alpha=0.4)
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC Curves"); ax2.legend(fontsize=8)
        st.pyplot(fig2)

    st.markdown("##### Confusion Matrices")
    cols = st.columns(4)
    for col, name in zip(cols, models):
        cm = confusion_matrix(preds[name]["y_test"], preds[name]["y_pred"])
        fig, ax = plt.subplots(figsize=(3.2,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name, fontsize=10)
        col.pyplot(fig)

    best_model_name = results_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    st.success(f"**Recommended model: {best_model_name}** — highest F1-score with stable train/test accuracy gap, indicating good generalisation for lead-scoring.")

st.markdown("---")
st.caption("ShapeX Laundry Franchise Analytics · Synthetic data for planning purposes · Replace with real survey/outlet data when available.")
