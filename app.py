import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pizza Prediction System",
    page_icon="🍕",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #d62828;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        font-size: 22px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Could not read {filename}: {e}")
            return None

    return None


# Load datasets
final_ingredients = load_csv("Final_Ingredients.csv")
final_sales = load_csv("Final_Sales.csv")
pizza_ingredients = load_csv("Pizza_ingredients - Pizza_ingredients.csv")
pizza_sales = load_csv("Pizza_Sale - pizza_sales.csv")
purchase_orders = load_csv("Purchase_Order.csv")
purchase_order_week = load_csv("Purchase_Order_Week_Start_2015-01-05.csv")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "trained_pizza_models.pkl")

models = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            models = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
else:
    st.warning(
        "trained_pizza_models.pkl was not found. "
        "Place app.py in the same folder as the model."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🍕 Pizza Predictive Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Pizza sales, ingredients and purchase order prediction system'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍕 Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Sales Analysis",
        "Ingredients",
        "Purchase Orders",
        "Prediction",
        "Model Information"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    # Find sales data
    df = pizza_sales if pizza_sales is not None else final_sales

    if df is not None:

        col1, col2, col3, col4 = st.columns(4)

        # Number of records
        with col1:
            st.metric(
                "Total Records",
                f"{len(df):,}"
            )

        # Total sales
        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        total_sales = 0

        for col in numeric_columns:
            if "price" in col.lower() or "sales" in col.lower():
                total_sales = df[col].sum()
                break

        with col2:
            st.metric(
                "Total Sales",
                f"{total_sales:,.2f}"
            )

        # Average
        with col3:
            if numeric_columns:
                avg_value = df[numeric_columns[0]].mean()
            else:
                avg_value = 0

            st.metric(
                "Average Value",
                f"{avg_value:,.2f}"
            )

        # Columns
        with col4:
            st.metric(
                "Columns",
                len(df.columns)
            )

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write("**Columns**")
            st.write(list(df.columns))

        with info_col2:
            st.write("**Data Types**")
            st.dataframe(
                pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str)
                }),
                use_container_width=True
            )

    else:
        st.error("No sales dataset could be loaded.")


# ============================================================
# SALES ANALYSIS
# ============================================================

elif page == "Sales Analysis":

    st.header("📈 Sales Analysis")

    df = pizza_sales if pizza_sales is not None else final_sales

    if df is None:
        st.error("Sales dataset not found.")
        st.stop()

    st.subheader("Sales Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        st.subheader("📊 Numeric Analysis")

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total",
                f"{df[selected_column].sum():,.2f}"
            )

        with col2:
            st.metric(
                "Average",
                f"{df[selected_column].mean():,.2f}"
            )

        with col3:
            st.metric(
                "Maximum",
                f"{df[selected_column].max():,.2f}"
            )

        st.subheader(
            f"Distribution of {selected_column}"
        )

        st.bar_chart(
            df[selected_column].value_counts().head(20)
        )


# ============================================================
# INGREDIENTS
# ============================================================

elif page == "Ingredients":

    st.header("🥗 Pizza Ingredients")

    df = (
        pizza_ingredients
        if pizza_ingredients is not None
        else final_ingredients
    )

    if df is None:
        st.error("Ingredient dataset not found.")
        st.stop()

    st.write(
        f"Total records: **{len(df):,}**"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Ingredient Statistics")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        selected_column = st.selectbox(
            "Select numeric ingredient column",
            numeric_columns
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total",
                f"{df[selected_column].sum():,.2f}"
            )

        with col2:
            st.metric(
                "Average",
                f"{df[selected_column].mean():,.2f}"
            )

        with col3:
            st.metric(
                "Maximum",
                f"{df[selected_column].max():,.2f}"
            )


# ============================================================
# PURCHASE ORDERS
# ============================================================

elif page == "Purchase Orders":

    st.header("📦 Purchase Orders")

    df = purchase_orders

    if df is None:
        df = purchase_order_week

    if df is None:
        st.error("Purchase order dataset not found.")
        st.stop()

    st.metric(
        "Total Purchase Order Records",
        f"{len(df):,}"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        st.subheader("Purchase Order Analysis")

        selected_column = st.selectbox(
            "Select column",
            numeric_columns
        )

        st.line_chart(
            df[selected_column]
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.header("🔮 Pizza Prediction")

    if models is None:
        st.error(
            "The trained model could not be loaded."
        )
        st.stop()

    st.write(
        "Select a trained model and provide the required input values."
    )

    # --------------------------------------------------------
    # MODEL TYPE
    # --------------------------------------------------------

    st.subheader("Available Model")

    if isinstance(models, dict):

        model_names = list(models.keys())

        selected_model_name = st.selectbox(
            "Select Model",
            model_names
        )

        model = models[selected_model_name]

    else:

        model = models
        selected_model_name = "Main Model"

    st.info(
        f"Selected model: **{selected_model_name}**"
    )

    # --------------------------------------------------------
    # GET FEATURE NAMES
    # --------------------------------------------------------

    feature_names = None

    # Pipeline
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    # Model itself
    elif hasattr(model, "get_feature_names_out"):
        try:
            feature_names = list(
                model.get_feature_names_out()
            )
        except Exception:
            pass

    # Pipeline final estimator
    if feature_names is None and hasattr(model, "steps"):

        for _, step in model.steps:

            if hasattr(step, "feature_names_in_"):
                feature_names = list(
                    step.feature_names_in_
                )
                break

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    if feature_names:

        st.subheader("Input Features")

        input_values = {}

        columns = st.columns(2)

        for index, feature in enumerate(feature_names):

            with columns[index % 2]:

                input_values[feature] = st.number_input(
                    feature,
                    value=0.0
                )

        if st.button(
            "🔮 Predict",
            use_container_width=True
        ):

            try:

                input_df = pd.DataFrame(
                    [input_values]
                )

                prediction = model.predict(
                    input_df
                )

                st.success("Prediction completed!")

                st.markdown(
                    f"""
                    <div class="prediction-box">
                    🍕 Prediction: {prediction[0]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    else:

        st.warning(
            "The model does not expose feature names."
        )

        st.write(
            "Use the Model Information page to inspect "
            "the trained model."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("🤖 Trained Model Information")

    if models is None:
        st.error(
            "trained_pizza_models.pkl could not be loaded."
        )
        st.stop()

    st.write(
        "### Python Object Type"
    )

    st.code(
        str(type(models))
    )

    # --------------------------------------------------------
    # DICTIONARY OF MODELS
    # --------------------------------------------------------

    if isinstance(models, dict):

        st.subheader("Available Models")

        for name, model in models.items():

            with st.expander(
                f"🔹 {name}"
            ):

                st.write(
                    "**Model Type:**"
                )

                st.code(
                    str(type(model))
                )

                if hasattr(
                    model,
                    "feature_names_in_"
                ):

                    st.write(
                        "**Features:**"
                    )

                    st.write(
                        list(
                            model.feature_names_in_
                        )
                    )

                if hasattr(
                    model,
                    "n_features_in_"
                ):

                    st.write(
                        "**Number of Features:**"
                    )

                    st.write(
                        model.n_features_in_
                    )

                if hasattr(
                    model,
                    "classes_"
                ):

                    st.write(
                        "**Classes:**"
                    )

                    st.write(
                        list(model.classes_)
                    )

    else:

        st.subheader("Model")

        st.code(
            str(type(models))
        )

        if hasattr(
            models,
            "feature_names_in_"
        ):

            st.write(
                "**Features:**"
            )

            st.write(
                list(
                    models.feature_names_in_
                )
            )

        if hasattr(
            models,
            "n_features_in_"
        ):

            st.write(
                "**Number of Features:**"
            )

            st.write(
                models.n_features_in_
            )

        if hasattr(
            models,
            "classes_"
        ):

            st.write(
                "**Classes:**"
            )

            st.write(
                list(models.classes_)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🍕 Pizza Predictive Analytics System | "
    "Built with Python & Streamlit"
)