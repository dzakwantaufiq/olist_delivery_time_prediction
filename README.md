# 🚚 Olist E-Commerce Delivery Time Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-Spatial-yellow.svg)](https://duckdb.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Olist%20Brazilian%20Ecommerce-blue)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Geospatial Analytics pipeline designed to predict total order delivery lead times on **Olist**, Brazil's largest marketplace. By combining DuckDB spatial boundary sanitization, relational SQL feature store generation, historical seller performance windows, and gradient-boosted regression models, this project provides accurate delivery estimates to enhance customer transparency and optimize logistics operations.

---

## 📌 Business Overview & Problem Statement

In nationwide e-commerce marketplaces across geographically vast regions like Brazil, inaccurate delivery estimations erode customer trust, reduce repeat purchases, and inflate customer support volume.

**Objectives:**
1. **Target Formulation:** Accurately predict total transit duration ($T_{\text{delivery}} = T_{\text{delivered}} - T_{\text{purchase}}$ in seconds).
2. **Geospatial Data Sanitization:** Filter coordinate noise and anomalies outside Brazil using the official `geoBoundaries-BRA-ADM0` boundary polygon via DuckDB Spatial.
3. **Leakage-Free Feature Store:** Compute dynamic historical seller performance indicators (approval speed, carrier dispatch speed, past late rates) using SQL window functions bounded strictly to past orders.
4. **Deployable Machine Learning Pipeline:** Train, calibrate, and export an atomic Scikit-Learn `Pipeline` handling heterogeneous spatial, logistical, and historical features for real-time inference.

---

## 🏗️ System Architecture & Workflow

```text
┌─────────────────────────┐
│ Kaggle Olist Dataset    │ (9 Relational CSVs)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ DuckDB Spatial Engine   │ ──► Filters lat/lng via geoBoundaries BRA ADM0 & aggregates zip medians
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ SQL Feature Store       │ ──► Derives distance, parcel density, payment types, and historical seller metrics
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Preprocessing Pipeline  │ ──► RobustScaler / StandardScaler + OneHotEncoder / TargetEncoder
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Model Benchmark & Tune  │ ──► Ridge, ElasticNet, RandomForest, LightGBM, XGBoost
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Serialized Model        │ ──► Exported as model_delivery_time.pkl
└─────────────────────────┘
```

---

## 🎯 Target Formulation & Data Sanitization

### 1. Delivery Time Decomposition
For completed shipments, the order journey is decomposed chronologically:
- **Approval Duration:** $\Delta t_{\text{approval}} = T_{\text{approved}} - T_{\text{purchase}}$
- **Carrier Dispatch Duration:** $\Delta t_{\text{carrier}} = T_{\text{carrier}} - T_{\text{approved}}$
- **Last-Mile Transit Duration:** $\Delta t_{\text{last-mile}} = T_{\text{delivered}} - T_{\text{carrier}}$
- **Total Lead Time (Target):** $\Delta t_{\text{total}} = T_{\text{delivered}} - T_{\text{purchase}}$

*Only orders satisfying $\Delta t_{\text{approval}} > 0$, $\Delta t_{\text{carrier}} > 0$, and $\Delta t_{\text{last-mile}} > 0$ with `delivered` status are retained to eliminate data-entry artifacts.*

### 2. Geospatial Filtering & Boundary Verification
Raw zip code coordinates contain measurement outliers outside Brazil. Coordinates are filtered via point-in-polygon verification.

Zip codes are subsequently grouped by prefix to extract median latitude/longitude and mode city/state strings, resolving multi-point variance.
---

## 📐 Feature Dictionary (Model Schema)

The end-to-end dataset combines spatial, physical, temporal, and historical seller signals:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `distance_m` | Float | Geodesic spatial distance (in meters) between buyer and seller zip codes. |
| `total_weight_g` | Float | Total combined weight of items in the order (grams). |
| `total_volume_cm3` | Float | Total volumetric package size ($L \times W \times H \text{ in } \text{cm}^3$). |
| `product_density` | Float | Ratio of order weight to package volume ($\text{g}/\text{cm}^3$). |
| `total_price` | Float | Gross monetary value of items in the order. |
| `freight_value` | Float | Total shipping charge paid by customer. |
| `freight_ratio` | Float | Shipping cost proportion relative to item value ($\text{Freight} / \text{Price}$). |
| `total_items` | Integer | Total count of individual physical items in the order. |
| `purchase_hour` | Integer | Hour of day when order was placed ($0 - 23$). |
| `purchase_dayofweek` | Integer | Day of week ($0 = \text{Monday}, 6 = \text{Sunday}$). |
| `is_purchase_on_weekend` | Boolean | Flag indicating whether order occurred on Saturday or Sunday. |
| `is_same_city` | Boolean | True if buyer and seller are registered in the same municipality. |
| `is_same_state` | Boolean | True if delivery is intra-state. |
| `is_customer_capital_city` | Boolean | True if customer resides in a state capital. |
| `is_seller_capital_city` | Boolean | True if seller is located in a state capital. |
| `avg_seller_hist_total_order` | Float | Lifetime historical order volume dispatched by seller prior to order date. |
| `seller_hist_avg_review_score` | Float | Seller's average customer review rating prior to order date. |
| `seller_hist_avg_approved_time_s` | Float | Historical mean duration for seller order approvals (seconds). |
| `seller_hist_avg_delivered_carrier_time_s` | Float | Historical mean time for seller to transfer parcel to carrier (seconds). |
| `seller_hist_avg_total_delivery_time_s` | Float | Historical mean overall delivery duration for seller (seconds). |
| `avg_seller_hist_late_rate` | Float | Historical percentage of orders delivered past the estimated date. |
| `mode_payment_type` | String | Primary payment method (`credit_card`, `boleto`, `voucher`, `debit_card`). |
| `mode_product_category_name_english`| String | Translated primary product category of ordered items. |
| `customer_state` | String | Two-letter state code of customer destination. |
| `mode_seller_state` | String | Two-letter state code of origin seller dispatch hub. |

---

## 🔬 Machine Learning Pipeline

The project implements a Scikit-Learn `Pipeline` coupled with a `ColumnTransformer` to guarantee zero data leakage between training and validation splits:

1. **Preprocessing Layer:**
   - **Numerical Pipeline:** Handled with `SimpleImputer(strategy='median')` followed by `RobustScaler()` to neutralize extreme freight and weight outliers.
   - **Categorical Pipeline:** High-cardinality state pairs and product categories encoded using `OneHotEncoder(handle_unknown='ignore')` or `TargetEncoder`.
2. **Evaluated Regressors:**
   - Linear Baselines: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
   - Tree & Ensemble Models: `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`
   - Boosting Frameworks: `XGBRegressor`, `LGBMRegressor`
3. **Evaluation Metrics:**
   - **MAE (Mean Absolute Error):** Direct interpretable delay error in days/hours.
   - **RMSE (Root Mean Squared Error):** Penalizes extreme logistics delays.
   - **$R^2$ Score:** Proportion of delivery lead-time variance captured by the model.

---

## 📂 Repository Structure

```text
.
├── olist_delivery_time_prediction.ipynb  # Main pipeline (DuckDB Spatial, Feature Store, ML)
├── model_delivery_time_prediction.sav    # Serialized Scikit-Learn regression pipeline
└── README.md                             # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/olist-delivery-time-prediction.git
cd olist-delivery-time-prediction

pip install pandas numpy scipy duckdb scikit-learn xgboost lightgbm category-encoders python-dotenv tqdm matplotlib seaborn
```

### 2. Environment Configuration

Create a `.env` file in the root directory to configure data directories:

```env
data_path=./data
data_path_geo=./data
```

### 3. DuckDB Spatial Setup

DuckDB Spatial is required for geospatial boundary verification. Ensure the extension is loaded prior to spatial transformations:

```python
import duckdb
duckdb.sql("INSTALL spatial; LOAD spatial;")
```

---

## 💻 Running Inference

Load the exported model artifact (`model_delivery_time.pkl`) to generate delivery time predictions for incoming orders:

```python
import pickle
import pandas as pd

# Load saved Scikit-Learn Pipeline
with open('model_delivery_time.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Sample order features
new_order = pd.DataFrame([{
    'distance_m': 190865.5,
    'total_weight_g': 2600.0,
    'total_volume_cm3': 7616.0,
    'product_density': 0.341,
    'total_price': 114.90,
    'freight_value': 11.90,
    'freight_ratio': 0.103,
    'total_items': 1,
    'purchase_hour': 14,
    'purchase_dayofweek': 1,
    'is_purchase_on_weekend': False,
    'is_same_city': False,
    'is_same_state': True,
    'is_customer_capital_city': False,
    'is_seller_capital_city': False,
    'avg_seller_hist_total_order': 1030.0,
    'seller_hist_avg_review_score': 4.14,
    'seller_hist_avg_approved_time_s': 37739.0,
    'seller_hist_avg_delivered_carrier_time_s': 196141.0,
    'seller_hist_avg_total_delivery_time_s': 1047966.0,
    'avg_seller_hist_late_rate': 7.96,
    'mode_payment_type': 'credit_card',
    'mode_product_category_name_english': 'bed_bath_table',
    'customer_state': 'SP',
    'mode_seller_state': 'SP'
}])

# Predict lead time (seconds converted to days)
pred_seconds = model_pipeline.predict(new_order)[0]
pred_days = pred_seconds / (24 * 3600)

print(f"Estimated Delivery Time: {pred_days:.1f} days ({pred_seconds:,.0f} seconds)")
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
