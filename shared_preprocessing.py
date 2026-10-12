"""Shared preprocessing for the Cook County single-family sale-price project.

Stage 2 (Sakura). Implements the rules agreed in Yuming's EDA (Sections 2.12-2.25):
  * single-family sales only, one row per (PIN, sale date, sale price) group,
    known card number, single-card indicators
  * chronological split: 2017-2023 train / 2024 validation / 2025 test
  * 81 inputs: 53 numerical + 28 categorical
  * numerical NaN -> training-set median; categorical NaN -> "Missing"
  * every learned step is fitted on the training split only

Model owners: load the exported files, then put the matching encoder from
make_ridge_encoder() / make_tree_encoder() inside your own Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "meta_sale_price"
SALE_KEY = ["meta_pin", "meta_sale_date", "meta_sale_price"]
TRAIN_YEARS = (2017, 2023)
VAL_YEAR = 2024
TEST_YEAR = 2025
MISSING_LABEL = "Missing"

# Non-input columns kept alongside each split for error analysis (Phase 4).
# They are NOT model inputs.
ANALYSIS_COLUMNS = [
    "meta_pin", "meta_sale_date", "meta_township_code",
    "meta_nbhd_code", "meta_triad_name",
]

NUMERIC_FEATURES = [
    "char_yrblt", "char_bldg_sf", "char_land_sf", "char_beds", "char_rooms",
    "char_fbath", "char_hbath", "char_frpl", "loc_longitude", "loc_latitude",
    "loc_env_flood_fs_factor", "loc_env_airport_noise_dnl",
    "loc_access_cmap_walk_nta_score", "loc_access_cmap_walk_total_score",
    "prox_num_pin_in_half_mile", "prox_num_bus_stop_in_half_mile",
    "prox_num_school_in_half_mile", "prox_num_school_with_rating_in_half_mile",
    "prox_airport_dnl_total", "prox_nearest_bike_trail_dist_ft",
    "prox_nearest_cemetery_dist_ft", "prox_nearest_cta_route_dist_ft",
    "prox_nearest_cta_stop_dist_ft", "prox_nearest_golf_course_dist_ft",
    "prox_nearest_grocery_store_dist_ft", "prox_nearest_hospital_dist_ft",
    "prox_lake_michigan_dist_ft", "prox_nearest_major_road_dist_ft",
    "prox_nearest_metra_route_dist_ft", "prox_nearest_metra_stop_dist_ft",
    "prox_nearest_park_dist_ft", "prox_nearest_railroad_dist_ft",
    "prox_nearest_road_arterial_daily_traffic", "prox_nearest_road_arterial_dist_ft",
    "prox_nearest_road_arterial_lanes", "prox_nearest_road_arterial_speed_limit",
    "prox_nearest_road_collector_daily_traffic", "prox_nearest_road_collector_dist_ft",
    "prox_nearest_road_collector_lanes", "prox_nearest_road_collector_speed_limit",
    "prox_nearest_road_highway_daily_traffic", "prox_nearest_road_highway_dist_ft",
    "prox_nearest_road_highway_lanes", "prox_nearest_secondary_road_dist_ft",
    "prox_nearest_stadium_dist_ft", "prox_nearest_university_dist_ft",
    "prox_nearest_water_dist_ft", "shp_parcel_centroid_dist_ft_sd",
    "shp_parcel_edge_len_ft_sd", "shp_parcel_interior_angle_sd",
    "shp_parcel_mrr_area_ratio", "shp_parcel_mrr_side_ratio",
    "shp_parcel_num_vertices",
]

CATEGORICAL_FEATURES = [
    "char_type_resd", "char_cnst_qlty", "char_attic_fnsh", "char_gar1_att",
    "char_gar1_area", "char_gar1_size", "char_gar1_cnst", "char_attic_type",
    "char_bsmt", "char_ext_wall", "char_heat", "char_bsmt_fin", "char_roof_cnst",
    "char_class", "char_use", "char_porch", "char_air", "loc_property_city",
    "loc_property_zip", "loc_tax_municipality_name",
    "loc_school_elementary_district_geoid", "loc_school_secondary_district_geoid",
    "loc_env_flood_fema_sfha", "loc_env_flood_fs_risk_direction",
    "loc_env_ohare_noise_contour_no_buffer_bool",
    "prox_nearest_road_arterial_surface_type",
    "prox_nearest_road_collector_surface_type",
    "prox_nearest_road_highway_surface_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
assert len(NUMERIC_FEATURES) == 53 and len(CATEGORICAL_FEATURES) == 28


# ---------------------------------------------------------------------------
# 1. Eligibility and chronological split (stateless: no statistics learned)
# ---------------------------------------------------------------------------
def select_eligible_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the EDA Section 2.23 eligibility rules to the raw dataset."""
    houses = df.loc[df["meta_modeling_group"] == "SF"].copy()
    houses["meta_sale_date"] = pd.to_datetime(houses["meta_sale_date"])
    rows_per_sale = (
        houses.groupby(SALE_KEY, dropna=False)["meta_pin"].transform("size")
    )
    eligible = (
        (rows_per_sale == 1)
        & houses["meta_card_num"].notna()
        & houses["ind_pin_is_multicard"].eq(False)
        & houses["meta_pin_num_cards"].eq(1)
    )
    return houses.loc[eligible].copy()


def chronological_split(eligible: pd.DataFrame):
    """Return (train, validation, test) DataFrames split by sale year."""
    year = eligible["meta_sale_date"].dt.year
    train = eligible.loc[year.between(*TRAIN_YEARS)].copy()
    val = eligible.loc[year == VAL_YEAR].copy()
    test = eligible.loc[year == TEST_YEAR].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Type standardisation (stateless)
# ---------------------------------------------------------------------------
def _code_to_label(value):
    """Turn a stored code into a clean string label; keep missing as NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))          # e.g. -1.0 -> "-1"
    return str(value)


def standardise_types(split: pd.DataFrame) -> pd.DataFrame:
    """Select the 81 inputs; numeric -> float64, categorical -> str labels."""
    X = split[FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object).map(_code_to_label).astype(object)
    return X


# ---------------------------------------------------------------------------
# 3. Shared imputation (learned -> fit on training split ONLY)
# ---------------------------------------------------------------------------
def make_shared_imputer() -> ColumnTransformer:
    """Median for numeric, constant 'Missing' for categorical."""
    imputer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL,
                                  missing_values=np.nan), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return imputer.set_output(transform="pandas")


def apply_imputer(imputer, X: pd.DataFrame) -> pd.DataFrame:
    out = imputer.transform(X)
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype("float64")
    out[CATEGORICAL_FEATURES] = out[CATEGORICAL_FEATURES].astype(str)
    return out[FEATURES]


# ---------------------------------------------------------------------------
# 4. Model-specific encoders (put these INSIDE each model's Pipeline so they
#    are fitted on X_train only)
# ---------------------------------------------------------------------------
def make_ridge_encoder(min_frequency: int = 20) -> ColumnTransformer:
    """Ridge: standardise numerics, one-hot categoricals.

    Levels seen fewer than `min_frequency` times in training are pooled into an
    'infrequent' column; levels unseen in training (2024/2025) map there too.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_frequency,
                                  sparse_output=True), CATEGORICAL_FEATURES),
        ]
    )


def make_tree_encoder() -> ColumnTransformer:
    """Random Forest / Gradient Boosting: numerics unscaled, ordinal codes.

    Levels unseen in training are encoded as -1.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1, dtype=np.float64),
             CATEGORICAL_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# 5. Loading helper for model owners
# ---------------------------------------------------------------------------
def load_splits(folder: str = "stage2_outputs"):
    """Return X_train, y_train, X_val, y_val, X_test, y_test."""
    from pathlib import Path
    folder = Path(folder)
    out = []
    for name in ["train", "val", "test"]:
        X = pd.read_parquet(folder / f"X_{name}.parquet")
        y = pd.read_parquet(folder / f"y_{name}.parquet")[TARGET]
        out += [X, y]
    return tuple(out)
