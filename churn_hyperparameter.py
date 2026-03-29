import warnings
warnings.filterwarnings("ignore")  # Hide warnings to keep output clean
import glob
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def main():

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    churn_colors = ["#2ecc71", "#e74c3c"]
    model_colors = {
        "Logistic Regression": "#3498db",
        "Decision Tree": "#e67e22",
        "Random Forest": "#2ecc71",
        "Gradient Boosting": "#e74c3c",
        "SVM (RBF)": "#9b59b6",
        "MLP": "#1abc9c",
    }


    def save_figure(filename: str) -> None:
        plt.savefig(f"figures_ieee/{filename}")
        plt.close()


    def manual_smote(X, y, k=5, random_state=42):
        np.random.seed(random_state)

        X_array = X.values if hasattr(X, "values") else X
        y_array = y.values if hasattr(y, "values") else y

        X_minority = X_array[y_array == 1]
        n_to_generate = int((y_array == 0).sum()) - X_minority.shape[0]

        if n_to_generate <= 0:
            return X_array, y_array

        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_minority)
        _, neighbor_idx = nn.kneighbors(X_minority)

        synthetic_samples = []
        for _ in range(n_to_generate):
            i = np.random.randint(len(X_minority))
            j = np.random.choice(neighbor_idx[i][1:])
            gap = np.random.random()
            sample = X_minority[i] + gap * (X_minority[j] - X_minority[i])
            synthetic_samples.append(sample)

        X_resampled = np.vstack([X_array, np.array(synthetic_samples)])
        y_resampled = np.hstack([y_array, np.ones(n_to_generate, dtype=int)])

        shuffle_idx = np.random.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]

    df = pd.read_csv("telco_customer_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # All my figures will be saved in figures_ieee/ and results in results/ 
    # Please check those folders after running this script to see the outputs.
    # All my other print statements are for logging and debugging purposes.
    # Fig 1a: Churn count
    fig, ax = plt.subplots(figsize=(6, 5))
    churn_counts = df["Churn"].value_counts()
    ax.bar(churn_counts.index, churn_counts.values, color=churn_colors, edgecolor="white", width=0.5)
    ax.set_title("Churn Distribution (Count)", fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Churn")

    for i, (_, count) in enumerate(churn_counts.items()):
        ax.text(i, count + 80, str(count), ha="center", fontweight="bold", fontsize=13)

    ax.set_ylim(0, churn_counts.max() * 1.15)
    plt.tight_layout()
    save_figure("fig01a_churn_bar.png")

    # Fig 1b: Churn percentage
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=churn_colors,
        startangle=90,
        explode=[0, 0.06],
        textprops={"fontsize": 14, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Churn Distribution (%)", fontweight="bold")
    plt.tight_layout()
    save_figure("fig01b_churn_pie.png")

    # Fig 2: categorical features
    categorical_features = {
        "Contract": "Contract Type",
        "InternetService": "Internet Service",
        "PaymentMethod": "Payment Method",
        "TechSupport": "Tech Support",
    }

    for feature, label in categorical_features.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        churn_table = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100
        churn_table.plot(kind="bar", ax=ax, color=churn_colors, edgecolor="white", width=0.7)

        ax.set_title(f"Churn Rate by {label}", fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel(label)
        ax.legend(title="Churn", loc="upper right")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, 105)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

        plt.tight_layout()
        save_figure(f"fig02_{feature.lower()}_churn.png")

    # Fig 3: numeric distributions
    df_numeric = df.copy()
    df_numeric["TotalCharges"] = df_numeric["TotalCharges"].fillna(df_numeric["TotalCharges"].median())

    for feature in ["tenure", "MonthlyCharges", "TotalCharges"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for churn_value, color in zip(["No", "Yes"], churn_colors):
            subset = df_numeric[df_numeric["Churn"] == churn_value][feature].dropna()
            ax.hist(
                subset,
                bins=30,
                alpha=0.6,
                color=color,
                label=f"Churn={churn_value}",
                edgecolor="white",
            )

        ax.set_title(f"Distribution of {feature} by Churn", fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        save_figure(f"fig03_{feature.lower()}_dist.png")

    # Fig 4: correlation heatmap
    df_corr = df.copy()
    df_corr["Churn_Num"] = (df_corr["Churn"] == "Yes").astype(int)
    df_corr["TotalCharges"] = df_corr["TotalCharges"].fillna(df_corr["TotalCharges"].median())

    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df_corr[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 12},
    )
    ax.set_title("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    save_figure("fig04_correlation_heatmap.png")

    # Fig 5: scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for churn_value, color, marker in zip(["No", "Yes"], churn_colors, ["o", "x"]):
        subset = df[df["Churn"] == churn_value]
        ax.scatter(
            subset["tenure"],
            subset["MonthlyCharges"],
            c=color,
            alpha=0.3,
            label=f"Churn={churn_value}",
            s=15,
            marker=marker,
        )

    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Tenure vs Monthly Charges", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_figure("fig05_tenure_vs_charges.png")

    # Fig 6: demographic features
    for feature, label in [("SeniorCitizen", "Senior Citizen"), ("gender", "Gender")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        churn_table = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100
        churn_table.plot(kind="bar", ax=ax, color=churn_colors, edgecolor="white", width=0.5)

        ax.set_title(f"Churn Rate by {label}", fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel(label)
        ax.legend(title="Churn")
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylim(0, 105)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

        plt.tight_layout()
        save_figure(f"fig06_{feature.lower()}_churn.png")

    # Fig 7: engineered feature visuals
    df_eng = df.copy()
    df_eng["TotalCharges"] = pd.to_numeric(df_eng["TotalCharges"], errors="coerce").fillna(0)

    df_eng["TenureGroup"] = pd.cut(
        df_eng["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df_eng["NumServices"] = sum(
        (df_eng[col].isin(["Yes", "DSL", "Fiber optic"])).astype(int) for col in service_cols
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    tenure_group_table = pd.crosstab(df_eng["TenureGroup"], df_eng["Churn"], normalize="index") * 100
    tenure_group_table.plot(kind="bar", ax=ax, color=churn_colors, edgecolor="white", width=0.6)

    ax.set_title("Churn Rate by Tenure Group", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Tenure Group")
    ax.legend(title="Churn")
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

    plt.tight_layout()
    save_figure("fig07a_tenure_group.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    num_services_table = pd.crosstab(df_eng["NumServices"], df_eng["Churn"], normalize="index") * 100
    num_services_table[["Yes"]].plot(
        kind="bar",
        ax=ax,
        color="#e74c3c",
        edgecolor="white",
        legend=False,
        width=0.7,
    )

    ax.set_title("Churn Rate by Number of Services", fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Number of Services")
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

    plt.tight_layout()
    save_figure("fig07b_numservices.png")

    dm = df.copy()
    dm["TotalCharges"] = pd.to_numeric(dm["TotalCharges"], errors="coerce")
    dm["TotalCharges"] = dm["TotalCharges"].fillna(dm["TotalCharges"].median())

    dm["AvgMonthlySpend"] = dm["TotalCharges"] / dm["tenure"].replace(0, 1)
    dm["TenureGroup"] = pd.cut(
        dm["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    dm["NumServices"] = sum(
        (dm[col].isin(["Yes", "DSL", "Fiber optic"])).astype(int) for col in service_cols
    )
    dm["HasSecurityBundle"] = (
        (dm["OnlineSecurity"] == "Yes")
        & (dm["TechSupport"] == "Yes")
        & (dm["DeviceProtection"] == "Yes")
    ).astype(int)
    dm["HasStreamingBundle"] = (
        (dm["StreamingTV"] == "Yes")
        & (dm["StreamingMovies"] == "Yes")
    ).astype(int)
    dm["ChargeTenureRatio"] = dm["MonthlyCharges"] / dm["tenure"].replace(0, 1)

    dm = dm.drop("customerID", axis=1)
    dm["Churn"] = (dm["Churn"] == "Yes").astype(int)

    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        dm[col] = LabelEncoder().fit_transform(dm[col])

    dm = pd.get_dummies(
        dm,
        columns=[
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
            "TenureGroup",
        ],
        drop_first=True,
    )

    X = dm.drop("Churn", axis=1)
    y = dm["Churn"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    numeric_to_scale = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpend",
        "NumServices",
        "ChargeTenureRatio",
    ]

    scaler = StandardScaler()
    X[numeric_to_scale] = scaler.fit_transform(X[numeric_to_scale])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    print(f"Feature count: {X_train.shape[1]}")

    print("Hyperparameter tuning...")

    cv_tuning = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Gradient Boosting
    gb_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_param_grid,
        cv=cv_tuning,
        scoring="f1",
        n_jobs=-1,
    )
    gb_grid.fit(X_train, y_train)
    print(f"Gradient Boosting best parameters: {gb_grid.best_params_}")
    print(f"Gradient Boosting best CV F1: {gb_grid.best_score_:.4f}")

    # Random Forest
    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_param_grid,
        cv=cv_tuning,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"Random Forest best parameters: {rf_grid.best_params_}")
    print(f"Random Forest best CV F1: {rf_grid.best_score_:.4f}")

    # SVM
    svm_param_grid = {
        "C": [1, 10, 50],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"],
    }
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42, max_iter=5000),
        svm_param_grid,
        cv=cv_tuning,
        scoring="f1",
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)
    print(f"SVM best parameters: {svm_grid.best_params_}")
    print(f"SVM best CV F1: {svm_grid.best_score_:.4f}")

    # Logistic Regression
    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    }
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_param_grid,
        cv=cv_tuning,
        scoring="f1",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    print(f"Logistic Regression best parameters: {lr_grid.best_params_}")
    print(f"Logistic Regression best CV F1: {lr_grid.best_score_:.4f}")

    # Decision Tree
    dt_param_grid = {
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid,
        cv=cv_tuning,
        scoring="f1",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    print(f"Decision Tree best parameters: {dt_grid.best_params_}")
    print(f"Decision Tree best CV F1: {dt_grid.best_score_:.4f}")

    # MLP
    best_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )

    models = {
        "Logistic Regression": lr_grid.best_estimator_,
        "Decision Tree": dt_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
        "Gradient Boosting": gb_grid.best_estimator_,
        "SVM (RBF)": svm_grid.best_estimator_,
        "MLP": best_mlp,
    }

    tuning_results = []
    for name, grid in [
        ("Logistic Regression", lr_grid),
        ("Decision Tree", dt_grid),
        ("Random Forest", rf_grid),
        ("Gradient Boosting", gb_grid),
        ("SVM (RBF)", svm_grid),
    ]:
        tuning_results.append(
            {
                "Model": name,
                "Best_Params": str(grid.best_params_),
                "Best_CV_F1": f"{grid.best_score_:.4f}",
            }
        )

    tuning_results.append(
        {
            "Model": "MLP",
            "Best_Params": "{'hidden_layer_sizes': (128,64,32), 'lr': 0.001, 'early_stopping': True}",
            "Best_CV_F1": "N/A (manual)",
        }
    )

    pd.DataFrame(tuning_results).to_csv("results/hyperparameter_tuning.csv", index=False)

    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_eval, scoring="f1")

        results[model_name] = {
            "acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_score),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_score": y_score,
            "model": model,
        }

        print(
            f"{model_name}: "
            f"Acc={results[model_name]['acc']:.4f}, "
            f"F1={results[model_name]['f1']:.4f}, "
            f"AUC={results[model_name]['auc']:.4f}"
        )

    comparison_rows = []
    for model_name, result in results.items():
        comparison_rows.append(
            {
                "Model": model_name,
                "Train Acc": "-",
                "Val Acc": "-",
                "Test Acc": f"{result['acc']:.4f}",
                "Precision": f"{result['precision']:.4f}",
                "Recall": f"{result['recall']:.4f}",
                "F1-Score": f"{result['f1']:.4f}",
                "ROC-AUC": f"{result['auc']:.4f}",
                "CV F1": f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}",
            }
        )

    pd.DataFrame(comparison_rows).to_csv("results/model_comparison_tuned.csv", index=False)

    # Fig 8: model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df = pd.DataFrame(
        {
            name: {
                "Accuracy": res["acc"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1"],
                "ROC-AUC": res["auc"],
            }
            for name, res in results.items()
        }
    ).T

    x = np.arange(len(metrics_df))
    bar_width = 0.15
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, (metric_name, color) in enumerate(zip(metrics_df.columns, bar_colors)):
        ax.bar(x + i * bar_width, metrics_df[metric_name], bar_width, label=metric_name, color=color, edgecolor="white")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(metrics_df.index, rotation=20, ha="right")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure("fig08_model_comparison.png")

    # Fig 9: confusion matrices
    for model_name, result in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, result["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            cbar=True,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 16, "fontweight": "bold"},
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

        filename = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.tight_layout()
        save_figure(f"fig09_cm_{filename}.png")

    # Fig 10: ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_score"])
        ax.plot(
            fpr,
            tpr,
            color=model_colors[model_name],
            lw=2.5,
            label=f"{model_name} (AUC={result['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure("fig10_roc_curves.png")

    # Fig 11: CV scores
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(results.keys())
    cv_means = [results[name]["cv_mean"] for name in model_names]
    cv_stds = [results[name]["cv_std"] for name in model_names]

    ax.barh(
        model_names,
        cv_means,
        xerr=cv_stds,
        color=[model_colors[name] for name in model_names],
        edgecolor="white",
        height=0.55,
        capsize=5,
    )
    ax.set_xlabel("F1-Score (5-Fold CV)")
    ax.set_title("Cross-Validation F1-Scores", fontweight="bold")
    ax.set_xlim(0.35, 0.75)

    for i, (mean_score, std_score) in enumerate(zip(cv_means, cv_stds)):
        ax.text(mean_score + std_score + 0.008, i, f"{mean_score:.3f}±{std_score:.3f}", va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_figure("fig11_cv_scores.png")

    # Fig 12: feature importance
    feature_importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": results["Gradient Boosting"]["model"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(15), feature_importance["Importance"].values, color="#3498db", edgecolor="white")
    ax.set_yticks(range(15))
    ax.set_yticklabels(feature_importance["Feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gradient Boosting)")
    ax.set_title("Top 15 Features", fontweight="bold")

    for i, value in enumerate(feature_importance["Importance"].values):
        ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_figure("fig12_feature_importance.png")

    X_smote, y_smote = manual_smote(X_train, y_train)
    print(f"SMOTE sample count: {len(y_smote)}")
    print(f"SMOTE churn count: {(y_smote == 1).sum()}")
    print(f"SMOTE non-churn count: {(y_smote == 0).sum()}")

    results_before_smote = {}
    results_after_smote = {}

    for model_name in models:
        print(f"Evaluating {model_name} with and without SMOTE...")

        # Before SMOTE
        model_before = deepcopy(models[model_name])
        model_before.fit(X_train, y_train)
        y_pred_before = model_before.predict(X_test)
        if hasattr(model_before, "predict_proba"):
            y_score_before = model_before.predict_proba(X_test)[:, 1]
        else:
            y_score_before = model_before.decision_function(X_test)

        results_before_smote[model_name] = {
            "f1": f1_score(y_test, y_pred_before),
            "recall": recall_score(y_test, y_pred_before),
            "acc": accuracy_score(y_test, y_pred_before),
            "auc": roc_auc_score(y_test, y_score_before),
            "precision": precision_score(y_test, y_pred_before),
            "y_pred": y_pred_before,
            "y_score": y_score_before,
        }

        # After SMOTE
        model_after = deepcopy(models[model_name])
        model_after.fit(X_smote, y_smote)
        y_pred_after = model_after.predict(X_test)
        if hasattr(model_after, "predict_proba"):
            y_score_after = model_after.predict_proba(X_test)[:, 1]
        else:
            y_score_after = model_after.decision_function(X_test)

        results_after_smote[model_name] = {
            "f1": f1_score(y_test, y_pred_after),
            "recall": recall_score(y_test, y_pred_after),
            "acc": accuracy_score(y_test, y_pred_after),
            "auc": roc_auc_score(y_test, y_score_after),
            "precision": precision_score(y_test, y_pred_after),
            "y_pred": y_pred_after,
            "y_score": y_score_after,
        }

        print(
            f"{model_name}: "
            f"before F1={results_before_smote[model_name]['f1']:.4f}, "
            f"before recall={results_before_smote[model_name]['recall']:.4f}, "
            f"after F1={results_after_smote[model_name]['f1']:.4f}, "
            f"after recall={results_after_smote[model_name]['recall']:.4f}"
        )

    smote_rows = []
    for model_name, result in results_before_smote.items():
        smote_rows.append(
            {
                "Model": model_name,
                "Accuracy": f"{result['acc']:.4f}",
                "Precision": f"{result['precision']:.4f}",
                "Recall": f"{result['recall']:.4f}",
                "F1-Score": f"{result['f1']:.4f}",
                "ROC-AUC": f"{result['auc']:.4f}",
                "SMOTE": "Before",
            }
        )

    for model_name, result in results_after_smote.items():
        smote_rows.append(
            {
                "Model": model_name,
                "Accuracy": f"{result['acc']:.4f}",
                "Precision": f"{result['precision']:.4f}",
                "Recall": f"{result['recall']:.4f}",
                "F1-Score": f"{result['f1']:.4f}",
                "ROC-AUC": f"{result['auc']:.4f}",
                "SMOTE": "After",
            }
        )

    pd.DataFrame(smote_rows).to_csv("results/smote_comparison_tuned.csv", index=False)

    print("Generating SMOTE figures...")

    model_names = list(results_before_smote.keys())
    x = np.arange(len(model_names))
    bar_width = 0.35

    # Fig 13a: F1 before vs after SMOTE
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - bar_width / 2,
        [results_before_smote[name]["f1"] for name in model_names],
        bar_width,
        label="Before SMOTE",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2,
        [results_after_smote[name]["f1"] for name in model_names],
        bar_width,
        label="After SMOTE",
        color="#2ecc71",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score: Before vs After SMOTE", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure("fig13a_smote_f1.png")

    # Fig 13b: recall before vs after SMOTE
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - bar_width / 2,
        [results_before_smote[name]["recall"] for name in model_names],
        bar_width,
        label="Before SMOTE",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2,
        [results_after_smote[name]["recall"] for name in model_names],
        bar_width,
        label="After SMOTE",
        color="#2ecc71",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_ylabel("Recall")
    ax.set_title("Recall: Before vs After SMOTE", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure("fig13b_smote_recall.png")

    # Fig 14: ROC after SMOTE
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, result in results_after_smote.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_score"])
        ax.plot(
            fpr,
            tpr,
            color=model_colors[model_name],
            lw=2.5,
            label=f"{model_name} (AUC={result['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves After SMOTE", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure("fig14_roc_after_smote.png")

    # Fig 15: confusion matrices after SMOTE
    for model_name, result in results_after_smote.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, result["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            cbar=True,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 16, "fontweight": "bold"},
        )
        ax.set_title(
            f"CM (SMOTE) — {model_name}\nF1={result['f1']:.3f}, AUC={result['auc']:.3f}",
            fontweight="bold",
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

        filename = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.tight_layout()
        save_figure(f"fig15_cm_smote_{filename}.png")

    all_figures = sorted(glob.glob("figures_ieee/*.png"))
    print(f"Generated {len(all_figures)} figures.")

if __name__ == "__main__":
    main()