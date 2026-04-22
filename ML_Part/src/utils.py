import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def plot_class_distribution(df, dataset_name, column='BL'):
    """
    Plots the class distribution of a given feature/column.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Class Distribution for {dataset_name} - {column}')
    plt.show()

def correlaciones(dataframe, lista_corr):
    """
    Plots correlation heatmaps for the provided methods (e.g. ['pearson', 'spearman']).
    """
    cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
    for i in lista_corr:
        corr = dataframe.corr(method=i)
        mask = np.zeros_like(corr.round(6))
        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(figsize=(12, 10))
            ax = sns.heatmap(corr.round(4),
                             mask=mask,
                             vmax=1,
                             center=0,
                             vmin=-1,
                             square=True,
                             cmap=cmap,
                             linewidths=.5,
                             annot=True,
                             annot_kws={"size": 12},
                             fmt='.2f')

            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Features', fontsize=15)
            plt.title(f'Heatmap for {i.capitalize()} Correlations of the Features', fontsize=15)
        plt.show()

def _arr_size(x):
    try:
        return np.array(x).size
    except Exception:
        return 0

def estimate_param_count(model):
    """
    Estimates the number of parameters or nodes in a scikit-learn compatible model.
    """
    if hasattr(model, "estimators_") and getattr(model, "estimators_", None) is not None:
        total = 0
        for est in np.array(model.estimators_, dtype=object).ravel():
            if est is not None:
                total += estimate_param_count(est)
        if total > 0:
            return total

    if hasattr(model, "tree_") and getattr(model, "tree_", None) is not None:
        return int(model.tree_.node_count)

    total = 0
    for attr in ["coef_", "intercept_", "feature_importances_", "singular_values_"]:
        if hasattr(model, attr):
            total += _arr_size(getattr(model, attr))
    if total > 0:
        return int(total)

    try:
        return len(model.get_params(deep=True))
    except Exception:
        return 1

def xgb_complexity(model):
    if hasattr(model, "named_steps"): 
        model = list(model.named_steps.values())[-1]

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif hasattr(model, "booster_"):
        booster = model.booster_
    else:
        raise ValueError("This model doesn't look like a fitted XGBoost sklearn model.")

    trees = booster.get_dump(dump_format="json")
    n_trees = len(trees)

    total_nodes = 0
    total_leaves = 0
    
    # We define inline counting to avoid recursion depth issues
    for t in trees:
        tree_dict = json.loads(t)
        stack = [tree_dict]
        while stack:
            node = stack.pop()
            total_nodes += 1
            if "leaf" in node:
                total_leaves += 1
            else:
                for child in node.get("children", []):
                    stack.append(child)

    return {
        "xgb_n_trees": n_trees,
        "xgb_total_nodes": total_nodes,
        "xgb_total_leaves": total_leaves,
        "complexity": total_nodes
    }

def lgbm_complexity(model):
    if hasattr(model, "named_steps"): 
        model = list(model.named_steps.values())[-1]

    if hasattr(model, "booster_"):
        booster = model.booster_
    elif hasattr(model, "_Booster"):
        booster = model._Booster
    else:
        raise ValueError("This model doesn't look like a fitted LightGBM sklearn model.")

    dump = booster.dump_model()
    tree_info = dump.get("tree_info", [])
    n_trees = len(tree_info)

    total_nodes = 0
    total_leaves = 0

    for t in tree_info:
        total_leaves += t.get("num_leaves", 0)
        total_nodes += t.get("num_leaves", 0) - 1 if t.get("num_leaves", 0) else 0 

    return {
        "lgbm_n_trees": n_trees,
        "lgbm_total_leaves": total_leaves,
        "lgbm_total_nodes_approx": total_nodes,
        "complexity": total_leaves
    }

def model_complexity(model):
    """
    Returns an integer representing the complexity proxy of any PyCaret model.
    """
    name = type(model).__name__.lower()

    if hasattr(model, "named_steps"):
        model = list(model.named_steps.values())[-1]
        name = type(model).__name__.lower()

    if hasattr(model, "get_booster") or hasattr(model, "booster_"):
        try:
            return xgb_complexity(model)["complexity"]
        except Exception:
            pass

    if hasattr(model, "booster_") or hasattr(model, "_Booster"):
        try:
            return lgbm_complexity(model)["complexity"]
        except Exception:
            pass

    return estimate_param_count(model)
