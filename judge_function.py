from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, average_precision_score
from scipy.stats import pearsonr, spearmanr
import oddt.metrics as vsmetrics
import numpy as np

metric_functions = {
    'AUROC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    'AUPRC': lambda y_true, y_pred: average_precision_score(y_true, y_pred),
    'BEDROC': lambda y_true, y_pred: vsmetrics.bedroc(y_true, y_pred, alpha=160.9, pos_label=1),
    'EF1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=1, pos_label=1, kind='fold'),
    'EF0.5%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.5, pos_label=1, kind='fold'),
    'EF0.1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.1, pos_label=1, kind='fold'),
    'EF0.05%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.05, pos_label=1, kind='fold'),
    'EF0.01%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.01, pos_label=1, kind='fold'),
    'logAUC': lambda y_true, y_pred: vsmetrics.roc_log_auc(y_true, y_pred, pos_label=1, ascending_score=False, log_min=0.001, log_max=1.0),
    'MSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    'MAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'SPEARMANR': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
    'PEARSONR': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    'ACC': lambda y_true, y_pred: accuracy_score(y_true, [round(num) for num in y_pred]),
    'specificity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred], pos_label=0),
    'precision': lambda y_true, y_pred: precision_score(y_true, [round(num) for num in y_pred]),
    'recall': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]),
    'sensitivity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]),
    'f1': lambda y_true, y_pred: f1_score(y_true, [round(num) for num in y_pred]),
}