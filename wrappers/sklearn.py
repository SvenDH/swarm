"""Explicit estimators with safe learned-array state (no pickle or refitting)."""
import numpy as np
from sklearn import preprocessing, model_selection, metrics, linear_model, cluster, decomposition
from .bridge import descriptor
from .common import freeze, thaw, view, iterate, assign
from .api import select

FACTORIES = {
    'StandardScaler': preprocessing.StandardScaler, 'MinMaxScaler': preprocessing.MinMaxScaler,
    'LabelEncoder': preprocessing.LabelEncoder, 'OneHotEncoder': preprocessing.OneHotEncoder,
    'LinearRegression': linear_model.LinearRegression, 'Ridge': linear_model.Ridge,
    'LogisticRegression': linear_model.LogisticRegression, 'KMeans': cluster.KMeans,
    'PCA': decomposition.PCA,
}
# Only these known fitted fields are saved/restored. Never arbitrary native objects.
FIELDS = {
    'StandardScaler': ('scale_', 'mean_', 'var_', 'n_features_in_', 'n_samples_seen_', 'feature_names_in_'),
    'MinMaxScaler': ('min_', 'scale_', 'data_min_', 'data_max_', 'data_range_', 'n_features_in_', 'n_samples_seen_', 'feature_names_in_'),
    'LabelEncoder': ('classes_',),
    'OneHotEncoder': ('categories_', 'drop_idx_', '_drop_idx_after_grouping', '_n_features_outs', '_infrequent_enabled', 'n_features_in_', 'feature_names_in_'),
    'LinearRegression': ('coef_', 'intercept_', 'rank_', 'singular_', 'n_features_in_', 'feature_names_in_'),
    'Ridge': ('coef_', 'intercept_', 'n_iter_', 'n_features_in_', 'feature_names_in_', 'solver_'),
    'LogisticRegression': ('coef_', 'intercept_', 'classes_', 'n_iter_', 'n_features_in_', 'feature_names_in_'),
    'KMeans': ('cluster_centers_', 'labels_', 'inertia_', 'n_iter_', 'n_features_in_', '_n_threads', '_n_features_out', 'feature_names_in_'),
    'PCA': ('components_', 'explained_variance_', 'explained_variance_ratio_', 'singular_values_',
            'mean_', 'n_components_', 'n_samples_', 'noise_variance_', 'n_features_in_', 'feature_names_in_'),
}


class Estimator:
    def __init__(self, name, params=None, state=None):
        self.name, self.params, self.state = name, params or {}, state or {}

    def native(self):
        result = FACTORIES[self.name](**self.params)
        result.__dict__.update(thaw(self.state))
        return result

    def call(self, name, *args, **kwargs):
        model = self.native()
        methods = {'fit': model.fit, 'fit_transform': model.fit_transform} if self.name in (
            'StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder', 'PCA', 'KMeans') else {'fit': model.fit}
        if self.name in ('StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder', 'PCA', 'KMeans'):
            methods['transform'] = model.transform
            if self.name != 'KMeans':
                methods['inverse_transform'] = model.inverse_transform
        if self.name in ('LinearRegression', 'Ridge', 'LogisticRegression', 'KMeans'):
            methods.update(predict=model.predict, score=model.score)
        if self.name == 'LogisticRegression':
            methods['predict_proba'] = model.predict_proba
        result = methods[name](*args, **kwargs)
        if name.startswith('fit'):
            self.state = {field: freeze(model.__dict__[field]) for field in FIELDS[self.name] if field in model.__dict__}
        return self if result is model else result


def factory(name):
    def create(**kwargs):
        if kwargs.get('verbose', 0) or kwargs.get('n_jobs', 1) not in (None, 1):
            raise ValueError('Native verbose output and parallel job pools are not exposed')
        # Sparse outputs and infrequent category internals are deliberately excluded.
        if name == 'OneHotEncoder':
            if kwargs.get('sparse_output', False) or kwargs.get('min_frequency') is not None or kwargs.get('max_categories') is not None:
                raise ValueError('OneHotEncoder supports dense output and ordinary categories only')
            kwargs['sparse_output'] = False
        if name in ('KMeans', 'PCA', 'LogisticRegression'):
            kwargs.setdefault('random_state', 0)
        FACTORIES[name](**kwargs)  # Validate constructor arguments now.
        return Estimator(name, kwargs)
    return create


def exports(fs):
    def split(*arrays, **kwargs):
        kwargs.setdefault('random_state', 0)
        return model_selection.train_test_split(*arrays, **kwargs)
    return {'preprocessing': {**{name: factory(name) for name in ('StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder')},
                              **select(vars(preprocessing), 'robust_scale minmax_scale maxabs_scale binarize label_binarize'),
                              'normalize': preprocessing.normalize, 'scale': preprocessing.scale},
            'model_selection': {'train_test_split': split},
            'metrics': {**select(vars(metrics), '''balanced_accuracy_score roc_auc_score roc_curve auc
                        precision_recall_curve average_precision_score log_loss brier_score_loss
                        matthews_corrcoef cohen_kappa_score hamming_loss zero_one_loss jaccard_score
                        fbeta_score precision_recall_fscore_support multilabel_confusion_matrix
                        median_absolute_error mean_absolute_percentage_error explained_variance_score
                        max_error mean_squared_log_error adjusted_rand_score adjusted_mutual_info_score
                        normalized_mutual_info_score homogeneity_score completeness_score v_measure_score
                        silhouette_score calinski_harabasz_score davies_bouldin_score'''),
                        'accuracy_score': metrics.accuracy_score, 'precision_score': metrics.precision_score,
                        'recall_score': metrics.recall_score, 'f1_score': metrics.f1_score,
                        'mean_squared_error': metrics.mean_squared_error, 'mean_absolute_error': metrics.mean_absolute_error,
                        'r2_score': metrics.r2_score, 'confusion_matrix': metrics.confusion_matrix,
                        'classification_report': metrics.classification_report},
            'linear_model': {name: factory(name) for name in ('LinearRegression', 'Ridge', 'LogisticRegression')},
            'cluster': {'KMeans': factory('KMeans')}, 'decomposition': {'PCA': factory('PCA')}}


def members(value, fs, name=None):
    if isinstance(value, Estimator):
        if name in value.state and not name.startswith('_'):
            item = thaw(value.state[name])
            if isinstance(item, np.ndarray) and item.dtype.hasobject:
                item = item.tolist()
            return {name: item}
        allowed = {'fit'}
        if value.name in ('StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder', 'PCA', 'KMeans'):
            allowed.update(('fit_transform', 'transform'))
            if value.name != 'KMeans':
                allowed.add('inverse_transform')
        if value.name in ('LinearRegression', 'Ridge', 'LogisticRegression', 'KMeans'):
            allowed.update(('predict', 'score'))
        if value.name == 'LogisticRegression':
            allowed.add('predict_proba')
        return {method: (lambda *a, _method=method, **kw: value.call(_method, *a, **kw)) for method in allowed}
    return {}


def pack(value):
    if isinstance(value, Estimator):
        return descriptor('sklearn', 'estimator', [value.name, freeze(value.params), value.state])


def restore(kind, data):
    if kind != 'estimator' or data[0] not in FACTORIES:
        raise ValueError('Unsupported estimator')
    return Estimator(data[0], thaw(data[1]), data[2])


def is_view(value):
    return False
