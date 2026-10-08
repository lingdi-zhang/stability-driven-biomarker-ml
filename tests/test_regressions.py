from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from biomarkerML_pipeline import BiomarkerMLPipeline, format_model_parameters
from biomarkerML_pipeline.config import parse_config_value
from biomarkerML_pipeline.inner_cv import run_inner_cv
from biomarkerML_pipeline.outer_cv import run_outer_cv
from biomarkerML_pipeline.evaluation import compute_metrics


def configuration(tune=False):
    return pd.DataFrame([['EN', 'LogisticRegression(random_state=42, max_iter=1000)',
                          'model__C', '0.5', 'roc_auc', tune]],
                        columns=['Model_name', 'Base_model', 'Parameters', 'Parameter_values', 'Scoring', 'tune_model'])


@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=6, n_informative=3, random_state=42)
    return pd.DataFrame(X, columns=list('abcdef')), y


def test_untuned_uses_constructor_settings(data):
    X, y = data
    pipe, params, score = run_inner_cv(X, y, 1, SVC(kernel='linear', probability=True, C=2),
                                      {'model__C': [100], 'model__kernel': ['rbf']}, 'roc_auc', False)
    pipe.fit(X, y)
    assert pipe.named_steps['model'].C == 2
    assert pipe.named_steps['model'].coef_.shape == (1, 6)
    assert pipe.named_steps['scaler'] == 'passthrough'
    assert params is score is None


@pytest.mark.parametrize('model,grid,tune', [
    (SVC(probability=True), {}, False),
    (SVC(kernel='linear', probability=True), {'model__kernel': ['linear', 'rbf']}, True),
    (SVC(kernel='linear'), {}, False),
])
def test_svc_validation(data, model, grid, tune):
    with pytest.raises(ValueError):
        run_inner_cv(*data, 1, model, grid, 'roc_auc', tune)


def test_safe_config():
    params = format_model_parameters(configuration())
    assert params['EN']['input_model'].random_state == 42
    np.testing.assert_allclose(parse_config_value('np.logspace(-2, 0, 3)'), [.01, .1, 1])
    assert parse_config_value('StandardScaler()').with_mean
    for expression in ["__import__('os').system('echo unsafe')", 'open("/tmp/unsafe", "w")',
                       'LogisticRegression(**{})', 'np.load("/tmp/data")']:
        with pytest.raises(ValueError):
            parse_config_value(expression)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'boolean', 'malformed'])
def test_invalid_configuration(change):
    config = configuration(True)
    if change == 'missing':
        config = config.drop(columns='Base_model')
    elif change == 'duplicate':
        config = pd.concat([config, config])
    elif change == 'boolean':
        config['tune_model'] = 'sometimes'
    else:
        config['Parameter_values'] = 'unknown()'
    with pytest.raises(ValueError):
        format_model_parameters(config)


@pytest.mark.parametrize('labels', ['numeric', 'strings'])
def test_label_mapping_and_holdout(data, tmp_path, labels):
    X, y = data
    y = y + 1 if labels == 'numeric' else np.where(y, 'case', 'control')
    pipeline = BiomarkerMLPipeline(configuration(), output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X.iloc[:80], y[:80], cross_folds_selection=0)
    result = pipeline.prediction_with_stable_biomarkers_across_models(X.iloc[:80], y[:80], X.iloc[80:], y[80:])
    assert (result.status == 'ok').all()
    assert np.isfinite(result[['roc_auc', 'pr_auc', 'f1']]).all().all()
    assert len(pipeline.classes_) == 2
    unknown = np.full(20, 999) if labels == 'numeric' else np.full(20, 'unknown')
    with pytest.raises(ValueError, match='target classes'):
        pipeline.prediction_with_stable_biomarkers_across_models(X.iloc[:80], y[:80], X.iloc[80:], unknown)
    with pytest.raises(ValueError, match='columns'):
        pipeline.prediction_with_stable_biomarkers_across_models(X.iloc[:80], y[:80], X.iloc[80:].drop(columns='a'), y[80:])


@pytest.mark.parametrize('invalid', ['missing', 'infinite', 'few', 'multiclass', 'length'])
def test_invalid_inputs(data, tmp_path, invalid):
    X, y = data
    if invalid == 'missing':
        X.loc[0, 'a'] = np.nan
    elif invalid == 'infinite':
        X.loc[0, 'a'] = np.inf
    elif invalid == 'few':
        y = np.array([0] * 96 + [1] * 4)
    elif invalid == 'multiclass':
        y[0] = 2
    else:
        y = y[:-1]
    with pytest.raises(ValueError):
        BiomarkerMLPipeline(configuration(), output_dir=tmp_path).stable_biomarker_selection(X, y)


def test_empty_biomarkers(data, tmp_path):
    X, y = data
    config = configuration()
    config['Base_model'] = "LogisticRegression(penalty='l1', solver='liblinear', C=1e-10)"
    pipeline = BiomarkerMLPipeline(config, output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X.iloc[:80], y[:80])
    result = pipeline.prediction_with_stable_biomarkers_across_models(X.iloc[:80], y[:80], X.iloc[80:], y[80:])
    assert (result.status == 'no_features').all()
    assert (result.n_features == 0).all()
    assert result[['roc_auc', 'pr_auc', 'f1']].isna().all().all()
    assert (tmp_path / 'Model_biomarker_list.csv').exists()
    assert not (tmp_path / 'feature_selection_consensus_heatmap.png').exists()


def test_metrics_require_encoded_labels():
    with pytest.raises(ValueError, match='encoded'):
        compute_metrics([1, 2], [.1, .9])


@pytest.mark.parametrize('kind', ['RF', 'XGBoost'])
def test_tree_model_tuning_and_importance(data, kind):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    model = (RandomForestClassifier(n_estimators=8, random_state=42, n_jobs=1)
             if kind == 'RF' else XGBClassifier(n_estimators=8, random_state=42, n_jobs=1))
    params = dict(input_model=model, input_param_grid={'model__max_depth': [2]},
                  scoring='average_precision', tune_model=True)
    X, y = data
    result = run_outer_cv(X, y, np.arange(80), np.arange(80, 100),
                          outer_cv_params=params)
    assert len(result['gene_list']) == len(result['feature_model_importance']) == 6
    assert np.isfinite(result['feature_model_importance']).all()
    assert 0 <= result['roc_auc'] <= 1


def test_importance_uses_all_measured_scores(tmp_path):
    from biomarkerML_pipeline.outer_cv_analysis import summarize_outer_cv_results
    values = [10., 8., 1., 1., 1.]
    folds = {f'fold_{i}': dict(gene_list=['a', 'b', 'zero'],
                              feature_model_importance=[value, 1. if i < 2 else 10., 0.],
                              roc_auc=.8, pr_auc=.8, f1=.8)
             for i, value in enumerate(values)}
    genes, *_ = summarize_outer_cv_results('EN', folds, .75, .4, tmp_path)
    raw = pd.read_csv(tmp_path / 'Feature_importance_by_fold_EN.csv')
    assert len(raw) == 15
    assert raw.loc[raw.gene == 'a', 'selected'].sum() == 2
    assert not raw.loc[raw.gene == 'zero', 'selected'].any()
    summary = pd.read_csv(tmp_path / 'Feature_stability_and_importance_summary_EN.csv').set_index('gene')
    assert summary.loc['a', 'mean_importance'] == pytest.approx(4.2)
    assert summary.loc['a', 'sd_importance'] == pytest.approx(np.std(values, ddof=1))
    assert summary.loc['a', 'measured_folds'] == 5
    assert summary.loc['a', 'appearances'] == 2
    assert summary.loc['a', 'percent_appearance'] == .4
    assert summary.loc['zero', 'mean_importance'] == 0
    assert summary.loc['zero', 'measured_folds'] == 5
    assert summary.loc['zero', 'appearances'] == 0
    assert set(genes) == {'a', 'b'}


def test_missing_scores_are_not_zero_measurements():
    from biomarkerML_pipeline.evaluation import summarize_feature_importance
    scores = pd.DataFrame({'gene': ['a', 'a', 'b'], 'fold': ['fold1', 'fold2', 'fold1'],
                           'abs_coef': [10., 8., 0.], 'selected': [True, True, False]})
    result = summarize_feature_importance(scores, 5).set_index('gene')
    assert result.loc['a', 'mean_importance'] == 9
    assert result.loc['a', 'sd_importance'] == pytest.approx(np.sqrt(2))
    assert result.loc['a', 'measured_folds'] == 2
    assert result.loc['a', 'percent_appearance'] == .4
    assert result.loc['b', 'mean_importance'] == 0
    assert result.loc['b', 'measured_folds'] == 1
    assert pd.isna(result.loc['b', 'sd_importance'])
    assert result.loc['b', 'appearances'] == 0


def test_zero_scores_remain_unselected_at_zero_stability_threshold(tmp_path):
    from biomarkerML_pipeline.outer_cv_analysis import summarize_outer_cv_results
    folds = {f'fold{i}': dict(gene_list=['a'], feature_model_importance=[0.],
                            roc_auc=.8, pr_auc=.8, f1=.8) for i in range(5)}
    genes, *_ = summarize_outer_cv_results('EN', folds, .75, 0, tmp_path)
    assert genes == []


@pytest.mark.parametrize('expression,expected', [
    ('logspace(-2, 0, 3)', [.01, .1, 1.]),
    ('np.logspace(-2, 0, 3)', [.01, .1, 1.]),
    ('arange(1, 4)', [1, 2, 3]),
    ('np.arange(1, 4)', [1, 2, 3]),
    ('0.01;0.1;1', [.01, .1, 1]),
    ('logspace(-2, -1, 2);1', [.01, .1, 1.]),
])
def test_array_expressions_expand_candidates(expression, expected):
    config = configuration(True)
    config['Parameter_values'] = expression
    params = format_model_parameters(config)['EN']
    np.testing.assert_allclose(params['input_param_grid']['model__C'], expected)
    assert all(np.isscalar(v) for v in params['input_param_grid']['model__C'])


def test_array_grid_runs_search_and_untuned_ignores_it(data):
    config = configuration(True)
    config['Parameter_values'] = 'logspace(-2, 0, 3)'
    params = format_model_parameters(config)['EN']
    fitted, best, score = run_inner_cv(*data, 1, **params)
    assert best['model__C'] in [.01, .1, 1.]
    assert np.isfinite(score)
    assert fitted.predict_proba(data[0]).shape == (100, 2)
    params['tune_model'] = False
    untuned, _, _ = run_inner_cv(*data, 1, **params)
    assert untuned.named_steps['model'].C == 1.0


def test_constructor_candidates_are_preserved():
    from sklearn.preprocessing import StandardScaler
    config = configuration(True)
    config['Parameters'] = 'scaler'
    config['Parameter_values'] = 'StandardScaler();"passthrough"'
    candidates = format_model_parameters(config)['EN']['input_param_grid']['scaler']
    assert len(candidates) == 2
    assert isinstance(candidates[0], StandardScaler)
    assert candidates[1] == 'passthrough'


def test_empty_array_grid_is_rejected():
    config = configuration(True)
    config['Parameter_values'] = 'arange(0)'
    with pytest.raises(ValueError, match='no candidate values'):
        format_model_parameters(config)


def test_multidimensional_array_grid_is_rejected(monkeypatch):
    import biomarkerML_pipeline.config as module
    monkeypatch.setitem(module.FUNCTIONS, 'arange', lambda *args: np.ones((2, 2)))
    config = configuration(True)
    config['Parameter_values'] = 'arange(2)'
    with pytest.raises(ValueError, match='one-dimensional array'):
        format_model_parameters(config)


@pytest.mark.parametrize('empty', [None, np.nan, '', '   '])
def test_untuned_blank_grids(data, empty):
    config = configuration(False)
    config['Parameters'] = empty
    config['Parameter_values'] = empty
    params = format_model_parameters(config)['EN']
    assert params['input_param_grid'] == {}
    pipe, _, _ = run_inner_cv(*data, 1, **params)
    pipe.fit(*data)
    assert pipe.named_steps['model'].C == 1.0
    assert pipe.predict_proba(data[0]).shape == (100, 2)


@pytest.mark.parametrize('column', ['Parameters', 'Parameter_values'])
@pytest.mark.parametrize('empty', [None, '', '   '])
def test_tuned_blank_grids_rejected(column, empty):
    config = configuration(True)
    config[column] = empty
    with pytest.raises(ValueError, match='blank when tuning is enabled'):
        format_model_parameters(config)


def test_untuned_grid_is_not_parsed(monkeypatch):
    import biomarkerML_pipeline.config as module
    original = module.parse_config_value
    seen = []
    def record(value):
        seen.append(value)
        return original(value)
    monkeypatch.setattr(module, 'parse_config_value', record)
    config = configuration(False)
    config['Parameter_values'] = 'unknown()'
    config = pd.concat([config, config], ignore_index=True)
    assert module.format_model_parameters(config)['EN']['input_param_grid'] == {}
    assert seen == [config.Base_model.iloc[0]]


def test_mixed_tuned_and_untuned_configurations():
    untuned = configuration(False)
    untuned['Model_name'] = 'untuned'
    untuned['Parameters'] = None
    untuned['Parameter_values'] = None
    tuned = configuration(True)
    tuned['Model_name'] = 'tuned'
    tuned['Parameter_values'] = 'logspace(-2, 0, 3)'
    params = format_model_parameters(pd.concat([untuned, tuned], ignore_index=True))
    assert params['untuned']['input_param_grid'] == {}
    np.testing.assert_allclose(params['tuned']['input_param_grid']['model__C'], [.01, .1, 1.])


def test_untuned_empty_grid_still_validates_svc():
    config = configuration(False)
    config['Parameters'] = None
    config['Parameter_values'] = None
    config['Base_model'] = 'SVC(probability=True)'
    with pytest.raises(ValueError, match='only linear SVC'):
        format_model_parameters(config)


@pytest.mark.parametrize('base,grid', [
    (SVC(kernel='linear', probability=True), {'model__probability': [True, False]}),
    (SVC(kernel='linear', probability=True), {'model': [SVC(kernel='rbf', probability=True)]}),
    (LogisticRegression(), {'model': [SVC(kernel='linear', probability=False)]}),
    (LogisticRegression(), [{'model': [SVC(kernel='linear', probability=True)]},
                            {'model': [SVC(kernel='poly', probability=True)]}]),
])
def test_invalid_svc_candidates_rejected_before_fit(data, monkeypatch, base, grid):
    def fail_fit(*args, **kwargs):
        raise AssertionError('Candidate validation must finish before any training')
    monkeypatch.setattr(SVC, 'fit', fail_fit)
    monkeypatch.setattr(LogisticRegression, 'fit', fail_fit)
    with pytest.raises(ValueError, match='linear SVC|probability=True'):
        run_inner_cv(*data, 1, base, grid, 'accuracy', True)


def test_valid_svc_replacement_and_empty_untuned_grid(data):
    fitted, _, _ = run_inner_cv(*data, 1, LogisticRegression(),
                               {'model': [SVC(kernel='linear', probability=True, random_state=42)]},
                               'roc_auc', True)
    assert fitted.predict_proba(data[0]).shape == (100, 2)
    untuned, _, _ = run_inner_cv(*data, 1, SVC(kernel='linear', probability=True), {}, 'roc_auc', False)
    untuned.fit(*data)
    assert untuned.predict_proba(data[0]).shape == (100, 2)


def test_model_names_do_not_collide_with_aggregates(data, tmp_path, monkeypatch):
    from biomarkerML_pipeline.evaluation import aggregate_biomarkers
    selections = {'union': ['a'], 'intersection': ['b'], 'n_models': ['c']}
    configs = []
    for name in selections:
        config = configuration(False)
        config['Model_name'] = name
        configs.append(config)
    def select(self, X, y, model_name, *args, **kwargs):
        return selections[model_name].copy(), [.8, 0., .8, 0., .8, 0.]
    monkeypatch.setattr(BiomarkerMLPipeline, '_biomarker_selection_across_folds', select)
    X, y = data
    pipeline = BiomarkerMLPipeline(pd.concat(configs, ignore_index=True), output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X.iloc[:80], y[:80])
    assert pipeline.model_biomarkers == selections
    assert pipeline.gene_list == selections
    assert pipeline.aggregate_biomarkers['union'] == ['a', 'b', 'c']
    assert pipeline.aggregate_biomarkers['intersection'] == []
    table = pd.read_csv(tmp_path / 'Model_biomarker_list.csv')
    assert table.loc[table.group_type == 'model', 'model'].tolist() == list(selections)
    result = pipeline.prediction_with_stable_biomarkers_across_models(
        X.iloc[:80], y[:80], X.iloc[80:], y[80:], appearance_threshold=.5)
    assert len(result) == 9
    assert result.loc[result.selected_gene_model == 'union', 'n_features'].eq(3).all()
    assert result.loc[result.selected_gene_model == 'consensus_features_for_appearance_threshold', 'status'].eq('no_features').all()
    assert pipeline.model_biomarkers == selections
    assert aggregate_biomarkers({'one': ['a', 'a'], 'two': []})['consensus_features_for_appearance_threshold'] == ['a']


def test_empty_rerun_removes_previous_heatmap(data, tmp_path):
    X, y = data
    pipeline = BiomarkerMLPipeline(configuration(False), output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X, y)
    heatmap = tmp_path / 'feature_selection_consensus_heatmap.png'
    assert heatmap.exists()
    pipeline.outer_model_params = configuration(False)
    pipeline.outer_model_params['Base_model'] = "LogisticRegression(penalty='l1', solver='liblinear', C=1e-10)"
    pipeline.stable_biomarker_selection(X, y)
    assert not heatmap.exists()
    assert pipeline.model_biomarkers == {'EN': []}
    assert all(not genes for genes in pipeline.aggregate_biomarkers.values())


@pytest.mark.parametrize('tune', [False, True])
def test_every_input_feature_is_scored_in_every_outer_fold(data, tmp_path, tune):
    X, y = data
    config = configuration(tune)
    # An untuned sparse model measures zeros too; no input columns are removed.
    if not tune:
        config['Base_model'] = "LogisticRegression(penalty='l1', solver='liblinear', C=1e-10)"
        config['Parameters'] = None
        config['Parameter_values'] = None
    pipeline = BiomarkerMLPipeline(config, output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X, y)
    scores = pd.read_csv(tmp_path / 'Feature_importance_by_fold_EN.csv')
    assert len(scores) == 5 * X.shape[1]
    for _, fold in scores.groupby('fold'):
        assert set(fold.gene) == set(X.columns)
    summary = pd.read_csv(tmp_path / 'Feature_stability_and_importance_summary_EN.csv')
    assert set(summary.gene) == set(X.columns)
    assert summary.measured_folds.eq(5).all()
    if not tune:
        assert scores.abs_coef.eq(0).all()
        assert pipeline.model_biomarkers == {'EN': []}


@pytest.mark.parametrize('failure', ['validation', 'training', 'output'])
def test_failed_rerun_invalidates_selection_and_can_recover(data, tmp_path, monkeypatch, failure):
    X, y = data
    pipeline = BiomarkerMLPipeline(configuration(False), output_dir=tmp_path)
    pipeline.stable_biomarker_selection(X.iloc[:80], y[:80])
    assert hasattr(pipeline, 'model_biomarkers')
    changed = X.rename(columns={column: 'new_' + column for column in X.columns})
    changed_y = np.where(y, 'case', 'control')
    with monkeypatch.context() as patch:
        if failure == 'training':
            patch.setattr(pipeline, 'parallel_outer_cv_n_jobs', 0)
        elif failure == 'output':
            original = pd.DataFrame.to_csv
            def fail_summary(self, path, *args, **kwargs):
                if Path(path).name == 'CV_model_performance.csv':
                    raise OSError('Simulated output failure')
                return original(self, path, *args, **kwargs)
            patch.setattr(pd.DataFrame, 'to_csv', fail_summary)
        with pytest.raises((ValueError, OSError)):
            pipeline.stable_biomarker_selection(
                changed.iloc[:80], changed_y[:80],
                cross_folds_selection=2 if failure == 'validation' else .6)
    for attribute in ('model_biomarkers', 'aggregate_biomarkers', 'gene_list',
                      'label_encoder_', 'classes_', 'feature_names_in_'):
        assert not hasattr(pipeline, attribute)
    with pytest.raises(ValueError, match='successfully before holdout'):
        pipeline.prediction_with_stable_biomarkers_across_models(
            changed.iloc[:80], changed_y[:80], changed.iloc[80:], changed_y[80:])
    pipeline.stable_biomarker_selection(changed.iloc[:80], changed_y[:80])
    assert pipeline.feature_names_in_ == changed.columns.tolist()
    assert pipeline.classes_.tolist() == ['case', 'control']
    result = pipeline.prediction_with_stable_biomarkers_across_models(
        changed.iloc[:80], changed_y[:80], changed.iloc[80:], changed_y[80:])
    assert result.status.eq('ok').all()
