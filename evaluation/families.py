"""Shared model-family aggregation for console, CSV, and visual reports."""
import re

import polars as pl

FAMILY_ORDER = ('Rule-based', 'Deep learning', 'XGBoost', 'ActionFormer')


def base_name(name):
    return re.sub(r'(_15spans|_hn\d+)+$', '', name)


def model_family(name):
    """Map requested families; other models remain available only as details."""
    base = base_name(name)
    if base == 'rule_based':
        return 'Rule-based'
    if base.startswith('mstcn_'):
        return 'Deep learning'
    if base == 'xgboost' or base.startswith('xgboost_'):
        return 'XGBoost'
    if base.startswith('actionformer'):
        return 'ActionFormer'
    return None


def include_family(family, include_actionformer=True):
    return family in FAMILY_ORDER and (include_actionformer or family != 'ActionFormer')


def selected_predictions(predictions, include_actionformer=True):
    return {name: table for name, table in predictions.items()
            if include_family(model_family(name), include_actionformer)}


def family_evaluations(evaluations, include_actionformer=True):
    """Equal-weight means of available variants, separately for every video.

    Predictions and labels are never pooled into an artificial ensemble. The
    per-video variant count records which models actually contributed.
    """
    grouped = {}
    for name, (results, warnings) in evaluations.items():
        family = model_family(name)
        if include_family(family, include_actionformer):
            grouped.setdefault(family, []).append((name, results, warnings))
    output = {}
    for family, items in grouped.items():
        combined = pl.concat([
            results.with_columns(pl.lit(name).alias('variant'))
            for name, results, _ in items
        ], how='diagonal_relaxed')
        values = [column for column, dtype in combined.schema.items()
                  if column not in ('video_id', 'variant') and dtype.is_numeric()]
        averaged = combined.group_by('video_id').agg([
            *[pl.col(column).mean().alias(column) for column in values],
            pl.col('variant').n_unique().alias('variant_count'),
        ])
        rows = averaged.filter(pl.col('video_id') != 'overall').sort('video_id')
        overall = averaged.filter(pl.col('video_id') == 'overall')
        results = pl.concat([rows, overall], how='diagonal_relaxed')
        warnings = [warning for _, _, messages in items for warning in messages]
        output[family] = (results, warnings)
    return {family: output[family] for family in FAMILY_ORDER if family in output}


def family_metric_table(table, include_actionformer=True):
    """Family means for held-out or window metric tables."""
    if table.is_empty():
        return table
    table = table.filter(pl.col('model').map_elements(
        lambda name: include_family(model_family(name), include_actionformer), return_dtype=pl.Boolean))
    if table.is_empty():
        return table
    keyed = table.with_columns(pl.col('model').map_elements(
        model_family, return_dtype=pl.String).alias('family'))
    groups = ['family'] + (['video_id'] if 'video_id' in table.columns else [])
    values = [column for column, dtype in table.schema.items()
              if column not in ('model', *groups) and dtype.is_numeric()]
    aggregations = [pl.col(column).mean().alias(column) for column in values]
    if 'test_unit' in table.columns:
        aggregations.append(pl.col('test_unit').first().alias('test_unit'))
    aggregations.append(pl.col('model').n_unique().alias('variant_count'))
    grouped = keyed.group_by(groups).agg(aggregations).rename({'family': 'model'})
    rank = {name: index for index, name in enumerate(FAMILY_ORDER)}
    return grouped.sort(pl.col('model').replace_strict(rank, default=len(rank), return_dtype=pl.Int64),
                        *(['video_id'] if 'video_id' in groups else []))


def family_tal_table(main, include_actionformer=True):
    """Family means for the temporal-localization mAP summary."""
    if main.is_empty():
        return main
    main = main.filter(pl.col('model').map_elements(
        lambda name: include_family(model_family(name), include_actionformer), return_dtype=pl.Boolean))
    if main.is_empty():
        return main
    keyed = main.with_columns(pl.col('model').map_elements(
        model_family, return_dtype=pl.String).alias('Method'))
    metrics = [column for column in main.columns if column.startswith('mAP@') or column == 'mAP Avg']
    return (keyed.group_by('Method').agg([
        pl.col('model').n_unique().alias('Variants'),
        *[pl.col(column).mean().alias(column) for column in metrics],
        pl.col('σ_video').mean().alias('σ_video'),
    ]).sort('mAP Avg', descending=True))


def family_summary_table(evaluations):
    """Ordered overall family metrics with contributing variant counts."""
    from evaluation.evaluate_models import summary_table
    table = summary_table(evaluations)
    if table.is_empty():
        return table
    variants = pl.DataFrame({
        'Model': list(evaluations),
        'Variants': [int(results.filter(pl.col('video_id') == 'overall')['variant_count'][0])
                     for results, _ in evaluations.values()],
    })
    table = table.join(variants, on='Model', how='left')
    return table.select('Model', 'Variants', pl.exclude('Model', 'Variants'))
