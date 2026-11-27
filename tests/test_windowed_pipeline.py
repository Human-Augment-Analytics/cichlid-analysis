import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from evaluation import evaluate_models as model_report
from evaluation.evaluate import evaluate
from evaluation.families import family_evaluations, family_summary_table
from evaluation.metrics import evaluate_video
from evaluation.windows import window_metrics
from feature_engineering.pair_motion.feature_groups import ALL_MODEL_FEATURES
from feature_engineering.pair_motion.features_motion import _orbit_toy
from models.common.pair_motion import (
    FRAME_COLUMNS,
    _canonical_frame_matrix,
    build_pair_motion_table,
    combine_motion_embedding,
)
from models.common.segments import sample_segments
from models.intervals.mstcn_dlc2action.dlc_yolo_converter.dlc2action_bridge import PAIR_CLIP
from models.intervals.sartab.train import frame_scores


class PairMotionTests(unittest.TestCase):
    def test_role_swap_canonicalizes_directional_features(self):
        row = {name: 1.0 for name in ALL_MODEL_FEATURES}
        row.update({
            "speed_bl_s_1": 2.0,
            "speed_bl_s_2": 5.0,
            "speed_diff_bl_s": -3.0,
            "bearing_1_sin": 0.25,
            "bearing_2_sin": 0.75,
            "rel_heading_sin": 0.5,
            "male_first": False,
            "FrameNum": 0,
        })
        matrix = _canonical_frame_matrix(pd.DataFrame([row]), 1)
        table = pl.DataFrame({name: matrix[:, i] for i, name in enumerate(FRAME_COLUMNS)})
        self.assertEqual(table["pm_speed_male_bl_s"][0], 5.0)
        self.assertEqual(table["pm_speed_partner_bl_s"][0], 2.0)
        self.assertEqual(table["pm_speed_male_minus_partner_bl_s"][0], 3.0)
        self.assertEqual(table["pm_bearing_male_sin"][0], 0.75)
        self.assertEqual(table["pm_rel_heading_sin"][0], -0.5)
        self.assertEqual(table["pm_pair_present"][0], 1.0)

    def test_combined_embedding_replaces_only_pair_block(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            with (data / "v_pairmotion.pickle").open("wb") as handle:
                pickle.dump({PAIR_CLIP: np.array([[1, 2], [3, 4]], np.float32)}, handle)
            with (data / "v_pairemb.pickle").open("wb") as handle:
                pickle.dump({PAIR_CLIP: np.array([[9, 9, 10, 1], [9, 9, 20, 1]], np.float32)}, handle)
            (data / "_pairmotion_info.json").write_text(json.dumps({"feature_names": ["motion_a", "motion_b"]}))
            (data / "_pairemb_info.json").write_text(json.dumps({"feature_names": ["old_a", "old_b", "pc1", "present"]}))
            meta = {"pair_stats": {"pair_features": ["old_a", "old_b"]}}
            target = combine_motion_embedding(Path(directory), "v", meta, "pairmotion", "pairemb")
            with target.open("rb") as handle:
                values = pickle.load(handle)[PAIR_CLIP]
            np.testing.assert_array_equal(values, [[1, 2, 10, 1], [3, 4, 20, 1]])
            info = json.loads((data / "_pairmotionemb_info.json").read_text())
            self.assertEqual(info["feature_names"], ["motion_a", "motion_b", "pc1", "present"])

    def test_toy_parquets_build_aligned_male_partner_table(self):
        pairs, tracks = _orbit_toy(frames=6)
        tracks["day_index"] = 1
        tracks["Time_s"] = tracks["FrameNum"] / 30.0
        tracks["TrackID"] = tracks["TrackUID"]
        tracks["Sex"] = np.where(tracks["TrackUID"] == 1, "male", "female")
        tracks["qc_n_missing_kp"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"fps": "30.0", "y_axis": "down", "coordinate_space": "raw_pixels", "units": "pixels"}
            # Write with pyarrow so the custom contract metadata is retained.
            for name, frame, extra in (
                ("toy.parquet", tracks, {}),
                ("toy_pairs.parquet", pairs, {"pair_ordering": "TrackUID_1 < TrackUID_2",
                                               "gap_tolerance_s": "0.2", "table": "pairs"}),
            ):
                table = pa.Table.from_pandas(frame, preserve_index=False)
                metadata = {**(table.schema.metadata or {}),
                            **{key.encode(): value.encode() for key, value in {**common, **extra}.items()}}
                import pyarrow.parquet as pq
                pq.write_table(table.replace_schema_metadata(metadata), root / name)
            slots = root / "toy_slots.npz"
            np.savez(slots, uid=np.tile(np.array([[1, 2]]), (6, 1)))
            meta = {"slots_file": str(slots), "stride": 1, "frame_min": 0}
            result = build_pair_motion_table("toy", meta, 6, root)
            self.assertEqual(result.height, 6)
            self.assertEqual(result.columns, ["FrameNum", *FRAME_COLUMNS])
            self.assertTrue((result["pm_pair_present"] == 1).all())
            self.assertTrue(np.isfinite(result["pm_w4s_close_frac"].to_numpy()).all())
            self.assertTrue(np.isfinite(result["pm_w16s_turn_support_frac"].to_numpy()).all())
            self.assertTrue(np.isfinite(result["pm_w8s_corotation_frac"].to_numpy()).all())

    def test_long_persistence_features_are_model_inputs(self):
        import feature_engineering.pair_motion.angular_temporal as angular

        # Column registration and generation must agree exactly.
        columns = angular.window_columns()
        self.assertIn("w8s_turn_support_frac", columns)
        self.assertIn("w16s_corotation_frac", columns)
        self.assertNotIn("w0p5s_turn_support_frac", columns)
        self.assertIn("pm_w8s_turn_support_frac", FRAME_COLUMNS)
        self.assertIn("pm_w16s_close_frac", FRAME_COLUMNS)
        self.assertEqual(len(FRAME_COLUMNS), len(set(FRAME_COLUMNS)))

    def test_oof_postprocess_selection_and_explicit_override(self):
        from models.common.postprocess_tuning import GRID, load_settings, select_postprocess

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            counts = {"a": 300, "b": 300}
            truth = {"a": [(60, 239)], "b": [(30, 209)]}
            scores = {}
            for video, spans in truth.items():
                start, end = spans[0]
                frames = np.arange(300)
                scores[video] = pl.DataFrame({
                    "FrameNum": frames,
                    "circling_score": np.where((frames >= start) & (frames <= end), 0.95, 0.05),
                })
            selected = select_postprocess(scores, truth, counts, output, folds=[["a"], ["b"]])
            candidates = pl.read_csv(output / "postprocess_candidates.csv")
            self.assertEqual(candidates.height, np.prod([len(values) for values in GRID.values()]))
            self.assertGreaterEqual(candidates["selection_score"].max(), 0.95)
            self.assertIn(selected["threshold"], GRID["threshold"])
            self.assertEqual(load_settings(output), selected)
            overridden = load_settings(output, {"threshold": 0.65, "smooth_seconds": None})
            self.assertEqual(overridden["threshold"], 0.65)
            self.assertEqual(overridden["smooth_seconds"], selected["smooth_seconds"])

    def test_sartab_scores_interpolate_between_window_centers(self):
        result = frame_scores(np.array([0.0, 1.0]), np.array([0, 2]), np.array([2, 4]), 4)
        np.testing.assert_allclose(result["circling_score"].to_numpy(), [0.0, 0.25, 0.75, 1.0])

    def test_mstcn_signature_covers_every_video_feature_file(self):
        from models.intervals.mstcn_dlc2action.train import run_signature

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for video in ("a", "b"):
                (data / f"{video}_pairmotion.pickle").write_bytes(video.encode())
            first = run_signature(root, "episode", "pairmotion", 100, ["a", "b"])
            self.assertEqual(set(first["feature_files"]), {"a", "b"})
            (data / "b_pairmotion.pickle").write_bytes(b"changed")
            second = run_signature(root, "episode", "pairmotion", 100, ["a", "b"])
            self.assertNotEqual(first, second)


class ExpandedEvaluationTests(unittest.TestCase):
    def test_negative_only_video_false_positives_are_counted(self):
        result = evaluate_video([], [(0, 9)], 100)
        self.assertEqual(result["frame_fp"], 10)
        self.assertEqual(result["mof"], 0.9)
        self.assertEqual(result["gt_segments"], 0)

    def test_window_metrics_include_negative_only_video(self):
        empty = pl.DataFrame(schema={"video_id": pl.String, "start": pl.Int64, "end": pl.Int64})
        predicted = pl.DataFrame({"video_id": ["negative"], "start": [0], "end": [99]})
        result = window_metrics(empty, predicted, {"negative": 100})
        self.assertEqual(result.filter(pl.col("video_id") == "overall")["fp"][0], 1)

    def test_full_evaluation_retains_explicit_negative_only_video(self):
        empty = pl.DataFrame(schema={"video_id": pl.String, "start": pl.Int64, "end": pl.Int64})
        predicted = pl.DataFrame({"video_id": ["negative"], "start": [0], "end": [99]})
        result, warnings = evaluate(empty, predicted, num_frames={"negative": 100})
        overall = result.filter(pl.col("video_id") == "overall").row(0, named=True)
        self.assertEqual(overall["frame_fp"], 100)
        self.assertEqual(overall["mof"], 0.0)
        self.assertEqual(warnings, [])

    def test_negative_only_video_contributes_training_negatives(self):
        table = sample_segments({"positive": [(0, 2)], "negative": []},
                                {"positive": 8, "negative": 8}, len_segment=4)
        negative = table.filter(pl.col("video") == "negative")
        self.assertEqual(negative.height, 1)
        self.assertFalse(negative["circling"].any())


class FamilyVisualizationTests(unittest.TestCase):
    def setUp(self):
        import model_performance_report_viz as viz

        self.viz = viz
        self.gt = pl.DataFrame({"video_id": ["v"], "start": [0], "end": [9]})
        first = pl.DataFrame({"video_id": ["v"], "start": [0], "end": [4]})
        second = pl.DataFrame({"video_id": ["v"], "start": [0], "end": [9]})
        self.predictions = {
            "xgboost_pairmotion": first,
            "xgboost_pairmotionvmae": second,
            "mstcn_pairmotion": second,
        }
        self.evaluations = {
            name: evaluate(self.gt, predicted, num_frames={"v": 20})
            for name, predicted in self.predictions.items()
        }

    def test_family_evaluation_means_and_variant_counts(self):
        families = self.viz.family_evaluations(self.evaluations)
        self.assertEqual(set(families), {"XGBoost", "Deep learning"})
        overall = families["XGBoost"][0].filter(pl.col("video_id") == "overall").row(0, named=True)
        self.assertEqual(overall["variant_count"], 2)
        self.assertAlmostEqual(overall["frame_f1"], (2 / 3 + 1) / 2)
        table = self.viz.slide_table(families)
        self.assertEqual(table.columns[:3], ["Model", "Variants", "Pred segments"])
        self.assertEqual(table.filter(pl.col("Model") == "XGBoost")["Variants"][0], 2)

    def test_family_metric_and_timeline_aggregation_respect_coverage(self):
        window = pl.concat([
            window_metrics(self.gt, predicted, {"v": 20}).with_columns(pl.lit(name).alias("model"))
            for name, predicted in list(self.predictions.items())[:2]
        ])
        family_window = self.viz.family_metric_table(window)
        overall = family_window.filter((pl.col("model") == "XGBoost") &
                                       (pl.col("video_id") == "overall")).row(0, named=True)
        self.assertEqual(overall["variant_count"], 2)
        agreement = self.viz.family_frame_agreement(
            self.predictions, "v", 20,
            {"xgboost_pairmotion": ["v"], "xgboost_pairmotionvmae": ["v"], "mstcn_pairmotion": ["other"]})
        self.assertEqual(set(agreement), {"XGBoost"})
        np.testing.assert_allclose(agreement["XGBoost"][:5], 1.0)
        np.testing.assert_allclose(agreement["XGBoost"][5:10], 0.5)
        np.testing.assert_allclose(agreement["XGBoost"][10:], 0.0)

    def test_family_figures_render(self):
        from unittest.mock import patch

        families = self.viz.family_evaluations(self.evaluations)
        table = self.viz.slide_table(families)
        self.viz.plot_slide_table(table)
        self.viz.plot_segment_f1(families)
        self.viz.plot_precision_recall(families)
        self.viz.plot_segment_counts(families)
        self.viz.plot_per_video(families)
        heldout = pl.DataFrame({
            "model": ["xgboost_pairmotion", "xgboost_pairmotionvmae"],
            "test_unit": ["frames", "frames"], "average_precision": [0.4, 0.8], "f1": [0.5, 0.7],
        })
        family_heldout = self.viz.family_metric_table(heldout)
        self.assertAlmostEqual(family_heldout["average_precision"][0], 0.6)
        self.viz.plot_heldout(family_heldout)
        windows = pl.concat([
            window_metrics(self.gt, predicted, {"v": 20}).with_columns(pl.lit(name).alias("model"))
            for name, predicted in list(self.predictions.items())[:2]
        ])
        self.viz.plot_window_metrics(self.viz.family_metric_table(windows))
        coverage = {name: ["v"] for name in self.predictions}
        with patch.object(self.viz, "video_frames", return_value={"v": 20}):
            self.viz.plot_timeline(self.gt, self.predictions, "v", coverage=coverage)

    def test_requested_family_filter_and_actionformer_versions(self):
        expanded = dict(self.evaluations)
        expanded["actionformer_pairmotion"] = evaluate(
            self.gt, self.predictions["mstcn_pairmotion"], num_frames={"v": 20})
        expanded["sartab"] = evaluate(self.gt, self.predictions["mstcn_pairmotion"], num_frames={"v": 20})
        expanded["logistic_regression"] = evaluate(
            self.gt, self.predictions["mstcn_pairmotion"], num_frames={"v": 20})
        without = self.viz.family_evaluations(expanded, include_actionformer=False)
        with_actionformer = self.viz.family_evaluations(expanded, include_actionformer=True)
        self.assertEqual(set(without), {"XGBoost", "Deep learning"})
        self.assertEqual(set(with_actionformer), {"XGBoost", "Deep learning", "ActionFormer"})
        self.assertIsNone(self.viz.model_family("lrcv_pairmotion"))
        self.assertEqual(set(self.viz.selected_predictions(
            {"actionformer_pairmotion": self.predictions["mstcn_pairmotion"],
             "sartab": self.predictions["mstcn_pairmotion"]}, False)), set())

    def test_family_mAP_summary_averages_variants(self):
        main = pl.DataFrame({
            "model": ["xgboost_pairmotion", "xgboost_pairmotionvmae", "mstcn_pairmotion"],
            "mAP@0.1": [20.0, 40.0, 80.0], "mAP Avg": [10.0, 30.0, 60.0],
            "σ_video": [2.0, 4.0, 8.0],
        })
        result = self.viz.family_tal_table(main)
        xgboost = result.filter(pl.col("Method") == "XGBoost").row(0, named=True)
        self.assertEqual(xgboost["Variants"], 2)
        self.assertEqual(xgboost["mAP@0.1"], 30.0)
        self.assertEqual(xgboost["mAP Avg"], 20.0)

    def test_shared_family_aggregation_has_stable_order_and_counts(self):
        expanded = {
            "actionformer_pairmotion": evaluate(
                self.gt, self.predictions["mstcn_pairmotion"], num_frames={"v": 20}),
            **self.evaluations,
            "rule_based": evaluate(self.gt, self.predictions["mstcn_pairmotion"], num_frames={"v": 20}),
        }
        families = family_evaluations(expanded)
        self.assertEqual(tuple(families), ("Rule-based", "Deep learning", "XGBoost", "ActionFormer"))
        summary = family_summary_table(families)
        self.assertEqual(summary["Model"].to_list(), list(families))
        self.assertEqual(summary.filter(pl.col("Model") == "XGBoost")["Variants"][0], 2)
        self.assertEqual(summary.filter(pl.col("Model") == "Rule-based")["Variants"][0], 1)

    def test_family_metric_profile_scaling_mAP_and_new_figures(self):
        families = family_evaluations(self.evaluations)
        main = pl.DataFrame({
            "model": ["xgboost_pairmotion", "xgboost_pairmotionvmae", "mstcn_pairmotion"],
            "mAP@0.1": [20.0, 40.0, 80.0], "mAP Avg": [10.0, 30.0, 60.0],
            "σ_video": [2.0, 4.0, 8.0],
        })
        profile = self.viz.family_metric_profile(families, self.viz.family_tal_table(main))
        xgboost = profile.filter(pl.col("Model") == "XGBoost").row(0, named=True)
        self.assertEqual(xgboost["Variants"], 2)
        self.assertEqual(xgboost["Videos"], 1)
        self.assertAlmostEqual(xgboost["Frame F1"], 100 * (2 / 3 + 1) / 2)
        self.assertEqual(xgboost["mAP Avg"], 20.0)
        self.assertEqual(profile.filter(pl.col("Model") == "Deep learning")["mAP Avg"][0], 60.0)
        self.viz.plot_family_metric_profile(profile)
        self.viz.plot_family_video_distributions(families)

    def test_family_first_csvs_keep_raw_models_in_details(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            figures = Path(directory) / "figures"
            figures.mkdir()
            stale = figures / "model_metrics_overall.csv"
            stale.write_text("stale")
            with (patch.object(model_report, "heldout_metrics", return_value=pl.DataFrame()),
                  patch.object(model_report, "window_report", return_value=pl.DataFrame())):
                tables = model_report.write_metric_tables(
                    self.gt, self.evaluations, self.predictions, figures=figures)
            self.assertEqual(set(tables["overall"]["Model"]), {"XGBoost", "Deep learning"})
            for name in ("overall", "per_video", "heldout", "60s_windows"):
                self.assertTrue((figures / f"family_metrics_{name}.csv").is_file())
            for name in ("overall", "per_video", "heldout", "60s_windows"):
                self.assertTrue((figures / "details" / f"model_metrics_{name}.csv").is_file())
            raw = pl.read_csv(figures / "details" / "model_metrics_overall.csv")
            self.assertEqual(set(raw["Model"]), set(self.evaluations))
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
