"""Synthetic checks that the official eval_search_prw receives correctly converted inputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import prw_evaluation as evaluation
from official_evaluation import build_official_inputs, camera_id

PEOPLE = 12          # identities per frame; the official function needs >= 10 detections
FRAMES = ["c1s1_000001", "c2s1_000002", "c3s1_000003", "c4s1_000004"]
SCALE = .5           # resized-to-original factor applied by _image_tensor


def person_boxes():
    """Non-overlapping 10x20 boxes in original-image coordinates."""
    return torch.tensor([[30. * i, 0., 30. * i + 10., 20.] for i in range(PEOPLE)])


class FakeDetector:
    """Returns every annotated person as a detection (resized coordinates)."""
    det_score_thresh = .5

    def eval(self):
        return self

    def __call__(self, batch):
        return [dict(boxes=person_boxes() * SCALE, scores=torch.full((PEOPLE,), .9)) for _ in batch]

    def embed_crop(self, crops):
        return crops


class OfficialEvaluationTests(unittest.TestCase):
    def run_evaluation(self, gallery_mode):
        generator = torch.Generator().manual_seed(0)
        embeddings = {frame: torch.nn.functional.normalize(torch.randn(PEOPLE, 16, generator=generator), dim=1)
                      for frame in FRAMES}
        identities = torch.arange(PEOPLE)
        queries = [dict(identity=i, frame=FRAMES[0], box=person_boxes()[i].tolist()) for i in range(PEOPLE)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "query_box").mkdir()
            for query in queries:
                (root / "query_box" / f"{query['identity']}_{query['frame']}.jpg").touch()
            with patch.object(evaluation, "_load_mat_table", return_value=torch.zeros(1)), \
                 patch.object(evaluation, "_mat_strings", return_value=FRAMES), \
                 patch.object(evaluation, "_image_tensor", return_value=(torch.zeros(3, 8, 8), SCALE)), \
                 patch.object(evaluation, "_load_gallery_target",
                              side_effect=lambda root, name: (person_boxes(), identities.clone())), \
                 patch.object(evaluation, "_load_query_info", return_value=queries), \
                 patch.object(evaluation, "_load_query_crop",
                              side_effect=lambda path, size: embeddings[FRAMES[0]][int(path.name.split("_")[0])]), \
                 patch.object(evaluation, "embed_detected_crops",
                              side_effect=lambda model, path, boxes, device, batch: embeddings[path.stem][:len(boxes)]):
                return evaluation.evaluate_prw(FakeDetector(), root, "cpu", gallery_mode=gallery_mode,
                                               official_eval=True)

    def test_official_metrics_match_in_house_ranking(self):
        # One occurrence per frame and non-overlapping boxes: both protocols coincide.
        for gallery_mode in ("detected", "gt"):
            metrics = self.run_evaluation(gallery_mode)
            self.assertEqual(metrics["official_queries"], PEOPLE)
            self.assertAlmostEqual(metrics["official_mAP"], metrics["retrieval_mAP"], places=6)
            self.assertAlmostEqual(metrics["official_top1"], metrics["retrieval_top1"], places=6)
            self.assertLess(metrics["official_mAP"], 1.)  # random embeddings: non-trivial ranking

    def test_inputs_use_original_coordinates_and_unit_features(self):
        gallery = [dict(frame="c2s1_000002", scale=SCALE,
                        target=(person_boxes() * SCALE, torch.arange(PEOPLE)),
                        result=dict(boxes=person_boxes() * SCALE, scores=torch.full((PEOPLE,), .9),
                                    embeddings=3 * torch.eye(PEOPLE)))]
        query_items = [(dict(identity=0, frame="c1s1_000001", box=[0., 0., 10., 20.]), None, None)]
        inputs = build_official_inputs(gallery, query_items, [2 * torch.eye(PEOPLE)[0]])
        annotation = inputs.gallery_dataset.annotations[0]
        self.assertTrue(torch.allclose(torch.as_tensor(annotation["boxes"]), person_boxes()))
        self.assertTrue(torch.allclose(torch.as_tensor(inputs.gallery_dets[0][:, :4]), person_boxes()))
        self.assertEqual(annotation["cam_id"], 2)
        self.assertAlmostEqual(float((inputs.gallery_feats[0] ** 2).sum(1).max()), 1., places=6)
        self.assertAlmostEqual(float((inputs.query_box_feats[0] ** 2).sum()), 1., places=6)
        self.assertEqual(inputs.query_dataset.annotations[0]["pids"], 0)

    def test_camera_id(self):
        self.assertEqual(camera_id("c6s2_123468"), 6)
        self.assertEqual(camera_id("frame"), -1)


if __name__ == "__main__":
    unittest.main()
