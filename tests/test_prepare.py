import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from rpmin.data import read_manifest
from rpmin.prepare import prepare_dataset


class PrepareTests(unittest.TestCase):
    collection = "ciliated epithelial cell for super resolution"

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name)
        self.root = self.base / "SuperResolutionDataset"
        self.images = self.root / self.collection
        for name in (self.collection, "MITOS-14 for super resolution"):
            for mag in (10, 20, 40):
                (self.root / name / f"x{mag}").mkdir(parents=True)

    def save(self, field, mag, size):
        path = self.images / f"x{mag}" / f"{field}_x{mag}.png"
        Image.new("RGB", size, "white").save(path)

    def test_group_splits_are_shared_across_available_scales(self):
        for field in ("1", "2", "3", "4"):
            for mag in (10, 20, 40):
                factor = mag // 10
                self.save(field, mag, (16 * factor, 20 * factor))
        self.save("extra", 10, (16, 20))
        self.save("bad", 10, (16, 20))
        self.save("bad", 20, (32, 39))
        groups = self.base / "groups.csv"
        with groups.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("dataset", "field_id", "group"))
            for field, slide in (("1", "a"), ("2", "a"), ("3", "b"), ("4", "c")):
                writer.writerow((self.collection, field, slide))
        out = self.base / "prepared"
        report = prepare_dataset(self.root, out, groups_path=groups)
        tasks = report["collections"]["mfa-cell"]["tasks"]
        self.assertEqual(tasks["x2"]["invalid_dimensions"][0]["field_id"], "bad")
        self.assertEqual(tasks["x4"]["missing_hr"], ["bad", "extra"])
        both = [read_manifest(out / tasks[scale]["manifest"]) for scale in ("x2", "x4")]
        self.assertEqual([{p.id: p.split for p in pairs} for pairs in both][0],
                         {p.id: p.split for p in both[1]})
        self.assertEqual(len({p.split for p in both[0] if p.group == "a"}), 1)
        self.assertEqual(report["collections"]["mitos-14"]["tasks"]["x4"]["valid_pairs"], 0)

    def test_single_field_demo_is_disjoint_and_preserves_sources(self):
        self.save("1", 10, (52, 20))
        self.save("1", 40, (208, 80))
        before = {p: p.read_bytes() for p in self.root.rglob("*.png")}
        out = self.base / "prepared"
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare_dataset(self.root, self.root / "prepared", demo_tile=12)
        report = prepare_dataset(self.root, out, demo_tile=12, demo_gap=2)
        task = report["collections"]["mfa-cell"]["tasks"]["x4"]
        self.assertNotIn("manifest", task)
        demo = task["demo"]
        self.assertEqual(demo["status"], "single_field_spatial_demo")
        self.assertEqual({p.split for p in read_manifest(out / demo["manifest"])}, {"train", "val", "test"})
        for a in demo["tiles"]:
            for b in demo["tiles"]:
                if a["split"] != b["split"]:
                    box_a, box_b = a["lr_xyxy"], b["lr_xyxy"]
                    self.assertGreaterEqual(max(box_a[0] - box_b[2], box_b[0] - box_a[2]), 2)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.png")})

    def test_mitos_names_match_across_formats_with_a_shared_source(self):
        folder = self.root / "MITOS-14 for super resolution"
        for field in ("H10_00", "H10_01", "H10_02"):
            for mag, extension in ((10, "tiff"), (20, "png"), (40, "png")):
                Image.new("RGB", (mag, mag)).save(folder / f"x{mag}" / f"{field}.{extension}")
        out = self.base / "prepared"
        report = prepare_dataset(self.root, out)
        collection = report["collections"]["mitos-14"]
        self.assertEqual(collection["source_groups_with_pairs"], 1)
        for task in collection["tasks"].values():
            pairs = read_manifest(out / task["pair_index"], require_split=False)
            self.assertEqual(len(pairs), 3)
            self.assertEqual({pair.group for pair in pairs}, {"h10"})
            self.assertNotIn("manifest", task)


if __name__ == "__main__":
    unittest.main()
