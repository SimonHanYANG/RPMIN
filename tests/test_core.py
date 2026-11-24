import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

from rpmin.calibration import CameraCalibration
from rpmin.data import Pair, PairedGrid, PairedImages, manifest_summary, read_manifest, save_rgb, split_manifest
from rpmin.engine import TrainConfig, evaluate, predict, train
from rpmin.model import AdaptiveConv, ModelConfig, RPMIN, RPUM
from rpmin.objective import ReconstructionLoss, grid_metrics, image_metrics
from rpmin.paired_reference import lift_transform
from rpmin.registration import overlap_rectangle
from rpmin.views import build_views, displacements


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_signed_difference_and_original_view(self):
        image = torch.arange(10, dtype=torch.float32).repeat(8, 1)[None, None]
        offsets = torch.tensor([[[0.5, 0], [-0.5, 0]]])
        views = build_views(image, offsets)
        torch.testing.assert_close(views[:, 0], image)
        inner = views[0, 1:, 0, 2:-2, 2:-2]
        expected = torch.tensor([0.5, -0.5])[:, None, None].expand_as(inner)
        torch.testing.assert_close(inner, expected)

    def test_registration_transform_lifts_pixel_centers(self):
        matrix = np.array([[1.02, 0.01, 0.8], [-0.02, 0.99, -1.2], [0.0001, 0, 1]])
        point = np.array([15., 31., 1.])
        target = matrix @ point
        for scale in (2, 4):
            to_hr = lambda xy: (xy + 0.5) * scale - 0.5
            lifted = lift_transform(matrix, scale) @ np.r_[to_hr(point[:2]), 1.]
            np.testing.assert_allclose(lifted[:2] / lifted[2], to_hr(target[:2] / target[2]))

    def test_self_ensemble_inverts_geometry_and_consumes_one_draw(self):
        image = torch.arange(24).float().reshape(1, 1, 4, 6)

        def model(sample, generator):
            return sample + torch.rand((), generator=generator)

        single, ensemble = (torch.Generator().manual_seed(11) for _ in range(2))
        torch.testing.assert_close(predict(model, image, single), predict(model, image, ensemble, True))
        self.assertTrue(torch.equal(single.get_state(), ensemble.get_state()))

    def test_detail_head_starts_as_zero_residual(self):
        model = RPMIN(ModelConfig(width=4, scale=2, detail_head=True))
        features = torch.randn(2, 4, 5, 7)
        residual = model.detail(features)
        self.assertEqual(residual.shape, (2, 3, 10, 14))
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_compass_isotropic_and_repeatable(self):
        xy = displacements(1, 0.5)[0]
        torch.testing.assert_close(xy.T @ xy, torch.eye(2))
        x, y = xy.T
        design = torch.stack((x, y, x.square() / 2, x * y, y.square() / 2), dim=1)
        self.assertEqual(torch.linalg.matrix_rank(design).item(), 5)
        a, b = (displacements(2, None, torch.Generator().manual_seed(12)) for _ in range(2))
        torch.testing.assert_close(a, b)

    def test_deformable_conv_at_zero_offsets_matches_conv(self):
        block = AdaptiveConv(2, 3)
        image = torch.randn(2, 2, 7, 9)
        actual = block(image)
        expected = F.conv2d(image, block.conv.weight, block.conv.bias, padding=1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_rpum_starts_at_half_pixel_bilinear_coordinates(self):
        for scale in (2, 4):
            block = RPUM(3, scale)
            with torch.no_grad():
                block.to_rgb.weight.zero_()
                block.to_rgb.weight[:, :, 1, 1].copy_(torch.eye(3))
                block.to_rgb.bias.zero_()
            image = torch.randn(2, 3, 6, 9)
            expected = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
            torch.testing.assert_close(block(image), expected, atol=3e-6, rtol=2e-5)

    def test_initial_residual_leaves_bicubic_unchanged(self):
        image = torch.rand(2, 3, 7, 9)
        for scale in (2, 4):
            model = RPMIN(ModelConfig(scale=scale, width=4))
            base = F.interpolate(image, scale_factor=scale, mode="bicubic", align_corners=False)
            torch.testing.assert_close(model(image), base, rtol=0, atol=0)

    def test_zero_initialized_readout_learns_and_exposes_offset_gradients(self):
        model = RPMIN(ModelConfig(scale=4, width=4, radius=0.5))
        image = torch.rand(2, 3, 8, 10)
        target = F.interpolate(image, scale_factor=4, mode="bicubic", align_corners=False) + 0.05
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(image), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        for layer in (model.stem[0], model.upsample.offsets, model.upsample.phase):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0.0)

    def test_energy_weighted_phase_is_finite_for_zero_spectrum(self):
        target = torch.rand(1, 3, 7, 9)
        prediction = torch.zeros_like(target, requires_grad=True)
        criterion = ReconstructionLoss(phase="energy-weighted")
        loss = criterion(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(criterion(target, target), torch.tensor(9e-7))

    def test_combined_distortion_loss_is_differentiable(self):
        target = torch.rand(2, 3, 16, 19)
        prediction = torch.rand_like(target, requires_grad=True)
        loss = ReconstructionLoss(phase="energy-weighted", mse_weight=0.1, ssim_weight=0.05)
        value = loss(prediction, target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        torch.testing.assert_close(loss(target, target), torch.tensor(9e-7), atol=1e-7, rtol=0)

    def test_patch_input_grid_uses_explicit_hr_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            low_path, high_path = root / "low.png", root / "high.png"
            ramp = Image.fromarray(np.repeat(np.arange(10, dtype=np.uint8)[None], 10, axis=0))
            ramp.save(low_path)
            ramp.resize((20, 20), Image.Resampling.NEAREST).save(high_path)
            pairs = [Pair("field", "val", "group", low_path, high_path)]
            scaled = PairedGrid(pairs, "val", 2, 4, 4, 4)
            original = PairedGrid(pairs, "val", 2, 4, 4, 4, hr_origin=1)
            self.assertEqual(len(scaled), 4)
            low, high, identifier = scaled[0]
            self.assertEqual(identifier, "field-x1-y1")
            self.assertEqual(high.shape[-2:], (8, 8))
            torch.testing.assert_close(low, high[:, ::2, ::2])
            torch.testing.assert_close(low, original[0][0])
            self.assertEqual(original[0][1][0, 0, 0].item(), 0.0)

    def test_model_forward_and_gradients(self):
        image = torch.rand(1, 3, 7, 9)
        for scale in (2, 4):
            model = RPMIN(ModelConfig(scale=scale, width=4))
            result, maps = model(image, return_maps=True)
            self.assertEqual(result.shape, (1, 3, 7 * scale, 9 * scale))
            self.assertEqual(set(maps), {"sce", "sda"})
            loss = ReconstructionLoss()(result, torch.rand_like(result))
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_image_metrics_at_identity(self):
        image = torch.rand(2, 3, 13, 19)
        psnr, ssim = image_metrics(image, image, border=2)
        torch.testing.assert_close(psnr, torch.full((2,), 120.0))
        torch.testing.assert_close(ssim, torch.ones(2))

    def test_grid_scores_match_explicit_crops(self):
        sr, hr = torch.rand(2, 1, 3, 30, 34)
        points, psnr, ssim = grid_metrics(sr, hr, scale=2, patch=7, stride_x=3, stride_y=4, border=1)
        self.assertEqual(points, [(x, y) for y in (1, 5) for x in (1, 4, 7)])
        scores = [
            image_metrics(sr[..., 2*y:2*y + 14, 2*x:2*x + 14], hr[..., 2*y:2*y + 14, 2*x:2*x + 14], border=1)
            for x, y in points
        ]
        torch.testing.assert_close(psnr.float(), torch.cat([p for p, _ in scores]))
        torch.testing.assert_close(ssim.float(), torch.cat([s for _, s in scores]), atol=1e-6, rtol=1e-5)

    def test_overlap_rectangle_keeps_all_valid_content_it_can(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[1:7, 2:9] = True
        mask[1, 2] = False
        x, y, w, h = overlap_rectangle(mask, min_size=3)
        self.assertEqual((x, y, w, h), (3, 1, 6, 6))
        self.assertTrue(mask[y:y + h, x:x + w].all())

    def test_camera_calibration_preserves_residual_and_shift_sign(self):
        image = torch.zeros(1, 3, 7, 9)
        image[:, :, 4, 3] = 1
        calibration = CameraCalibration((1, -2), ((0.1, 1, 0, 0, 0, 0),) * 3, 0.5)
        aligned = calibration.align(image)
        torch.testing.assert_close(aligned[:, :, 2, 4], torch.ones(1, 3))
        base = F.interpolate(aligned, scale_factor=2, mode="bicubic", align_corners=False)
        prediction = base + 0.04
        torch.testing.assert_close(calibration.correct_tone(prediction, aligned), prediction + 0.05)

    def test_repeated_cached_crops_keep_field_identity_and_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            low, high = root / "lr.png", root / "hr.png"
            Image.new("RGB", (12, 14), (9, 143, 226)).save(low)
            Image.new("RGB", (24, 28), (9, 143, 226)).save(high)
            data = PairedImages([Pair("field", "train", "slide", low, high)],
                                "train", 2, patch=8, samples_per_pair=3, cache=True)
            self.assertEqual(len(data), 3)
            for index in range(len(data)):
                lr, hr, name = data[index]
                self.assertEqual(name, "field")
                torch.testing.assert_close(lr.repeat_interleave(2, -1).repeat_interleave(2, -2), hr)
            self.assertEqual(len(data.cache), 2)

    def test_group_split_keeps_source_images_together(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "pairs.csv"
            groups = [f"slide{i // 2}" for i in range(20)]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", "group", "lr", "hr"))
                for index, group in enumerate(groups):
                    lr, hr = f"{index}-low.png", f"{index}-high.png"
                    Image.new("RGB", (4, 4)).save(folder / lr)
                    Image.new("RGB", (8, 8)).save(folder / hr)
                    writer.writerow((f"image-{index}", group, lr, hr))
            destination = folder / "assigned.csv"
            split_manifest(source, destination, 42)
            pairs = read_manifest(destination)
            summary = manifest_summary(pairs)
            self.assertEqual([summary[s]["pairs"] for s in ("train", "val", "test")], [16, 2, 2])
            for group in set(groups):
                self.assertEqual(len({pair.split for pair in pairs if pair.group == group}), 1)

    def test_epoch_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "pairs.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", "split", "lr", "hr"))
                for split in ("train", "val"):
                    save_rgb(torch.rand(3, 8, 8), folder / f"{split}_lr.png")
                    save_rgb(torch.rand(3, 16, 16), folder / f"{split}_hr.png")
                    writer.writerow((split, split, f"{split}_lr.png", f"{split}_hr.png"))
            model_config = ModelConfig(scale=2, width=4)
            training = TrainConfig(epochs=3, warmup=0, batch=1, patch=8, validate_every=3)
            device = torch.device("cpu")
            train(manifest, folder / "full", model_config, training, device)
            calls = 0

            def interrupt_after_first_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("Simulated interruption in the final epoch.")
                return evaluate(*args, **kwargs)

            with patch("rpmin.engine.evaluate", side_effect=interrupt_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    train(manifest, folder / "resumed", model_config, training, device)
            train(manifest, folder / "resumed", model_config, training, device, resume=True)
            full = torch.load(folder / "full/latest.pt", weights_only=True)
            resumed = torch.load(folder / "resumed/latest.pt", weights_only=True)
            self.assertIsNone(resumed["history"][1]["val_psnr"])
            torch.testing.assert_close(full["model_state"], resumed["model_state"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
