"""Unit tests for safe instance-segmentation release freezing."""
from __future__ import annotations
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

PATH = Path(__file__).resolve().parents[1] / 'tools' / 'freeze_instance_segmentation_v1.py'
spec = importlib.util.spec_from_file_location('freeze_instance_tool', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FreezeReleaseTest(unittest.TestCase):
    def make_repo(self, where: Path):
        root = where / '3D-TissueMap-AI'
        (root / 'scripts').mkdir(parents=True)
        (root / 'tests').mkdir()
        (root / 'runs' / 'learned_region_graph_val_diagnostic').mkdir(parents=True)
        (root / 'runs' / 'voxel_affinity_extended').mkdir()
        (root / 'runs' / 'spatial_morphology_future').mkdir()
        (root / 'scripts' / 'evaluate_learned_region_graph.py').write_text('print("ok")\n')
        (root / 'scripts' / 'train_voxel_affinity.py').write_text('print("training")\n')
        (root / 'tests' / 'test_region.py').write_text('assert True\n')
        (root / 'runs' / 'learned_region_graph_val_diagnostic' / 'metrics.csv').write_text('pq\n0.08377\n')
        (root / 'runs' / 'voxel_affinity_extended' / 'xyz_downweighted.pt').write_bytes(b'FAKE-CHECKPOINT')
        (root / 'runs' / 'spatial_morphology_future' / 'later.json').write_text('{}')
        (where / '3D-TissueMap-Something-Update.zip').write_bytes(b'ZIP TEST')
        (where / '3Dpathology.wps').write_bytes(b'KEEP')
        return root

    def test_dry_run_non_destructive(self):
        with TemporaryDirectory() as d:
            root = self.make_repo(Path(d))
            with redirect_stdout(io.StringIO()):
                module.freeze(root, apply=False, max_file_mb=10,
                              include_figures=True, include_models=True)
            self.assertFalse((root / 'releases').exists())
            self.assertTrue((root / 'scripts' / 'train_voxel_affinity.py').exists())

    def test_freeze_verify_and_preserve_paths(self):
        with TemporaryDirectory() as d:
            root = self.make_repo(Path(d))
            with redirect_stdout(io.StringIO()):
                module.freeze(root, apply=True, max_file_mb=10,
                              include_figures=True, include_models=True)
                module.verify_release(root)
            release = root / 'releases' / module.VERSION
            self.assertEqual((release / 'snapshot/scripts/train_voxel_affinity.py').read_text(), 'print("training")\n')
            self.assertEqual((release / 'snapshot/runs/voxel_affinity_extended/xyz_downweighted.pt').read_bytes(), b'FAKE-CHECKPOINT')
            self.assertFalse((release / 'snapshot/runs/spatial_morphology_future/later.json').exists())
            self.assertTrue((root / 'runs' / 'learned_region_graph_val_diagnostic/metrics.csv').exists())
            self.assertIn('04_region_graph', (release / 'script_workflow_index.csv').read_text())
            self.assertIn('不是已验证的最优', (release / 'README_冻结说明.md').read_text())
            with self.assertRaises(SystemExit):
                module.freeze(root, apply=True, max_file_mb=10, include_figures=True, include_models=True)
            (release / 'snapshot/scripts/train_voxel_affinity.py').write_text('tampered')
            with self.assertRaises(RuntimeError):
                module.verify_release(root)

    def test_optional_archive_requires_valid_release_and_preserves_other_file(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            root = self.make_repo(base)
            with self.assertRaises(SystemExit):
                module.archive_parent_zips(root, apply=True)
            with redirect_stdout(io.StringIO()):
                module.freeze(root, apply=True, max_file_mb=10, include_figures=False, include_models=False)
                module.archive_parent_zips(root, apply=False)
            self.assertTrue((base / '3D-TissueMap-Something-Update.zip').exists())
            with redirect_stdout(io.StringIO()):
                module.archive_parent_zips(root, apply=True)
            self.assertFalse((base / '3D-TissueMap-Something-Update.zip').exists())
            self.assertEqual((base / '_archive/instance_segmentation_v1_updates/3D-TissueMap-Something-Update.zip').read_bytes(), b'ZIP TEST')
            self.assertEqual((base / '3Dpathology.wps').read_bytes(), b'KEEP')

    def test_oversized_files_reported_without_copy(self):
        with TemporaryDirectory() as d:
            root = self.make_repo(Path(d))
            (root / 'runs' / 'voxel_affinity_extended' / 'large.pt').write_bytes(b'a' * (2 * 1024 * 1024))
            with redirect_stdout(io.StringIO()):
                module.freeze(root, apply=True, max_file_mb=1, include_figures=False, include_models=True)
            release = root / 'releases' / module.VERSION
            self.assertFalse((release / 'snapshot/runs/voxel_affinity_extended/large.pt').exists())
            self.assertIn('large.pt', (release / 'excluded_files.csv').read_text())

if __name__ == '__main__':
    unittest.main()
