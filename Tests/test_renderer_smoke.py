import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT_DIR / "Fixtures" / "sample_stroke_program.json"
RUNNER_PATH = ROOT_DIR / "Scripts" / "run_renderer.sh"


class RendererSmokeTest(unittest.TestCase):
    def test_fixture_renders_expected_outputs(self):
        with tempfile.TemporaryDirectory() as output_dir_name:
            output_dir = Path(output_dir_name)
            subprocess.run(
                [str(RUNNER_PATH), str(FIXTURE_PATH), str(output_dir)],
                cwd=ROOT_DIR,
                check=True,
            )

            final_image = output_dir / "final.png"
            replay_gif = output_dir / "replay.gif"
            manifest_path = output_dir / "render_manifest.json"
            frames = sorted((output_dir / "frames").glob("frame_*.png"))

            self.assertTrue(final_image.exists())
            self.assertGreater(final_image.stat().st_size, 0)
            self.assertTrue(replay_gif.exists())
            self.assertGreater(replay_gif.stat().st_size, 0)
            self.assertEqual(len(frames), 3)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["canvas"], {"width": 512, "height": 512})
            self.assertEqual(manifest["stroke_count"], 3)
            self.assertEqual(manifest["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
