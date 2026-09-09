from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_resume import (
    epoch_batches,
    split_rows,
    save_snapshot,
    restore_snapshot,
    completed,
    mark_completed,
)


class ResumeTests(unittest.TestCase):
    def test_image_split_and_schedule(self):
        rows = [{"vg_image_id": str(i // 3), "id": str(i)} for i in range(30)]
        tr, va = split_rows(rows, 0.2, 13)
        self.assertFalse(
            {r["vg_image_id"] for r in tr} & {r["vg_image_id"] for r in va}
        )
        schedule = epoch_batches(31, 2, 13, 0)
        self.assertEqual(schedule, epoch_batches(31, 2, 13, 0))
        self.assertEqual(
            sorted(i for batch in schedule for i in batch), list(range(31))
        )
        self.assertNotEqual(schedule, epoch_batches(31, 2, 13, 1))

    def test_interrupted_matches_uninterrupted_with_dropout_and_accumulation(self):
        import torch
        from torch import nn

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.mapping_txt = nn.Sequential(
                    nn.Linear(3, 4), nn.Dropout(0.3), nn.Linear(4, 1)
                )
                self.mapping_vis = None

            def forward(self, x):
                return self.mapping_txt(x)

        torch.manual_seed(3)
        inputs = torch.randn(19, 3)
        gold = torch.randn(19, 1)
        schedule = epoch_batches(19, 2, 13, 0)
        config = {"fixture": "dropout"}

        def initial():
            torch.manual_seed(9)
            random.seed(9)
            m = Toy()
            o = torch.optim.AdamW(m.parameters(), lr=0.01)
            return m, o

        def train(m, o, start=0, stop=None):
            o.zero_grad(set_to_none=True)
            count = 0
            for j in range(start, len(schedule)):
                ii = schedule[j]
                ((m(inputs[ii]) - gold[ii]).square().mean() / 2).backward()
                count += 1
                if count == 2 or j + 1 == len(schedule):
                    if count != 2:
                        for p in m.parameters():
                            p.grad.mul_(2 / count)
                    o.step()
                    o.zero_grad(set_to_none=True)
                    count = 0
                    if stop == j + 1:
                        return j + 1
            return len(schedule)

        full, fo = initial()
        train(full, fo)
        part, po = initial()
        n = train(part, po, stop=4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.pt"
            save_snapshot(path, part, po, {"next_batch": n}, config)
            resumed, ro = initial()
            # Deliberately disturb both RNGs before restoring.
            torch.rand(100)
            random.random()
            progress = restore_snapshot(path, resumed, ro, config)
            train(resumed, ro, progress["next_batch"])
            for k, v in full.state_dict().items():
                self.assertTrue(torch.equal(v, resumed.state_dict()[k]), k)
            for k, v in fo.state_dict()["state"].items():
                for key, t in v.items():
                    self.assertTrue(torch.equal(t, ro.state_dict()["state"][k][key]))
            with self.assertRaisesRegex(ValueError, "configuration"):
                restore_snapshot(path, resumed, ro, {"different": True})
            resumed(inputs[:2]).sum().backward()
            with self.assertRaisesRegex(ValueError, "optimizer.step"):
                save_snapshot(path, resumed, ro, progress, config)

    def test_best_checkpoint_does_not_mean_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "mapping").mkdir()
            (root / "mapping/pytorch_model.bin").write_bytes(b"epoch1")
            config = {"arguments": {"epochs": 2}}
            self.assertFalse(completed(root, config))
            (root / "training_state.pt").write_bytes(b"epoch2")
            mark_completed(root, config, 2)
            self.assertTrue(completed(root, config))
            (root / "training_state.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                completed(root, config)


if __name__ == "__main__":
    unittest.main()
