#!/usr/bin/env python3
"""Export a tiny MindSpore .ms model for ms_bench (optional, needs mindspore).

  pip install mindspore
  python generate_tiny_model.py
  -> testdata/tiny.ms
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "testdata" / "tiny.ms"


def main() -> None:
    try:
        import mindspore as ms
        from mindspore import nn, Tensor, export
    except ImportError:
        print("Install: pip install mindspore")
        print("Or copy your own small .ms model to testdata/tiny.ms")
        return

    class TinyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(128, 128, activation="relu")
            self.out = nn.Dense(128, 64)

        def construct(self, x):
            return self.out(self.dense(x))

    net = TinyNet()
    net.set_train(False)
    dummy = Tensor([[0.0] * 128], ms.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    export(net, dummy, file_name=str(OUT.with_suffix("")), file_format="MINDIR")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
