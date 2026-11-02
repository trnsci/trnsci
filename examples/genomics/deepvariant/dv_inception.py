"""DeepVariant 1.9 WGS classifier rebuilt natively in PyTorch.

DeepVariant's model (``deepvariant/keras_modeling.py::inceptionv3``) is
``tf.keras.applications.InceptionV3(include_top=False, weights=None,
pooling="avg")`` on a ``(100, 221, 7)`` pileup, then dropout (a no-op at
inference) and a ``Dense(3, activation="softmax")`` head named
``classification``. This module reproduces that graph layer for layer so the
SavedModel variables map one-to-one onto it:

* every ``conv2d_bn`` is ``Conv2D(use_bias=False)`` + ``BatchNormalization(
  scale=False, epsilon=1e-3)`` + ReLU;
* Keras ``padding="same"`` only ever appears with stride 1 and odd kernels,
  so symmetric ``k // 2`` padding is exact;
* the in-block ``AveragePooling2D(3, 1, "same")`` excludes padded cells from
  the divisor (TF semantics), i.e. ``count_include_pad=False``.

Keras names the convolutions ``conv2d, conv2d_1, ...`` and the norms
``batch_normalization, batch_normalization_1, ...`` in creation order. The
module creates its ``ConvBN`` units in exactly the order the Keras source
calls ``conv2d_bn``, so unit ``i`` takes ``conv2d_i`` / ``batch_normalization_i``.

The forward pass takes the raw ``uint8`` NHWC pileup (what ``make_examples``
writes) and applies DeepVariant's ``preprocess_images`` on device:
``(x - 128) / 128``. Shipping uint8 keeps the host-to-device copy at 4x fewer
bytes than shipping fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SHAPE = (100, 221, 7)  # H, W, C for the DeepVariant 1.9 WGS model
NUM_CLASSES = 3  # P(hom-ref), P(het), P(hom-alt)
BN_EPS = 1e-3  # Keras BatchNormalization default


class ConvBN(nn.Module):
    """Keras ``conv2d_bn``: conv (no bias) -> BN (no gamma) -> ReLU."""

    def __init__(
        self, cin: int, cout: int, kh: int, kw: int, stride: int = 1, same: bool = True
    ):
        super().__init__()
        pad = (kh // 2, kw // 2) if same else (0, 0)
        if same and stride != 1:
            raise ValueError(
                "Keras 'same' with stride>1 pads asymmetrically; not used by InceptionV3"
            )
        self.conv = nn.Conv2d(
            cin, cout, (kh, kw), stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS, affine=True)
        # Keras uses scale=False: gamma is fixed at 1.
        self.bn.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


def _avgpool_same(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)


def _maxpool_valid(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=2)


class InceptionA(nn.Module):  # mixed0..mixed2
    def __init__(self, cin: int, pool_ch: int):
        super().__init__()
        self.b1 = ConvBN(cin, 64, 1, 1)
        self.b5_1 = ConvBN(cin, 48, 1, 1)
        self.b5_2 = ConvBN(48, 64, 5, 5)
        self.b3_1 = ConvBN(cin, 64, 1, 1)
        self.b3_2 = ConvBN(64, 96, 3, 3)
        self.b3_3 = ConvBN(96, 96, 3, 3)
        self.bp = ConvBN(cin, pool_ch, 1, 1)

    def forward(self, x):
        return torch.cat(
            [
                self.b1(x),
                self.b5_2(self.b5_1(x)),
                self.b3_3(self.b3_2(self.b3_1(x))),
                self.bp(_avgpool_same(x)),
            ],
            1,
        )


class InceptionB(nn.Module):  # mixed3
    def __init__(self, cin: int):
        super().__init__()
        self.b3 = ConvBN(cin, 384, 3, 3, stride=2, same=False)
        self.bd_1 = ConvBN(cin, 64, 1, 1)
        self.bd_2 = ConvBN(64, 96, 3, 3)
        self.bd_3 = ConvBN(96, 96, 3, 3, stride=2, same=False)

    def forward(self, x):
        return torch.cat(
            [self.b3(x), self.bd_3(self.bd_2(self.bd_1(x))), _maxpool_valid(x)], 1
        )


class InceptionC(nn.Module):  # mixed4..mixed7
    def __init__(self, cin: int, c7: int):
        super().__init__()
        self.b1 = ConvBN(cin, 192, 1, 1)
        self.b7_1 = ConvBN(cin, c7, 1, 1)
        self.b7_2 = ConvBN(c7, c7, 1, 7)
        self.b7_3 = ConvBN(c7, 192, 7, 1)
        self.bd_1 = ConvBN(cin, c7, 1, 1)
        self.bd_2 = ConvBN(c7, c7, 7, 1)
        self.bd_3 = ConvBN(c7, c7, 1, 7)
        self.bd_4 = ConvBN(c7, c7, 7, 1)
        self.bd_5 = ConvBN(c7, 192, 1, 7)
        self.bp = ConvBN(cin, 192, 1, 1)

    def forward(self, x):
        return torch.cat(
            [
                self.b1(x),
                self.b7_3(self.b7_2(self.b7_1(x))),
                self.bd_5(self.bd_4(self.bd_3(self.bd_2(self.bd_1(x))))),
                self.bp(_avgpool_same(x)),
            ],
            1,
        )


class InceptionD(nn.Module):  # mixed8
    def __init__(self, cin: int):
        super().__init__()
        self.b3_1 = ConvBN(cin, 192, 1, 1)
        self.b3_2 = ConvBN(192, 320, 3, 3, stride=2, same=False)
        self.b7_1 = ConvBN(cin, 192, 1, 1)
        self.b7_2 = ConvBN(192, 192, 1, 7)
        self.b7_3 = ConvBN(192, 192, 7, 1)
        self.b7_4 = ConvBN(192, 192, 3, 3, stride=2, same=False)

    def forward(self, x):
        return torch.cat(
            [
                self.b3_2(self.b3_1(x)),
                self.b7_4(self.b7_3(self.b7_2(self.b7_1(x)))),
                _maxpool_valid(x),
            ],
            1,
        )


class InceptionE(nn.Module):  # mixed9, mixed10
    def __init__(self, cin: int):
        super().__init__()
        self.b1 = ConvBN(cin, 320, 1, 1)
        self.b3 = ConvBN(cin, 384, 1, 1)
        self.b3_a = ConvBN(384, 384, 1, 3)
        self.b3_b = ConvBN(384, 384, 3, 1)
        self.bd_1 = ConvBN(cin, 448, 1, 1)
        self.bd_2 = ConvBN(448, 384, 3, 3)
        self.bd_a = ConvBN(384, 384, 1, 3)
        self.bd_b = ConvBN(384, 384, 3, 1)
        self.bp = ConvBN(cin, 192, 1, 1)

    def forward(self, x):
        b3 = self.b3(x)
        bd = self.bd_2(self.bd_1(x))
        return torch.cat(
            [
                self.b1(x),
                self.b3_a(b3),
                self.b3_b(b3),
                self.bd_a(bd),
                self.bd_b(bd),
                self.bp(_avgpool_same(x)),
            ],
            1,
        )


class DeepVariantInceptionV3(nn.Module):
    """uint8 NHWC pileups -> (probabilities, logits), both fp32.

    ``backbone_dtype`` selects the precision of everything up to the global
    average pool; the pooled features, the Dense head and the softmax always
    run in fp32 so the three probabilities sum to 1 within DeepVariant's
    ``round_gls`` tolerance (1e-6).
    """

    def __init__(self, in_ch: int = INPUT_SHAPE[2]):
        super().__init__()
        self.stem = nn.ModuleList(
            [
                ConvBN(in_ch, 32, 3, 3, stride=2, same=False),
                ConvBN(32, 32, 3, 3, same=False),
                ConvBN(32, 64, 3, 3),
                ConvBN(64, 80, 1, 1, same=False),
                ConvBN(80, 192, 3, 3, same=False),
            ]
        )
        self.mixed = nn.ModuleList(
            [
                InceptionA(192, 32),
                InceptionA(256, 64),
                InceptionA(288, 64),
                InceptionB(288),
                InceptionC(768, 128),
                InceptionC(768, 160),
                InceptionC(768, 160),
                InceptionC(768, 192),
                InceptionD(768),
                InceptionE(1280),
                InceptionE(2048),
            ]
        )
        self.classification = nn.Linear(2048, NUM_CLASSES)
        self.backbone_dtype = torch.float32

    def conv_units(self) -> list[ConvBN]:
        """ConvBN units in Keras creation order (unit i <-> conv2d_i)."""
        return [m for m in self.modules() if isinstance(m, ConvBN)]

    def forward(self, x_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepVariant preprocess_images: cast, subtract 128, divide by 128.
        x = x_u8.to(torch.float32)
        x = (x - 128.0) / 128.0
        x = x.permute(0, 3, 1, 2).to(self.backbone_dtype)  # NHWC -> NCHW
        s = self.stem
        x = s[1](s[0](x))
        x = _maxpool_valid(s[2](x))
        x = _maxpool_valid(s[4](s[3](x)))
        for m in self.mixed:
            x = m(x)
        feat = x.to(torch.float32).mean(dim=(2, 3))
        logits = self.classification(feat)
        return torch.softmax(logits, dim=-1), logits


def build_model(
    state_dict_path: str | None = None, backbone_dtype: torch.dtype = torch.float32
):
    model = DeepVariantInceptionV3()
    assert len(model.conv_units()) == 94
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
    model.eval()
    if backbone_dtype != torch.float32:
        # Cast the backbone only; the head stays fp32.
        for unit in model.conv_units():
            unit.to(backbone_dtype)
    model.backbone_dtype = backbone_dtype
    return model
