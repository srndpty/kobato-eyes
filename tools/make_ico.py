# tools/make_ico.py
from pathlib import Path

from PIL import Image, ImageFilter

SRC = Path("src/ui/assets/icons/kobato_eye_right-upscale-4x.png")  # 高解像度PNG
OUT = Path("src/ui/assets/icons/kobato_eye_right.ico")
SIZES = [256, 128, 96, 64, 48, 40, 32, 24, 20, 16]

img = Image.open(SRC).convert("RGBA")
icons = [img.resize((s, s), Image.Resampling.LANCZOS) for s in SIZES]
# 16px/20px/24px は若干シャープに
for i, s in enumerate(SIZES):
    if s <= 24:
        icons[i] = icons[i].filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))
icons[0].save(
    OUT,
    format="ICO",
    sizes=[(s, s) for s in SIZES],
)
print("wrote:", OUT)
