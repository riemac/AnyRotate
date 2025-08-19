from pathlib import Path

usd_path = str(Path(__file__).parent.parent.parent / "assets" / "leaphand_object_scene.usda")
print(usd_path)
print(usd_path.__class__)