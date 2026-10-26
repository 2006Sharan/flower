from PIL import Image
from models.flower_classifier import FlowerClassifier

print("Initializing FlowerClassifier...")
fc = FlowerClassifier()

img = Image.open("static/samples/sunflower.jpg")
print("Processing image static/samples/sunflower.jpg...")
res = fc.process_image(img, filename_hint="sunflower.jpg")

print(f"Classification Success: {res['success']}")
print(f"Predicted Species: {res['common_name']} ({res['scientific_name']})")
print(f"Confidence: {res['confidence']}% [{res['confidence_tier']}]")
print(f"Boxes detected: {len(res['boxes'])}")
print("Top 5:")
for item in res["top_5"]:
    print(f"  - {item['common_name']}: {item['confidence']}%")
print(f"Color Palette: {[c['name'] + ' ' + str(c['percentage']) + '%' for c in res['palette']]}")
print("Test completed successfully!")
