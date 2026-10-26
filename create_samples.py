import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

os.makedirs("static/samples", exist_ok=True)

def create_flower_image(species: str, filename: str):
    w, h = 600, 600
    # Natural soft garden background
    base = Image.new("RGB", (w, h), (30, 41, 59))
    draw = ImageDraw.Draw(base)

    # Soft gradient background
    for y in range(h):
        ratio = y / h
        # Deep forest green to dark slate
        r = int(18 + ratio * 20)
        g = int(35 + ratio * 35)
        b = int(28 + ratio * 20)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Add soft bokeh blurred circles
    bokeh_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(bokeh_layer)
    np.random.seed(abs(hash(species)) % 10000)
    for _ in range(18):
        bx = np.random.randint(20, w - 20)
        by = np.random.randint(20, h - 20)
        br = np.random.randint(25, 75)
        opacity = np.random.randint(25, 70)
        b_draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(220, 252, 231, opacity))
    bokeh_layer = bokeh_layer.filter(ImageFilter.GaussianBlur(12))
    base.paste(bokeh_layer, (0, 0), bokeh_layer)

    flower_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    f_draw = ImageDraw.Draw(flower_layer)

    cx, cy = w // 2, h // 2

    # Draw stem and leaf
    stem_color = (34, 197, 94, 255)
    f_draw.line([(cx, cy + 30), (cx + 10, h - 20)], fill=stem_color, width=12)
    # Leaf
    f_draw.polygon([(cx + 8, cy + 140), (cx + 85, cy + 110), (cx + 110, cy + 150), (cx + 15, cy + 180)], fill=(22, 163, 74, 255))

    if species == "rose":
        # Velvet red layers
        for r_size in [140, 115, 90, 65, 40]:
            num_petals = 12
            for i in range(num_petals):
                angle = (i / num_petals) * 2 * math.pi + (r_size * 0.05)
                px = cx + int(math.cos(angle) * (r_size * 0.55))
                py = cy + int(math.sin(angle) * (r_size * 0.55))
                pr = r_size // 2
                shade = 225 - (r_size // 2)
                f_draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(shade, 15, 45, 230), outline=(136, 19, 55, 255), width=2)
        # Rose core
        f_draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(136, 19, 55, 255))

    elif species == "sunflower":
        # Golden yellow petals
        num_petals = 28
        for i in range(num_petals):
            angle = (i / num_petals) * 2 * math.pi
            tip_x = cx + int(math.cos(angle) * 210)
            tip_y = cy + int(math.sin(angle) * 210)
            base_x1 = cx + int(math.cos(angle - 0.12) * 85)
            base_y1 = cy + int(math.sin(angle - 0.12) * 85)
            base_x2 = cx + int(math.cos(angle + 0.12) * 85)
            base_y2 = cy + int(math.sin(angle + 0.12) * 85)
            f_draw.polygon([(tip_x, tip_y), (base_x1, base_y1), (base_x2, base_y2)], fill=(250, 204, 21, 255), outline=(234, 179, 8, 255))
        # Inner dark chocolate disc
        f_draw.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], fill=(69, 26, 3, 255), outline=(113, 63, 18, 255), width=5)
        # Disk floret pattern
        for _ in range(80):
            rad = np.random.uniform(5, 82)
            ang = np.random.uniform(0, 2 * math.pi)
            fx = cx + int(math.cos(ang) * rad)
            fy = cy + int(math.sin(ang) * rad)
            f_draw.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=(161, 98, 7, 240))

    elif species == "tulip":
        # Cup-shaped vibrant scarlet-orange tulip
        # Back petals
        f_draw.ellipse([cx - 100, cy - 130, cx + 10, cy + 80], fill=(225, 29, 72, 255))
        f_draw.ellipse([cx - 10, cy - 130, cx + 100, cy + 80], fill=(239, 68, 68, 255))
        # Center cup
        f_draw.polygon([
            (cx - 90, cy - 40), (cx - 70, cy - 120), (cx - 20, cy - 60),
            (cx, cy - 140), (cx + 20, cy - 60), (cx + 70, cy - 120), (cx + 90, cy - 40),
            (cx + 60, cy + 90), (cx - 60, cy + 90)
        ], fill=(244, 63, 94, 255), outline=(190, 18, 60, 255))

    elif species == "daisy":
        # White radiating petals with golden disk
        num_petals = 22
        for i in range(num_petals):
            angle = (i / num_petals) * 2 * math.pi
            tip_x = cx + int(math.cos(angle) * 190)
            tip_y = cy + int(math.sin(angle) * 190)
            base_x1 = cx + int(math.cos(angle - 0.14) * 60)
            base_y1 = cy + int(math.sin(angle - 0.14) * 60)
            base_x2 = cx + int(math.cos(angle + 0.14) * 60)
            base_y2 = cy + int(math.sin(angle + 0.14) * 60)
            f_draw.polygon([(tip_x, tip_y), (base_x1, base_y1), (base_x2, base_y2)], fill=(255, 255, 255, 255), outline=(226, 232, 240, 255), width=2)
        # Golden center
        f_draw.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(250, 204, 21, 255), outline=(202, 138, 4, 255), width=4)

    elif species == "dandelion":
        # Fine dandelion yellow pompom
        for r_len in [180, 150, 120, 80]:
            num_spikes = 60
            for i in range(num_spikes):
                angle = (i / num_spikes) * 2 * math.pi + (r_len * 0.1)
                tx = cx + int(math.cos(angle) * r_len)
                ty = cy + int(math.sin(angle) * r_len)
                col = (254, 240, 138, 240) if r_len > 140 else (234, 179, 8, 240)
                f_draw.line([(cx, cy), (tx, ty)], fill=col, width=3)
        f_draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(202, 138, 4, 255))

    elif species == "lotus":
        # Sacred Lotus with layered soft pink petals
        for layer, count, rad in [(1, 8, 170), (2, 8, 135), (3, 6, 95)]:
            for i in range(count):
                angle = (i / count) * 2 * math.pi + (layer * 0.4)
                tx = cx + int(math.cos(angle) * rad)
                ty = cy + int(math.sin(angle) * rad)
                bx1 = cx + int(math.cos(angle - 0.28) * (rad * 0.4))
                by1 = cy + int(math.sin(angle - 0.28) * (rad * 0.4))
                bx2 = cx + int(math.cos(angle + 0.28) * (rad * 0.4))
                by2 = cy + int(math.sin(angle + 0.28) * (rad * 0.4))
                f_draw.polygon([(tx, ty), (bx1, by1), (bx2, by2)], fill=(244, 114, 182, 230), outline=(219, 39, 119, 255))
        # Lotus center pod
        f_draw.ellipse([cx - 38, cy - 38, cx + 38, cy + 38], fill=(250, 204, 21, 255), outline=(217, 119, 6, 255), width=3)

    elif species == "orchid":
        # Elegant Moth Orchid
        # Lateral sepals
        f_draw.ellipse([cx - 190, cy - 80, cx - 10, cy + 50], fill=(236, 72, 153, 235), outline=(190, 24, 93, 255))
        f_draw.ellipse([cx + 10, cy - 80, cx + 190, cy + 50], fill=(236, 72, 153, 235), outline=(190, 24, 93, 255))
        # Top sepal
        f_draw.ellipse([cx - 65, cy - 180, cx + 65, cy - 10], fill=(244, 114, 182, 245), outline=(219, 39, 119, 255))
        # Lip / Labellum
        f_draw.polygon([(cx - 45, cy + 10), (cx + 45, cy + 10), (cx + 65, cy + 90), (cx, cy + 140), (cx - 65, cy + 90)],
                       fill=(168, 85, 247, 255), outline=(126, 34, 206, 255))
        # Orchid column
        f_draw.ellipse([cx - 18, cy - 15, cx + 18, cy + 25], fill=(254, 240, 138, 255))

    elif species == "hibiscus":
        # Giant scarlet petals with long pistil
        for i in range(5):
            angle = (i / 5) * 2 * math.pi - math.pi / 2
            px = cx + int(math.cos(angle) * 160)
            py = cy + int(math.sin(angle) * 160)
            f_draw.ellipse([px - 75, py - 75, px + 75, py + 75], fill=(225, 29, 72, 240), outline=(159, 18, 57, 255), width=3)
        # Deep center
        f_draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(136, 19, 55, 255))
        # Prominent projecting stamen
        f_draw.line([(cx, cy), (cx + 110, cy - 90)], fill=(234, 179, 8, 255), width=8)
        f_draw.ellipse([cx + 100, cy - 100, cx + 120, cy - 80], fill=(225, 29, 72, 255))

    elif species == "jasmine":
        # Delicate fragrant star
        for i in range(7):
            angle = (i / 7) * 2 * math.pi
            tx = cx + int(math.cos(angle) * 170)
            ty = cy + int(math.sin(angle) * 170)
            bx1 = cx + int(math.cos(angle - 0.2) * 45)
            by1 = cy + int(math.sin(angle - 0.2) * 45)
            bx2 = cx + int(math.cos(angle + 0.2) * 45)
            by2 = cy + int(math.sin(angle + 0.2) * 45)
            f_draw.polygon([(tx, ty), (bx1, by1), (bx2, by2)], fill=(255, 255, 255, 245), outline=(226, 232, 240, 255))
        f_draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(254, 240, 138, 255))

    elif species == "marigold":
        # Dense ruffled marigold orange pompom
        for r in range(160, 20, -18):
            pts = 24
            for i in range(pts):
                angle = (i / pts) * 2 * math.pi + (r * 0.1)
                mx = cx + int(math.cos(angle) * r)
                my = cy + int(math.sin(angle) * r)
                pr = 28
                col = (249, 115, 22, 240) if r > 90 else (234, 88, 12, 250)
                f_draw.ellipse([mx - pr, my - pr, mx + pr, my + pr], fill=col, outline=(194, 65, 12, 255))

    elif species == "lavender":
        # Vertical lavender spikes
        for ox, oy in [(-50, 0), (0, -25), (50, 10)]:
            f_draw.line([(cx + ox, cy + 180), (cx + ox, cy - 150)], fill=(74, 222, 128, 255), width=7)
            for by in range(cy + 130, cy - 140, -18):
                f_draw.ellipse([cx + ox - 24, by - 12, cx + ox - 4, by + 12], fill=(167, 139, 250, 255))
                f_draw.ellipse([cx + ox + 4, by - 12, cx + ox + 24, by + 12], fill=(147, 51, 234, 255))
                f_draw.ellipse([cx + ox - 10, by - 16, cx + ox + 10, by + 8], fill=(192, 132, 252, 255))

    elif species == "lily":
        # Six large flared tepals
        for i in range(6):
            angle = (i / 6) * 2 * math.pi
            tx = cx + int(math.cos(angle) * 195)
            ty = cy + int(math.sin(angle) * 195)
            bx1 = cx + int(math.cos(angle - 0.35) * 55)
            by1 = cy + int(math.sin(angle - 0.35) * 55)
            bx2 = cx + int(math.cos(angle + 0.35) * 55)
            by2 = cy + int(math.sin(angle + 0.35) * 55)
            f_draw.polygon([(tx, ty), (bx1, by1), (bx2, by2)], fill=(255, 255, 255, 240), outline=(249, 168, 212, 255), width=2)
            # Central freckles
            for dist in range(60, 140, 15):
                fx = cx + int(math.cos(angle) * dist)
                fy = cy + int(math.sin(angle) * dist)
                f_draw.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=(159, 18, 57, 230))
        # Long stamens
        for a_off in [-0.2, -0.1, 0, 0.1, 0.2]:
            st_x = cx + int(math.cos(math.pi / 2 + a_off) * 90)
            st_y = cy - int(math.sin(math.pi / 2 + a_off) * 90)
            f_draw.line([(cx, cy), (st_x, st_y)], fill=(134, 239, 172, 255), width=4)
            f_draw.ellipse([st_x - 7, st_y - 7, st_x + 7, st_y + 7], fill=(180, 83, 9, 255))

    base.paste(flower_layer, (0, 0), flower_layer)
    base.save(f"static/samples/{filename}", "JPEG", quality=95)
    print(f"[GENERATED] static/samples/{filename}")

species_map = {
    "rose": "rose.jpg",
    "sunflower": "sunflower.jpg",
    "tulip": "tulip.jpg",
    "daisy": "daisy.jpg",
    "dandelion": "dandelion.jpg",
    "lotus": "lotus.jpg",
    "orchid": "orchid.jpg",
    "hibiscus": "hibiscus.jpg",
    "jasmine": "jasmine.jpg",
    "marigold": "marigold.jpg",
    "lavender": "lavender.jpg",
    "lily": "lily.jpg"
}

for sp, fn in species_map.items():
    create_flower_image(sp, fn)

print("All sample images generated successfully!")
