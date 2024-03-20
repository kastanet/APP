from PIL import Image, ImageDraw
import math

# GIF画像のサイズとフレーム数
width, height = 400, 400
num_frames = 30


# 光ファイバーの位置を計算する関数
def calculate_fiber_position(radius, angle_offset):
    center_x, center_y = width // 2, height // 2
    angle_offset = math.radians(angle_offset)
    angle = math.radians(radius * 12) + angle_offset
    x = center_x + math.cos(angle) * radius
    y = center_y + math.sin(angle) * radius
    return x, y


# GIFを作成する
frames = []
for i in range(num_frames):
    # 白い背景の画像を作成
    frame = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(frame)

    # 光ファイバーの位置を計算して描画
    for j in range(12):
        x1, y1 = calculate_fiber_position(j, i * 10)
        x2, y2 = calculate_fiber_position(j, i * 10 + 180)
        draw.line([(x1, y1), (x2, y2)], fill="blue", width=3)

    frames.append(frame)

# GIFを保存する
frames[0].save("fiber_optic.gif", save_all=True, append_images=frames[1:], loop=0, duration=50)
