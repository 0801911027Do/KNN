import os
import urllib.request
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1. ตั้งค่าฟอนต์ภาษาไทย (TH Sarabun New)
# =========================================================
font_path = "thsarabunnew.ttf"
if not os.path.exists(font_path):
    try:
        url = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"
        urllib.request.urlretrieve(url, font_path)
    except Exception:
        pass

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "TH Sarabun New"})

plt.rcParams.update({"axes.unicode_minus": False, "font.size": 14})


# =========================================================
# 2. ชุดข้อมูลและค่าเริ่มต้น
# =========================================================
# ข้อมูลฝึกสอน (X) และป้ายกำกับคลาส (y)
X_train = np.array([
    [1, 4], [2, 5], [3, 6],  # Class 0: สีน้ำเงิน
    [5, 5], [6, 4], [7, 3],  # Class 1: สีเขียว
    [2, 2], [1, 1], [4, 0]   # Class 2: สีแดง
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# จุดทดสอบที่ต้องการทำนาย (Query Point)
query_pt = np.array([3.5, 3.5])

# กำหนดสีและชื่อภาษาไทยของแต่ละคลาส
class_info = {
    0: {"name": "สีน้ำเงิน", "color": "royalblue"},
    1: {"name": "สีเขียว", "color": "mediumseagreen"},
    2: {"name": "สีแดง", "color": "tomato"}
}


# =========================================================
# 3. คลาส KNN อย่างง่าย (เข้าใจการทำงานได้ทันที)
# =========================================================
class KNearestNeighbors:
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=int)
        return self

    def predict(self, point, k, weighted=True):
        # ขั้นที่ 1: คำนวณระยะห่าง Euclidean จากจุดทดสอบไปยังทุกจุดในข้อมูลฝึกสอน
        distances = np.linalg.norm(self.X_train - point, axis=1)

        # ขั้นที่ 2: หาดัชนีและระยะห่างของจุดที่ใกล้ที่สุด k จุดแรก
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = self.y_train[nearest_idx]
        nearest_dists = distances[nearest_idx]

        # ขั้นที่ 3: โหวตหาคลาส (ถ่วงน้ำหนักด้วยระยะทาง เพื่อลด Classification Error)
        if weighted:
            # จุดที่อยู่ใกล้กว่าจะได้น้ำหนักโหวตมากกว่า (1 / distance) ช่วยลดความผิดพลาดในการทำนาย
            weights = 1.0 / (nearest_dists + 1e-5)
            predicted_class = int(np.bincount(nearest_labels, weights=weights).argmax())
        else:
            predicted_class = int(np.bincount(nearest_labels).argmax())

        # ระยะห่างของเพื่อนบ้านตัวที่ k (ใช้เป็นรัศมีวาดวงกลม)
        radius = float(nearest_dists[-1])

        return predicted_class, radius


# =========================================================
# 4. รับค่า k และทำนายผล
# =========================================================
knn = KNearestNeighbors().fit(X_train, y_train)

print("กำหนดค่า k (เลือกค่าจำนวนเต็มระหว่าง 1 ถึง 9):")
k_list = []
for i in range(3):
    while True:
        try:
            val = int(input(f"กรอกค่า k ตัวที่ {i+1}: ").strip())
            if 1 <= val <= len(X_train):
                k_list.append(val)
                break
            print(f"  [แจ้งเตือน] กรุณากรอกตัวเลขระหว่าง 1 ถึง {len(X_train)}")
        except ValueError:
            print("  [แจ้งเตือน] กรุณากรอกเฉพาะตัวเลขจำนวนเต็มเท่านั้น")

print("\n=== ผลการทำนายด้วย KNN ===")
# เก็บผลการทำนายและรัศมีสำหรับนำไปวาดกราฟ
results = {}
for k in k_list:
    pred_class, radius = knn.predict(query_pt, k)
    results[k] = (pred_class, radius)
    class_name = class_info[pred_class]["name"]
    print(f"เมื่อ k={k} -> จุด X จัดอยู่ในประเภท: {class_name}")


# =========================================================
# 5. วาดกราฟแสดงผล (Visualization)
# =========================================================
fig, ax = plt.subplots(figsize=(6.5, 6.5))

# 1. วาดจุดข้อมูลฝึกสอนแยกตามคลาส
for c_id, info in class_info.items():
    pts = X_train[y_train == c_id]
    ax.scatter(pts[:, 0], pts[:, 1], c=info["color"], s=90, label=f"Class {c_id}", zorder=3)

# 2. วาดจุดทดสอบ (จุด X)
ax.scatter(query_pt[0], query_pt[1], c="black", marker="x", s=110, linewidths=1.8, label="จุด X", zorder=5)

# 3. จัดกลุ่มค่า k ที่มีรัศมีเท่ากัน เพื่อไม่ให้วาดวงกลมซ้ำซ้อน
radius_map = {}
for k, (_, r) in results.items():
    r_key = round(r, 4)
    radius_map.setdefault(r_key, []).append(str(k))

# 4. วาดวงกลมรัศมีและใส่ป้ายบอกค่า k
y_shifts = [0.35, -0.35, 0.85]
for idx, (r, group) in enumerate(sorted(radius_map.items())):
    label_text = f"k={', '.join(group)}"

    # วาดวงกลมขอบเขตเพื่อนบ้าน
    circle = plt.Circle(query_pt, r, fill=False, ls="--", color="black", lw=1, alpha=0.85, zorder=2)
    ax.add_patch(circle)

    # วางป้ายข้อความบนเส้นรอบวง
    dy = y_shifts[idx % len(y_shifts)]
    dx = np.sqrt(max(0, r**2 - dy**2))
    ax.text(
        query_pt[0] + dx + 0.08, query_pt[1] + dy, label_text,
        color="black", fontsize=12, va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", lw=0.8, alpha=0.85),
        zorder=6
    )

# 5. ตั้งค่าแกนและชื่อกราฟ
ax.set_title("ตัวอย่างการทำงานของ K-Nearest Neighbor", fontsize=16)
ax.set_xlabel("คุณสมบัติ 1", fontsize=14)
ax.set_ylabel("คุณสมบัติ 2", fontsize=14)
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")
ax.grid(True, ls=":", alpha=0.6)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig("knn_result.png", dpi=150, bbox_inches="tight")
plt.show()