import os
import urllib.request
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. การตั้งค่าฟอนต์ภาษาไทย (Thai Font Setup)
# -----------------------------------------------------------------------------
font_path = "thsarabunnew.ttf"
if not os.path.exists(font_path):
    try:
        font_url = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"
        urllib.request.urlretrieve(font_url, font_path)
    except Exception:
        pass

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "TH Sarabun New"

plt.rcParams.update({"axes.unicode_minus": False, "font.size": 13})


# -----------------------------------------------------------------------------
# 2. ชุดข้อมูลและค่าเริ่มต้น (Dataset & Configurations)
# -----------------------------------------------------------------------------
X_train = np.array([
    [1, 4], [2, 5], [3, 6],   # Class 0: กลุ่มสีน้ำเงิน
    [5, 5], [6, 4], [7, 3],   # Class 1: กลุ่มสีเขียว
    [2, 2], [1, 1], [4, 0]    # Class 2: กลุ่มสีแดง
])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
query_pt = np.array([3.5, 3.5])

class_cfg = {
    0: {"name": "สีน้ำเงิน", "color": "#2563EB"},
    1: {"name": "สีเขียว",  "color": "#059669"},
    2: {"name": "สีแดง",   "color": "#DC2626"}
}


# -----------------------------------------------------------------------------
# 3. อัลกอริทึม KNN (K-Nearest Neighbors Algorithm)
# -----------------------------------------------------------------------------
class KNearestNeighbors:
    """อัลกอริทึม KNN พร้อมการโหวตแบบถ่วงน้ำหนักด้วยระยะทาง"""

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        return self

    def predict(self, point, k, weighted=True):
        # 1. คำนวณระยะห่าง Euclidean ระหว่างจุดทดสอบกับทุกจุด
        dists = np.linalg.norm(self.X - point, axis=1)

        # 2. คัดเลือก k เพื่อนบ้านที่ใกล้ที่สุด
        top_idx = np.argsort(dists)[:k]
        top_labels, top_dists = self.y[top_idx], dists[top_idx]

        # 3. โหวตเลือกคลาส (ถ่วงน้ำหนักด้วยส่วนกลับของระยะทาง: 1/dist)
        weights = 1.0 / (top_dists + 1e-5) if weighted else None
        pred_class = int(np.bincount(top_labels, weights=weights).argmax())

        return pred_class, float(top_dists[-1])


# -----------------------------------------------------------------------------
# 4. รับค่า k และทำนายผลลัพธ์ (User Input & Inference)
# -----------------------------------------------------------------------------
knn = KNearestNeighbors().fit(X_train, y_train)

print(f"\nกำหนดค่า k (เลือก 3 ค่า จำนวนเต็มระหว่าง 1 ถึง {len(X_train)}):")
k_list = []
while len(k_list) < 3:
    try:
        val = int(input(f"  • กรอกค่า k ตัวที่ {len(k_list) + 1}: ").strip())
        if 1 <= val <= len(X_train):
            k_list.append(val)
        else:
            print(f"    [แจ้งเตือน] กรุณากรอกเลขระหว่าง 1 ถึง {len(X_train)}")
    except ValueError:
        print("    [แจ้งเตือน] กรุณากรอกเฉพาะตัวเลขจำนวนเต็ม")

print("\n" + "=" * 45)
print("            ผลการทำนายด้วย KNN")
print("=" * 45)
results = {}
for k in k_list:
    pred, radius = knn.predict(query_pt, k)
    results[k] = (pred, radius)
    print(f"  • เมื่อ k = {k:<2}  ➔  จุด X จัดอยู่ในกลุ่ม: {class_cfg[pred]['name']}")
print("=" * 45 + "\n")


# -----------------------------------------------------------------------------
# 5. แสดงผลลัพธ์ด้วยกราฟ (Data Visualization)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

# 1. วาดจุดข้อมูลฝึกสอน (Training Data)
for c_id, info in class_cfg.items():
    pts = X_train[y_train == c_id]
    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=info["color"], s=110,
        edgecolors="white", linewidths=1.5,
        label=f"Class {c_id}: {info['name']}",
        zorder=3
    )

# 2. วาดจุดทดสอบ (Query Point)
ax.scatter(
    *query_pt,
    c="#1E293B", marker="X", s=130, linewidths=2.5,
    label="จุดทดสอบ X (Query)",
    zorder=5
)

# 3. จัดกลุ่มค่า k ที่มีรัศมีเท่ากัน เพื่อไม่ให้วาดวงกลมซ้ำซ้อน
radius_map = {}
for k, (_, r) in results.items():
    radius_map.setdefault(round(r, 4), []).append(str(k))

# 4. วาดวงกลมรัศมีเพื่อนบ้าน และป้ายกำกับค่า k
y_shifts = [0.35, -0.35, 0.85]
for idx, (r, group) in enumerate(sorted(radius_map.items())):
    # เส้นประขอบเขตวงกลม
    ax.add_patch(
        plt.Circle(query_pt, r, fill=False, ls="--", color="#475569", lw=1.2, alpha=0.75, zorder=2)
    )

    # คำนวณตำแหน่งป้ายข้อความบนเส้นรอบวง x = √(r² - y²)
    dy = y_shifts[idx % len(y_shifts)]
    dx = np.sqrt(max(0, r**2 - dy**2))
    ax.text(
        query_pt[0] + dx + 0.1, query_pt[1] + dy, f"k = {', '.join(group)}",
        fontsize=11, va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", lw=1, alpha=0.9),
        zorder=6
    )

# 5. ตกแต่งแกนและรายละเอียดกราฟให้โมเดิร์น
ax.set_title("การทำงานของ K-Nearest Neighbors (KNN)", fontsize=16, pad=16, fontweight="bold")
ax.set_xlabel("คุณสมบัติ 1 (Feature 1)", fontsize=13, labelpad=8)
ax.set_ylabel("คุณสมบัติ 2 (Feature 2)", fontsize=13, labelpad=8)
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")

# ปรับเส้นกรอบและตารางให้ดูสะอาดตา
ax.grid(True, ls=":", color="#E2E8F0", alpha=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#CBD5E1")

ax.legend(
    loc="upper left", bbox_to_anchor=(1.02, 1),
    frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=11
)

plt.tight_layout()
plt.savefig("knn_result.png", dpi=200, bbox_inches="tight")
print("[บันทึกภาพสำเร็จ] บันทึกไฟล์กราฟไว้ที่: knn_result.png\n")
plt.show()