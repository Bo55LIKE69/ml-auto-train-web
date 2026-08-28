# -*- coding: utf-8 -*-
"""生成测试数据集（分类 / 回归），用于演示与端到端验证。
输出到 sample_data/，不进入 git 仓库（.gitignore 已排除 sample_data 外的临时数据）。
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  已生成 {path.name}: {len(rows)} 行, {len(header)} 列")


# ---------- 1. 客户流失预测（分类，二分类） ----------
def gen_churn(n=320):
    header = ["客户ID", "年龄", "月费", "在网时长", "通话时长", "流量使用",
              "投诉次数", "合同类型", "是否流失"]
    contract = ["月付", "年付", "两年合约"]
    rows = []
    for i in range(n):
        age = random.randint(18, 70)
        tenure = random.randint(1, 72)
        monthly = round(random.uniform(30, 120), 1)
        call = round(random.uniform(100, 1200), 1)
        data_use = round(random.uniform(5, 100), 1)
        complaint = random.randint(0, 5)
        ctype = random.choice(contract)
        # 流失概率：高月费 + 短在网 + 多投诉 -> 易流失
        risk = (monthly - 30) / 90 * 0.3 + (72 - tenure) / 71 * 0.4 + complaint * 0.12
        if ctype == "两年合约":
            risk -= 0.25
        churn = "是" if random.random() < min(0.85, max(0.05, risk)) else "否"
        # 随机缺失
        complaint_v = "" if random.random() < 0.04 else complaint
        rows.append([f"C{i:04d}", age, monthly, tenure, call, data_use, complaint_v, ctype, churn])
    write_csv(OUT / "客户流失预测.csv", header, rows)


# ---------- 2. 房价预测（回归） ----------
def gen_house(n=260):
    header = ["房源ID", "面积", "房间数", "楼龄", "距地铁", "学区评分",
              "楼层", "是否电梯", "绿化率", "成交价"]
    rows = []
    for i in range(n):
        area = random.randint(40, 200)
        rooms = random.randint(1, 5)
        age = random.randint(0, 30)
        metro = round(random.uniform(0.2, 5), 1)
        school = random.randint(1, 10)
        floor = random.randint(1, 33)
        elevator = random.choice(["有", "无"])
        green = random.randint(10, 45)
        base = area * 0.9 + rooms * 12 + school * 6 - age * 1.5 - metro * 8
        base += 20 if elevator == "有" else 0
        base += green * 0.3
        price = round(max(50, base + random.gauss(0, 15)), 1)  # 万元
        area_v = "" if random.random() < 0.03 else area
        rows.append([f"H{i:04d}", area_v, rooms, age, metro, school, floor, elevator, green, price])
    write_csv(OUT / "房价预测.csv", header, rows)


# ---------- 3. 学生成绩（分类，沿用风格，更大样本） ----------
def gen_student(n=200):
    header = ["学号", "平时成绩", "出勤率", "作业分", "期中成绩", "是否通过"]
    rows = []
    for i in range(n):
        reg = round(random.uniform(60, 100), 1)
        att = round(random.uniform(0.6, 1.0), 2)
        hw = round(random.uniform(50, 100), 1)
        mid = round(random.uniform(40, 100), 1)
        total = reg * 0.3 + att * 100 * 0.1 + hw * 0.2 + mid * 0.4
        pass_v = "是" if total >= 60 else "否"
        att_v = "" if random.random() < 0.05 else att
        rows.append([f"S{i:04d}", reg, att_v, hw, mid, pass_v])
    write_csv(OUT / "学生成绩分类.csv", header, rows)


if __name__ == "__main__":
    print("生成测试数据集 ->", OUT)
    gen_churn()
    gen_house()
    gen_student()
    print("完成。")
