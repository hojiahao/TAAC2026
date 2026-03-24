"""
TAAC 2026 腾讯广告算法大赛 - 样本数据EDA分析
数据集: sample_data.parquet
任务: 基于用户行为序列的广告推荐 (序列推荐)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# 加载 Google Sans Code 字体
_font_dir = os.path.expanduser('~/Downloads/Google_Sans_Code/static')
for _f in os.listdir(_font_dir):
    if _f.endswith('.ttf'):
        font_manager.fontManager.addfont(os.path.join(_font_dir, _f))

matplotlib.rcParams['font.sans-serif'] = ['Google Sans Code', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet('sample_data.parquet')
print("=" * 70)
print("TAAC 2026 Sample Data EDA Analysis")
print("=" * 70)

# ============================================================
# 1. 基本信息
# ============================================================
print("\n[1] 基本信息")
print(f"  样本数量: {len(df)}")
print(f"  列名: {df.columns.tolist()}")
print(f"  数据类型:\n{df.dtypes.to_string()}")
print(f"\n  缺失值:\n{df.isnull().sum().to_string()}")

# ============================================================
# 2. 用户和物品分布
# ============================================================
print("\n[2] 用户与物品分布")
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print(f"  唯一用户数: {n_users}")
print(f"  唯一物品数: {n_items}")
print(f"  用户-物品比: {n_users / n_items:.2f}")

# 物品出现频次
item_counts = df['item_id'].value_counts()
print(f"\n  物品出现频次统计:")
print(f"    均值: {item_counts.mean():.2f}")
print(f"    中位数: {item_counts.median():.1f}")
print(f"    最大值: {item_counts.max()}")
print(f"    出现1次的物品占比: {(item_counts == 1).sum() / len(item_counts) * 100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(item_counts.values, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Item Frequency Distribution')
axes[0].set_xlabel('Frequency')
axes[0].set_ylabel('Count')

axes[1].hist(item_counts.values, bins=50, edgecolor='black', alpha=0.7, log=True)
axes[1].set_title('Item Frequency Distribution (Log Scale)')
axes[1].set_xlabel('Frequency')
axes[1].set_ylabel('Count (log)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_item_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. Label 分析 (行为类型)
# ============================================================
print("\n[3] Label (行为类型) 分析")
action_types = []
action_times = []
for label_list in df['label']:
    for item in label_list:
        action_types.append(item['action_type'])
        action_times.append(item['action_time'])

action_counter = Counter(action_types)
print(f"  行为类型分布: {dict(action_counter)}")
print(f"  action_type=1 (点击): {action_counter.get(1, 0)} ({action_counter.get(1, 0)/len(action_types)*100:.1f}%)")
print(f"  action_type=2 (转化): {action_counter.get(2, 0)} ({action_counter.get(2, 0)/len(action_types)*100:.1f}%)")
print(f"  每条样本的label数量: 均为 {df['label'].apply(len).unique()} 个")

fig, ax = plt.subplots(figsize=(8, 5))
types = list(action_counter.keys())
counts = list(action_counter.values())
bars = ax.bar([f'Type {t}' for t in types], counts, color=['#2196F3', '#FF5722'], edgecolor='black')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({count/sum(counts)*100:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_title('Action Type Distribution')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_action_type_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Timestamp 时间分析
# ============================================================
print("\n[4] Timestamp 分析")
import datetime
ts_min = df['timestamp'].min()
ts_max = df['timestamp'].max()
print(f"  最早时间: {datetime.datetime.fromtimestamp(ts_min)} (ts={ts_min})")
print(f"  最晚时间: {datetime.datetime.fromtimestamp(ts_max)} (ts={ts_max})")
print(f"  时间跨度: {(ts_max - ts_min) / 3600:.1f} 小时 ({(ts_max - ts_min) / 86400:.1f} 天)")

# label中的action_time vs timestamp
time_diffs = []
for i, row in df.iterrows():
    for label_item in row['label']:
        diff = label_item['action_time'] - row['timestamp']
        time_diffs.append(diff)

time_diffs = np.array(time_diffs)
print(f"\n  action_time - timestamp 差值统计:")
print(f"    均值: {time_diffs.mean():.1f}秒 ({time_diffs.mean()/60:.1f}分钟)")
print(f"    中位数: {np.median(time_diffs):.1f}秒")
print(f"    最小值: {time_diffs.min():.1f}秒")
print(f"    最大值: {time_diffs.max():.1f}秒")
print(f"    全部>0: {(time_diffs > 0).all()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# timestamp分布
ts_hours = [(t - ts_min) / 3600 for t in df['timestamp']]
axes[0].hist(ts_hours, bins=50, edgecolor='black', alpha=0.7, color='#4CAF50')
axes[0].set_title('Timestamp Distribution (hours from min)')
axes[0].set_xlabel('Hours')
axes[0].set_ylabel('Count')

# time diff分布
axes[1].hist(time_diffs, bins=50, edgecolor='black', alpha=0.7, color='#FF9800')
axes[1].set_title('Action Time - Timestamp Diff (seconds)')
axes[1].set_xlabel('Seconds')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_timestamp_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. User Feature 分析
# ============================================================
print("\n[5] User Feature 分析")
user_feat_lens = df['user_feature'].apply(len)
print(f"  用户特征数量统计:")
print(f"    均值: {user_feat_lens.mean():.1f}")
print(f"    最小值: {user_feat_lens.min()}")
print(f"    最大值: {user_feat_lens.max()}")
print(f"    标准差: {user_feat_lens.std():.1f}")

# 统计所有user feature_id
user_fid_counter = Counter()
user_ftype_map = {}
for feats in df['user_feature']:
    for f in feats:
        user_fid_counter[f['feature_id']] += 1
        user_ftype_map[f['feature_id']] = f['feature_value_type']

print(f"\n  用户特征ID总数: {len(user_fid_counter)}")
print(f"  所有特征ID: {sorted(user_fid_counter.keys())}")
print(f"\n  特征类型分布:")
ftype_counter = Counter(user_ftype_map.values())
for ft, cnt in ftype_counter.items():
    print(f"    {ft}: {cnt} 个特征")

# 每个feature_id的出现率
print(f"\n  特征覆盖率 (出现率 < 100% 的特征):")
for fid in sorted(user_fid_counter.keys()):
    rate = user_fid_counter[fid] / len(df) * 100
    if rate < 100:
        print(f"    feature_id={fid}: {rate:.1f}% ({user_ftype_map[fid]})")

fig, ax = plt.subplots(figsize=(16, 5))
fids = sorted(user_fid_counter.keys())
rates = [user_fid_counter[fid] / len(df) * 100 for fid in fids]
ax.bar(range(len(fids)), rates, color='#2196F3', edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(fids)))
ax.set_xticklabels(fids, rotation=90, fontsize=7)
ax.set_title('User Feature Coverage Rate (%)')
ax.set_xlabel('Feature ID')
ax.set_ylabel('Coverage Rate (%)')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_user_feature_coverage.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Item Feature 分析
# ============================================================
print("\n[6] Item Feature 分析")
item_feat_lens = df['item_feature'].apply(len)
print(f"  物品特征数量统计:")
print(f"    均值: {item_feat_lens.mean():.1f}")
print(f"    最小值: {item_feat_lens.min()}")
print(f"    最大值: {item_feat_lens.max()}")

item_fid_counter = Counter()
item_ftype_map = {}
for feats in df['item_feature']:
    for f in feats:
        item_fid_counter[f['feature_id']] += 1
        item_ftype_map[f['feature_id']] = f['feature_value_type']

print(f"\n  物品特征ID总数: {len(item_fid_counter)}")
print(f"  所有特征ID: {sorted(item_fid_counter.keys())}")
print(f"\n  各特征类型:")
for fid in sorted(item_fid_counter.keys()):
    rate = item_fid_counter[fid] / len(df) * 100
    print(f"    feature_id={fid}: type={item_ftype_map[fid]}, coverage={rate:.1f}%")

# 分析特征值范围 (int_value类型)
print(f"\n  int_value型特征的值范围:")
for fid in sorted(item_fid_counter.keys()):
    if item_ftype_map[fid] == 'int_value':
        vals = []
        for feats in df['item_feature']:
            for f in feats:
                if f['feature_id'] == fid and f.get('int_value') is not None:
                    vals.append(f['int_value'])
        if vals:
            vals = np.array(vals)
            print(f"    feature_id={fid}: min={vals.min():.0f}, max={vals.max():.0f}, nunique={len(np.unique(vals))}")

fig, ax = plt.subplots(figsize=(10, 5))
fids = sorted(item_fid_counter.keys())
rates = [item_fid_counter[fid] / len(df) * 100 for fid in fids]
colors = ['#4CAF50' if item_ftype_map[fid] == 'int_value' else '#FF5722' for fid in fids]
ax.bar(range(len(fids)), rates, color=colors, edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(fids)))
ax.set_xticklabels(fids, rotation=45)
ax.set_title('Item Feature Coverage Rate (Green=int_value, Red=other)')
ax.set_xlabel('Feature ID')
ax.set_ylabel('Coverage Rate (%)')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_item_feature_coverage.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Sequence Feature 分析
# ============================================================
print("\n[7] Sequence Feature 分析")
print(f"  seq_feature 包含3个子序列: action_seq, content_seq, item_seq")

# action_seq
print(f"\n  [7.1] action_seq 分析:")
action_seq_lens = []
action_seq_fids = set()
for seq in df['seq_feature']:
    aseq = seq['action_seq']
    action_seq_fids.update([f['feature_id'] for f in aseq])
    # 取第一个特征的数组长度作为序列长度
    if len(aseq) > 0 and aseq[0].get('int_array') is not None:
        action_seq_lens.append(len(aseq[0]['int_array']))

print(f"    特征数量(每条样本): {df['seq_feature'].apply(lambda x: len(x['action_seq'])).unique()}")
print(f"    特征IDs: {sorted(action_seq_fids)}")
action_seq_lens = np.array(action_seq_lens)
print(f"    序列长度统计:")
print(f"      均值: {action_seq_lens.mean():.1f}")
print(f"      中位数: {np.median(action_seq_lens):.1f}")
print(f"      最小值: {action_seq_lens.min()}")
print(f"      最大值: {action_seq_lens.max()}")
print(f"      标准差: {action_seq_lens.std():.1f}")

# content_seq
print(f"\n  [7.2] content_seq 分析:")
content_seq_lens = []
content_seq_fids = set()
for seq in df['seq_feature']:
    cseq = seq['content_seq']
    if isinstance(cseq, np.ndarray):
        for item in cseq:
            if isinstance(item, dict):
                content_seq_fids.add(item['feature_id'])
                if item.get('int_array') is not None:
                    content_seq_lens.append(len(item['int_array']))
                    break

content_seq_nfeats = df['seq_feature'].apply(lambda x: len(x['content_seq']))
print(f"    特征数量(每条样本): min={content_seq_nfeats.min()}, max={content_seq_nfeats.max()}, mean={content_seq_nfeats.mean():.1f}")
print(f"    特征IDs: {sorted(content_seq_fids)}")
content_seq_lens = np.array(content_seq_lens)
print(f"    序列长度统计:")
print(f"      均值: {content_seq_lens.mean():.1f}")
print(f"      中位数: {np.median(content_seq_lens):.1f}")
print(f"      最小值: {content_seq_lens.min()}")
print(f"      最大值: {content_seq_lens.max()}")

# item_seq
print(f"\n  [7.3] item_seq 分析:")
item_seq_lens = []
item_seq_fids = set()
for seq in df['seq_feature']:
    iseq = seq['item_seq']
    if isinstance(iseq, np.ndarray):
        for item in iseq:
            if isinstance(item, dict):
                item_seq_fids.add(item['feature_id'])
                if item.get('int_array') is not None:
                    item_seq_lens.append(len(item['int_array']))
                    break

item_seq_nfeats = df['seq_feature'].apply(lambda x: len(x['item_seq']))
print(f"    特征数量(每条样本): min={item_seq_nfeats.min()}, max={item_seq_nfeats.max()}, mean={item_seq_nfeats.mean():.1f}")
print(f"    特征IDs: {sorted(item_seq_fids)}")
item_seq_lens = np.array(item_seq_lens)
print(f"    序列长度统计:")
print(f"      均值: {item_seq_lens.mean():.1f}")
print(f"      中位数: {np.median(item_seq_lens):.1f}")
print(f"      最小值: {item_seq_lens.min()}")
print(f"      最大值: {item_seq_lens.max()}")

# 绘制序列长度分布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(action_seq_lens, bins=50, edgecolor='black', alpha=0.7, color='#2196F3')
axes[0].set_title(f'Action Seq Length Distribution\n(mean={action_seq_lens.mean():.0f}, median={np.median(action_seq_lens):.0f})')
axes[0].set_xlabel('Sequence Length')
axes[0].set_ylabel('Count')

axes[1].hist(content_seq_lens, bins=50, edgecolor='black', alpha=0.7, color='#4CAF50')
axes[1].set_title(f'Content Seq Length Distribution\n(mean={content_seq_lens.mean():.0f}, median={np.median(content_seq_lens):.0f})')
axes[1].set_xlabel('Sequence Length')
axes[1].set_ylabel('Count')

axes[2].hist(item_seq_lens, bins=50, edgecolor='black', alpha=0.7, color='#FF9800')
axes[2].set_title(f'Item Seq Length Distribution\n(mean={item_seq_lens.mean():.0f}, median={np.median(item_seq_lens):.0f})')
axes[2].set_xlabel('Sequence Length')
axes[2].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_sequence_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 8. 特征ID全局映射
# ============================================================
print("\n[8] 特征ID全局映射")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  Feature ID Range  │  Category      │  Value Type          │")
print("  ├─────────────────────────────────────────────────────────────┤")
print("  │  1-5               │  User (sparse) │  int_value           │")
print("  │  6-17, 75,77-79    │  Item           │  int_value           │")
print("  │  18-28             │  Action Seq     │  int_array           │")
print("  │  29-40             │  Item Seq       │  int_array           │")
print("  │  41-49             │  Content Seq    │  int_array           │")
print("  │  50-105            │  User (dense)   │  int_value/int_array │")
print("  └─────────────────────────────────────────────────────────────┘")

# ============================================================
# 9. 数据schema总结
# ============================================================
print("\n[9] 数据Schema总结")
print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        数据结构概览                                   │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  user_id      : 用户标识 (str, e.g., "user_3059")                    │
  │  item_id      : 物品/广告标识 (int)                                   │
  │  timestamp    : 曝光时间戳 (int, Unix timestamp)                      │
  │  user_feature : 用户画像特征列表 (List[Dict])                         │
  │                 - 约38-54个特征, feature_id: 1-105                    │
  │                 - 以int_value为主, 部分为int_array                    │
  │  item_feature : 物品/广告特征列表 (List[Dict])                        │
  │                 - 约12-16个特征, feature_id: 6-79                     │
  │                 - 全部为int_value                                     │
  │  seq_feature  : 用户历史行为序列 (Dict)                               │
  │                 - action_seq: 10个特征(id:19-28), 用户行为序列        │
  │                 - content_seq: ~9个特征(id:41-49), 内容消费序列       │
  │                 - item_seq: ~12个特征(id:29-40), 物品交互序列         │
  │  label        : 标签 (List[Dict])                                     │
  │                 - action_type: 1=点击, 2=转化                         │
  │                 - action_time: 行为发生时间戳                          │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 10. 行为类型与物品的关联分析
# ============================================================
print("[10] 行为类型与物品关联分析")
df['action_type'] = df['label'].apply(lambda x: x[0]['action_type'])
click_items = set(df[df['action_type'] == 1]['item_id'])
convert_items = set(df[df['action_type'] == 2]['item_id'])
print(f"  点击行为涉及的物品数: {len(click_items)}")
print(f"  转化行为涉及的物品数: {len(convert_items)}")
print(f"  交集物品数: {len(click_items & convert_items)}")
print(f"  转化率: {len(df[df['action_type']==2])/len(df)*100:.1f}%")

# ============================================================
# 11. 综合统计图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 用户特征数量分布
axes[0, 0].hist(user_feat_lens, bins=30, edgecolor='black', alpha=0.7, color='#9C27B0')
axes[0, 0].set_title('User Feature Count Distribution')
axes[0, 0].set_xlabel('Number of Features')

# 物品特征数量分布
axes[0, 1].hist(item_feat_lens, bins=10, edgecolor='black', alpha=0.7, color='#009688')
axes[0, 1].set_title('Item Feature Count Distribution')
axes[0, 1].set_xlabel('Number of Features')

# 行为类型饼图
axes[1, 0].pie(counts, labels=[f'Type {t}\n({c})' for t, c in zip(types, counts)],
               colors=['#2196F3', '#FF5722'], autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Action Type Distribution')

# 时间分布
dates = [datetime.datetime.fromtimestamp(t) for t in df['timestamp']]
axes[1, 1].hist([d.hour + d.minute/60 for d in dates], bins=24, edgecolor='black', alpha=0.7, color='#FFC107')
axes[1, 1].set_title('Hour of Day Distribution')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Count')

plt.suptitle('TAAC 2026 Sample Data Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print(f"EDA分析完成! 所有图表已保存到 {OUTPUT_DIR}/ 目录")
print(f"{'='*70}")
print(f"""
生成的图表:
  1. 01_item_distribution.png      - 物品频次分布
  2. 02_action_type_distribution.png - 行为类型分布
  3. 03_timestamp_analysis.png     - 时间戳分析
  4. 04_user_feature_coverage.png  - 用户特征覆盖率
  5. 05_item_feature_coverage.png  - 物品特征覆盖率
  6. 06_sequence_length_distribution.png - 序列长度分布
  7. 07_overview.png               - 综合总览
""")
