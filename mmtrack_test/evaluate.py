import motmetrics as mm  # 导入该库
import numpy as np

gt_file = './7.txt'
ts_file = './test_file3.txt'

gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果
acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值创建度量器并计算
mh = mm.metrics.create()

# mh模块中有内置的显示格式
summary = mh.compute_many([acc, acc.events.loc[0:1]],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)