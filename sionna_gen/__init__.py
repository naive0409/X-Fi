"""Sionna 合成 CSI 数据集生成管线（复现 MUSE-FM 数据方法 + WiFi 化适配）。

模块：config（全部参数）→ scene_generator（程序化场景 XML+PLY）→
floorplan_renderer（FLOORPLAN_SPEC 标准 PNG，数据集唯一 floorplan 形式）→
make_tensors（PNG -> 4 通道模型输入张量，预处理产物）→ rt_runner（GPU 射追 -> CSI）→
build_cache（编排 + dloc_cache 同构缓存）→ sanity_check（单场景端到端体检）。
运行环境：conda env `sionna`（python 3.11 + sionna-rt==2.2.0）；训练消费在 xfi。
文档：仓库顶层 SIONNA_PIPELINE.md。
"""
