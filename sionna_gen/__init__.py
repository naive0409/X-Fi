"""Sionna 合成 CSI 数据集生成管线（复现 MUSE-FM 数据方法 + WiFi 化适配）。

rooms10 对齐主线（Phase PC）：config（GenConfig 全部参数）→ scene_generator（程序化
场景 XML+PLY：外壳 U(7,15)m、0-2 贯通带门洞隔墙、圆柱/方柱、家具）→
pointcloud_exporter（env_pc.npz：全局面积加权表面采样，默认不含房顶）→
rt_runner.rooms10（单 TX-SISO × 轨迹 480 快照 + 可选人体逐快照 solve，几何 LOS 标签）；
build_cache（四阶段编排：scenes/floorplans(QC 可选)/env_pc/rt，断点续跑）；sanity_check
（单场景端到端体检）。deprecated dloc 布局（floorplan_renderer/make_tensors/旧 rt 路径）
保留未删。运行环境：conda env `sionna`（python 3.11 + sionna-rt==2.2.0）；训练消费在
xfi。文档：仓库顶层 SIONNA_PIPELINE.md（唯一维护文档）。
"""
