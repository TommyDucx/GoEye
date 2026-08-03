# Changelog

## v1.0.0 (2026-08-03)

首个公开发布版本。

### 功能
- **实时屏幕围棋棋盘识别**：基于 `mss` 截屏 + 一维网格线拟合 + 自适应 HSV/kMeans 棋子分类，识别准确率在测试中达 100%。
- **引擎分析 + 下一步推荐**：封装 KataGo（`metal → opencl → eigen` 后端自动回退）与 GNU Go（零配置后备）；给出 Top-5 着法、胜率、目差、PV 与主宰区域。
- **选定区域自动持续监控**：框选一次后自动进入监控，对手落子（棋盘一变动）即自动重新分析，无需反复框选或手动点击。
- **自动棋盘尺寸识别**：自动检测 9 / 13 / 19 路并调整应用内棋盘大小。
- **悬浮面板**：实时棋盘、推荐着法、PV、屏幕落点标记、SGF 导出、自动对弈演示。
- **macOS 启动器**：`/Applications/GoEye.app` 一键启动，README 含完整使用说明。

### 工程
- PyQt6 浮动面板（macOS 上用 `show()` 避免跳到新桌面/黑屏）。
- 引擎不可用时自动降级为「仅识别」模式并提示用户。
- `build_app.sh`：一键用 PyInstaller 打包成 `GoEye.app`（可选 `--dmg`）。
- 权重文件 `models/` 已通过 `.gitignore` 排除（约 60MB，可本地打包进 .app）。

### 使用
```bash
./run.sh          # 启动
./run.sh --test   # 运行识别 + 引擎自测
```
依赖：`pip install -r requirements.txt`（PyQt6 / opencv-python-headless / mss / numpy）；
引擎：`brew install katago`（或 `brew install gnugo` 作为轻量后备）。
