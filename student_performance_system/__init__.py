"""Student performance prediction package.

中文：包根目录保持轻量，不自动导入 app、训练模型或创建窗口。这样写可以让测试、
工具和外部脚本安全导入这个包，而不会触发 GUI 副作用。

English: The package root stays lightweight: it does not auto-import the app, train
the model, or create windows. This lets tests, tools, and external scripts import
the package without GUI side effects.
"""

# 中文：目前没有包级公开 API；调用方应明确导入所需子模块。
# English: No package-level public API is exported; callers should import submodules explicitly.
__all__ = []
