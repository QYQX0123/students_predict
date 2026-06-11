"""Convenience launcher for the Student Performance Prediction desktop app.

中文：从项目根目录运行本文件即可启动界面，实际入口逻辑保留在包内 app.py。
English: Run this file from the project root to launch the GUI. The actual entry
point remains in the package's app.py module.
"""

from student_performance_system.app import main


if __name__ == "__main__":
    # 中文：仅直接执行时启动；导入本模块不会自动打开窗口。
    # English: Launch only when executed directly; importing does not open the UI.
    main()
