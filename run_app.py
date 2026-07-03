"""Project-root launcher for the desktop application.

中文：这个文件只做一件事：从包内导入真正的 main 并启动它。这样写是为了让用户可以
在项目根目录直接运行 run_app.py，同时保持应用入口集中在 student_performance_system.app。

English: This file does one thing: import the real main function from the package
and run it. It lets users launch run_app.py from the project root while keeping the
application entry point centralized in student_performance_system.app.
"""

from student_performance_system.app import main


if __name__ == "__main__":
    # 中文：只有直接执行脚本才打开 GUI，被测试或其他模块导入时不会产生副作用。
    # English: Open the GUI only on direct execution, not when tests or modules import it.
    main()
