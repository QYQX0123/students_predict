"""Convenience launcher for the Student Performance Prediction desktop app."""

from student_performance_system.app import main


if __name__ == "__main__":
    # Keep the package entry point in one place and delegate to app.main().
    main()
