"""
datetime_tool.py — Date/Time Tool
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class DateTimeTool(BaseTool):
    """
    日期时间工具

    能力:
    - 获取当前日期/时间（UTC 或本地时区）
    - 日期计算（加减天数、月数等）
    - 格式化为用户友好格式

    示例:
    - "现在是什么时候？" → "2026-06-13 21:30 UTC"
    - "距今天 100 天后的日期" → "2026-09-21"
    """

    def __init__(self):
        super().__init__(
            name="datetime",
            description="Get current date/time or perform date calculations. Use when the user asks about dates, times, or needs date arithmetic.",
        )

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "add_days", "subtract_days"],
                    "description": "The datetime action to perform",
                },
                "days": {
                    "type": "number",
                    "description": "Number of days to add or subtract (for add_days/subtract_days actions)",
                },
                "tz": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'Asia/Shanghai'). Defaults to UTC.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str = "now",
        days: float = 0,
        tz: str = "UTC",
        **kwargs,
    ) -> ToolResult:
        """执行日期时间操作"""
        try:
            if action == "now":
                result = self._get_current_time(tz)
            elif action == "add_days":
                result = self._add_days(days, tz)
            elif action == "subtract_days":
                result = self._subtract_days(days, tz)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}",
                )

            return ToolResult(
                success=True,
                result=result,
                metadata={"action": action, "tz": tz},
            )
        except Exception as e:
            logger.warning(f"DateTime tool failed: {e}")
            return ToolResult(success=False, error=str(e))

    def _get_current_time(self, tz: str) -> dict:
        if tz == "UTC":
            now = datetime.now(timezone.utc)
        else:
            import zoneinfo
            tzinfo = zoneinfo.ZoneInfo(tz)
            now = datetime.now(tzinfo)

        return {
            "iso": now.isoformat(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "unix": int(now.timestamp()),
        }

    def _add_days(self, days: float, tz: str) -> dict:
        from datetime import timedelta

        if tz == "UTC":
            now = datetime.now(timezone.utc)
        else:
            import zoneinfo
            tzinfo = zoneinfo.ZoneInfo(tz)
            now = datetime.now(tzinfo)

        target = now + timedelta(days=days)
        return {
            "original_date": now.strftime("%Y-%m-%d"),
            "target_date": target.strftime("%Y-%m-%d"),
            "days_added": days,
            "iso": target.isoformat(),
        }

    def _subtract_days(self, days: float, tz: str) -> dict:
        return self._add_days(-days, tz)
