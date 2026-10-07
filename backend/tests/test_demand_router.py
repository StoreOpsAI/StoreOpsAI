"""수요 예측 요청의 프록시 전달 규칙을 검증합니다."""

import unittest
from unittest.mock import patch

from app.routers import demand
from app.routers.demand import ForecastDraftRequest
from app.schemas.auth import AuthUser


class DemandRouterTest(unittest.TestCase):
    def test_forecast_draft_forwards_sales_status_without_zero_imputation(self):
        payload = ForecastDraftRequest.model_validate(
            {
                "demand": {
                    "product_id": "P001",
                    "as_of": "2026-09-30",
                    "history": [
                        {
                            "date": f"2026-09-{day:02d}",
                            "sales": day if day < 30 else None,
                            "status": "observed" if day < 30 else "missing",
                        }
                        for day in range(24, 31)
                    ],
                    "calendar": [
                        {"date": "2026-10-01", "is_holiday": 0},
                        {"date": "2026-10-02", "is_holiday": 1},
                    ],
                    "country": "US",
                },
                "on_hand": 15,
                "incoming": 3,
            }
        )
        user = AuthUser(
            user_id="U01",
            store_id="S01",
            store_name="매장",
            email="owner@example.com",
            display_name="점주",
        )
        router = demand.create_demand_router(lambda: user)
        endpoint = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/demand/orders/forecast-draft"
        )

        with (
            patch.object(demand, "_demand_token", return_value="service-token"),
            patch.object(demand, "_forward", return_value={"ok": True}) as forward,
        ):
            result = endpoint(payload, "request-1", user)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(forward.call_args.args[0], "/api/orders/forecast-draft")
        forwarded = forward.call_args.args[2]
        self.assertEqual(forwarded["demand"]["history"][-1]["sales"], None)
        self.assertEqual(forwarded["demand"]["history"][-1]["status"], "missing")
        self.assertEqual(forwarded["demand"]["calendar"][1]["is_holiday"], 1)


if __name__ == "__main__":
    unittest.main()