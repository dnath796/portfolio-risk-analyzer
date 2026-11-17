import unittest
import numpy as np
import pandas as pd

from portfolio_analyzer_fixed import (
    validate_weights, monthly_return_table, select_complete_assets,
    calculate_monthly_series, max_drawdown, statistics,
)


class AnalyzerTests(unittest.TestCase):
    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            validate_weights({"NVDA": .7, "KO": .2})
        with self.assertRaises(ValueError):
            validate_weights({"NVDA": 1.2, "KO": -.2})

    def test_missing_prices_do_not_create_zero_returns(self):
        dates = pd.date_range('2026-01-31', periods=4, freq='ME')
        prices = pd.DataFrame({'A': [100, np.nan, 120, 130],
                               'SPY': [100, 102, 103, 104]}, index=dates)
        rets = monthly_return_table(prices, '2026-02-28', '2026-04-30')
        self.assertTrue(pd.isna(rets.loc['2026-02-28', 'A']))
        self.assertTrue(pd.isna(rets.loc['2026-03-31', 'A']))

    def test_strict_fails_and_drop_assets_renormalizes(self):
        index = pd.date_range('2026-01-31', periods=3, freq='ME')
        r = pd.DataFrame({'A': [.1, .05, .02], 'B': [.01, np.nan, .03],
                          'SPY': [.04, -.02, .01]}, index=index)
        with self.assertRaisesRegex(ValueError, 'Incomplete holding'):
            select_complete_assets(r, {'A': .5, 'B': .5}, 'strict')
        selected, weights, excluded = select_complete_assets(r, {'A': .5, 'B': .5}, 'drop-assets')
        self.assertEqual(excluded, ['B'])
        self.assertAlmostEqual(weights['A'], 1.0)
        self.assertEqual(list(selected), ['A', 'SPY'])

    def test_spy_gaps_are_always_fatal(self):
        r = pd.DataFrame({'A': [.03, .02], 'SPY': [np.nan, .01]},
                         index=pd.date_range('2026-01-31', periods=2, freq='ME'))
        with self.assertRaisesRegex(ValueError, 'Benchmark'):
            select_complete_assets(r, {'A': 1.0}, 'drop-assets')

    def test_weighted_returns_and_fee(self):
        r = pd.DataFrame({'A': [.10, -.10], 'B': [0, .10], 'SPY': [.02, .03]})
        result = calculate_monthly_series(r, pd.Series({'A': .5, 'B': .5}), 0, 0)
        np.testing.assert_allclose(result['Portfolio Gross'], [.05, 0])
        np.testing.assert_allclose(result['Portfolio Net (before trading costs)'], [.05, 0])

    def test_drawdown_includes_initial_capital(self):
        r = pd.Series([-.20, .10, -.10])
        self.assertAlmostEqual(max_drawdown(r), -.208)

    def test_metric_values(self):
        r = pd.Series([.10, -.20, .05, .03])
        b = pd.Series([.04, -.10, .02, -.01])
        stats = statistics(r, b, 0)
        expected_cagr = (1.1 * .8 * 1.05 * 1.03) ** 3 - 1
        self.assertAlmostEqual(stats['CAGR'], expected_cagr)
        self.assertAlmostEqual(stats['Annual volatility'], r.std(ddof=1) * np.sqrt(12))
        self.assertAlmostEqual(stats['Monthly 95% historical VaR (loss convention)'], .1655)
        self.assertAlmostEqual(stats['Monthly 95% Expected Shortfall (loss convention)'], .20)


if __name__ == '__main__':
    unittest.main()
