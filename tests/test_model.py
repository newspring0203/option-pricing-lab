import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from option_pricing.model import Option,black_scholes,from_draws,monte_carlo,summarize
from option_pricing.data import parse_prices,estimate_volatility

class PricingTests(unittest.TestCase):
    def setUp(self): self.o=Option(100,100,1,.05,.01,.2)
    def test_known_call(self):
        self.assertAlmostEqual(black_scholes(replace(self.o,dividend_yield=0)),10.450583572185565,12)
    def test_parity(self):
        o=self.o
        self.assertAlmostEqual(black_scholes(o)-black_scholes(o,'put'),100*math.exp(-.01)-100*math.exp(-.05),12)
    def test_expiry(self):
        self.assertEqual(black_scholes(replace(self.o,maturity=0,spot=110)),10)
        self.assertEqual(black_scholes(replace(self.o,maturity=0,spot=90),'put'),10)
    def test_zero_sigma(self):
        o=replace(self.o,volatility=0)
        self.assertAlmostEqual(black_scholes(o),100*(math.exp(-.01)-math.exp(-.05)))
        _,p=from_draws(o,[-2,0,2]);np.testing.assert_allclose(p,black_scholes(o))
    def test_bounds(self):
        o=self.o;c=black_scholes(o)
        self.assertGreaterEqual(c,max(0,o.spot*math.exp(-o.dividend_yield)-o.strike*math.exp(-o.rate)))
        self.assertLessEqual(c,o.spot*math.exp(-o.dividend_yield))
    def test_scalar_terminal_and_payoff(self):
        z=[-2.,-.3,0.,1.,3.];o=self.o
        t,p=from_draws(o,z)
        expected=[o.spot*math.exp((o.rate-o.dividend_yield-.5*o.volatility**2)*o.maturity+o.volatility*math.sqrt(o.maturity)*x) for x in z]
        np.testing.assert_allclose(t,expected,rtol=1e-14)
        np.testing.assert_allclose(p,[math.exp(-o.rate*o.maturity)*max(x-o.strike,0) for x in expected],rtol=1e-14)
    def test_reproducible(self): self.assertEqual(monte_carlo(self.o,n=1000),monte_carlo(self.o,n=1000))
    def test_prefix(self):
        a=np.random.Generator(np.random.PCG64(42)).standard_normal(100)
        b=np.random.Generator(np.random.PCG64(42)).standard_normal(1000)
        np.testing.assert_array_equal(a,b[:100])
    def test_summary(self):
        s=summarize([1,2,3]);self.assertEqual(s['price'],2);self.assertAlmostEqual(s['standard_error'],1/math.sqrt(3))
    def test_antithetic_pair_standard_error(self):
        z=np.random.Generator(np.random.PCG64(7)).standard_normal(100)
        _,a=from_draws(self.o,z);_,b=from_draws(self.o,-z)
        expected=summarize((a+b)/2);actual=monte_carlo(self.o,n=200,seed=7,antithetic=True)
        for k,v in expected.items(): self.assertEqual(actual[k],v)
        self.assertEqual(actual['independent_units'],100)
    def test_monotonicity(self):
        c=black_scholes(self.o)
        self.assertGreater(black_scholes(replace(self.o,volatility=.3)),c)
        self.assertGreater(black_scholes(replace(self.o,spot=110)),c)
        self.assertLess(black_scholes(replace(self.o,strike=110)),c)
    def test_invalid_inputs(self):
        for key,value in [('spot',0),('strike',-1),('volatility',-.1),('maturity',-1),('rate',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):black_scholes(replace(self.o,**{key:value}))
        for n in [True,1,1.5]:
            with self.assertRaises(ValueError):monte_carlo(self.o,n=n)
        with self.assertRaises(ValueError):monte_carlo(self.o,n=5,antithetic=True)
        with self.assertRaises(ValueError):from_draws(self.o,[float('nan')])
        with self.assertRaises(ValueError):black_scholes(self.o,'invalid')
    def test_scaling(self):
        self.assertAlmostEqual(black_scholes(replace(self.o,spot=200,strike=200)),2*black_scholes(self.o))

class DataTests(unittest.TestCase):
    def test_volatility_by_hand(self):
        rows=[{'adjusted_close':x} for x in [100,110,99]]
        r,s=estimate_volatility(rows)
        expected=[math.log(1.1),math.log(.9)]
        np.testing.assert_allclose(r,expected)
        self.assertAlmostEqual(s,np.std(expected,ddof=1)*math.sqrt(252))
    def test_invalid_prices(self):
        with self.assertRaises(ValueError):estimate_volatility([{'adjusted_close':x} for x in [100,0,90]])
        with self.assertRaises(ValueError):estimate_volatility([{'adjusted_close':100}])
    def test_missing_close_rejected(self):
        raw=json.loads(Path('data/aapl_2024_raw.json').read_text())
        raw['chart']['result'][0]['indicators']['quote'][0]['close'][0]=None
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):parse_prices(p)
    def test_actual_data_integrity(self):
        rows,meta=parse_prices('data/aapl_2024_raw.json')
        self.assertEqual(len(rows),252);self.assertEqual(rows[-1]['date'],'2024-12-31')
        self.assertEqual(meta['currency'],'USD')

if __name__=='__main__':unittest.main()
