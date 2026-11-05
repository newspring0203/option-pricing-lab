import json
import math
from pathlib import Path
import tempfile
import unittest
from option_pricing.model import Option,control_variate
from option_pricing.data import verify_source

class ControlTests(unittest.TestCase):
    def test_linear_payoff_control(self):
        o=Option(100,1,.25,.04,.005,.01)
        a=control_variate(o,n=10000,pilot_n=1000)
        expected=100*math.exp(-.005*.25)-math.exp(-.04*.25)
        self.assertAlmostEqual(a['price'],expected,11)
        self.assertLess(a['standard_error'],1e-12)
    def test_expiry_deterministic(self):
        a=control_variate(Option(110,100,0,.04,.005,.2),n=100,pilot_n=50)
        self.assertEqual(a['price'],10);self.assertEqual(a['standard_error'],0)
    def test_repeatability_and_accounting(self):
        o=Option(100,100,1,.04,0,.2)
        a=control_variate(o,n=1000,pilot_n=200)
        self.assertEqual(a,control_variate(o,n=1000,pilot_n=200))
        self.assertEqual(a['total_paths'],1200);self.assertEqual(a['independent_units'],1000)
    def test_invalid_counts(self):
        o=Option(100,100,1,.04,0,.2)
        for kw in [{'n':True},{'pilot_n':1},{'n':1.5}]:
            with self.assertRaises(ValueError):control_variate(o,**kw)
    def test_source_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';p.write_text('changed')
            with self.assertRaises(ValueError):verify_source(p,'data/source_manifest.json')

if __name__=='__main__':unittest.main()
