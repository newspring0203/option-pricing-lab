"""Black–Scholes–Merton and exact-terminal GBM Monte Carlo.

Rates r and q are annual continuously compounded rates; T is measured in years.
Prices are USD per underlying share. Volatility is an annual standard deviation.
Historical realized volatility is an input estimate, not implied volatility.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    volatility: float

    def validate(self):
        for key,value in vars(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        if self.spot<=0 or self.strike<=0:
            raise ValueError('spot and strike must be positive')
        if self.maturity<0 or self.volatility<0 or self.dividend_yield<0:
            raise ValueError('T, sigma and q must be nonnegative')


def normal_cdf(x):
    return .5*math.erfc(-x/math.sqrt(2))


def black_scholes(option,kind='call'):
    option.validate()
    if kind not in ('call','put'):raise ValueError('kind must be call or put')
    s,k,t,r,q,sigma=(option.spot,option.strike,option.maturity,option.rate,option.dividend_yield,option.volatility)
    if t==0:return max(s-k,0.) if kind=='call' else max(k-s,0.)
    sd,kd=s*math.exp(-q*t),k*math.exp(-r*t)
    if sigma==0:return max(sd-kd,0.) if kind=='call' else max(kd-sd,0.)
    d1=(math.log(s/k)+(r-q+.5*sigma*sigma)*t)/(sigma*math.sqrt(t))
    d2=d1-sigma*math.sqrt(t)
    if kind=='call':return sd*normal_cdf(d1)-kd*normal_cdf(d2)
    return kd*normal_cdf(-d2)-sd*normal_cdf(-d1)


def terminal_prices(option,z):
    option.validate()
    z=np.asarray(z,dtype=float)
    if z.ndim!=1 or len(z)==0 or not np.isfinite(z).all():
        raise ValueError('normal draws must be a nonempty finite 1-D array')
    exponent=(option.rate-option.dividend_yield-.5*option.volatility**2)*option.maturity
    exponent+=option.volatility*math.sqrt(option.maturity)*z
    with np.errstate(over='ignore',under='ignore'):
        prices=option.spot*np.exp(exponent)
    if not np.isfinite(prices).all() or np.any(prices<=0):
        raise ValueError('terminal prices outside representable range')
    return prices


def summarize(discounted_payoffs):
    values=np.asarray(discounted_payoffs,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('at least two finite independent sampling units required')
    estimate=float(values.mean())
    se=float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(price=estimate,standard_error=se,ci95_low=estimate-1.96*se,
                ci95_high=estimate+1.96*se,independent_units=len(values))


def from_draws(option,z,kind='call'):
    if kind not in ('call','put'):raise ValueError('kind must be call or put')
    terminal=terminal_prices(option,z)
    payoff=np.maximum(terminal-option.strike,0) if kind=='call' else np.maximum(option.strike-terminal,0)
    return terminal,payoff*math.exp(-option.rate*option.maturity)


def monte_carlo(option,n=100000,seed=20261105,antithetic=False):
    if not isinstance(n,int) or isinstance(n,bool) or n<2:raise ValueError('n must be an integer >=2')
    if antithetic and (n%2 or n<4):raise ValueError('antithetic n must be even and >=4')
    rng=np.random.Generator(np.random.PCG64(seed))
    if antithetic:
        z=rng.standard_normal(n//2)
        _,plus=from_draws(option,z);_,minus=from_draws(option,-z)
        # Pairs, not their correlated individual members, are independent units.
        units=(plus+minus)/2
    else:
        z=rng.standard_normal(n)
        _,units=from_draws(option,z)
    return dict(paths=n,seed=seed,method='antithetic' if antithetic else 'plain',**summarize(units))


def control_variate(option,n=1_000_000,pilot_n=20000,seed=20261105):
    """Use an independently fitted pilot coefficient and known discounted S_T mean."""
    option.validate()
    if type(n) is not int or n<2 or type(pilot_n) is not int or pilot_n<2:
        raise ValueError('main and pilot counts must be integers >=2')
    pilot_seed,main_seed=np.random.SeedSequence(seed).spawn(2)
    pilot_rng=np.random.Generator(np.random.PCG64(pilot_seed))
    main_rng=np.random.Generator(np.random.PCG64(main_seed))
    pilot_terminal,pilot_y=from_draws(option,pilot_rng.standard_normal(pilot_n))
    pilot_x=math.exp(-option.rate*option.maturity)*pilot_terminal
    variance=float(pilot_x.var(ddof=1))
    beta=0. if option.volatility==0 or option.maturity==0 or variance==0 else float(np.cov(pilot_y,pilot_x,ddof=1)[0,1]/variance)
    terminal,y=from_draws(option,main_rng.standard_normal(n))
    x=math.exp(-option.rate*option.maturity)*terminal
    expected_x=option.spot*math.exp(-option.dividend_yield*option.maturity)
    adjusted=y-beta*(x-expected_x)
    plain=summarize(y)
    return dict(method='control_variate',paths=n,pilot_paths=pilot_n,total_paths=n+pilot_n,seed=seed,beta=beta,
                control_expectation=expected_x,rng='PCG64; SeedSequence child 0 pilot, child 1 main',
                plain_same_draws=plain,**summarize(adjusted))
