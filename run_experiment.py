"""Offline pricing experiment: python3 run_experiment.py after downloading data."""
import csv
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from option_pricing.data import parse_prices,estimate_volatility,save_csv,verify_source
from option_pricing.model import Option,black_scholes,from_draws,monte_carlo,summarize,control_variate

ROOT=Path(__file__).resolve().parent
SEED=20261105


def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    raw=ROOT/'data/aapl_2024_raw.json'
    verify_source(raw,ROOT/'data/source_manifest.json')
    rows,metadata=parse_prices(raw)
    returns,sigma=estimate_volatility(rows)
    # Raw quote floats are rounded to the quoted cent for the spot input only.
    option=Option(spot=round(rows[-1]['close'],2),strike=250.,maturity=.25,
                  rate=.04,dividend_yield=.005,volatility=sigma)
    benchmark=black_scholes(option)
    rng=np.random.Generator(np.random.PCG64(SEED))
    z=rng.standard_normal(1_000_000)
    terminal,payoffs=from_draws(option,z)
    convergence=[]
    for n in [1000,10000,100000,1000000]:
        stat=summarize(payoffs[:n]);stat.update(paths=n,error=stat['price']-benchmark,
                  absolute_error=abs(stat['price']-benchmark),benchmark_inside_ci=stat['ci95_low']<=benchmark<=stat['ci95_high'])
        convergence.append(stat)
    antithetic=monte_carlo(option,n=1_000_000,seed=SEED,antithetic=True)
    seeds=[monte_carlo(option,n=100000,seed=s) for s in [SEED,SEED+1,SEED+2,SEED+3,SEED+4]]
    sensitivity=[]
    for multiplier in [.8,1.,1.2]:
        a=replace(option,volatility=sigma*multiplier)
        sensitivity.append(dict(sigma=a.volatility,call=black_scholes(a),label=f'{multiplier:.1f}x historical sigma'))
    # Check core invariants here as well as the independent unit tests.
    terminal_target=option.spot*math.exp((option.rate-option.dividend_yield)*option.maturity)
    terminal_mcse=float(terminal.std(ddof=1)/math.sqrt(len(terminal)))
    plain=summarize(payoffs)
    repeat=monte_carlo(option,n=len(z),seed=SEED)
    if repeat['price']!=plain['price']:raise RuntimeError('Seed reproduction failed')
    if not math.isclose(black_scholes(Option(100,100,1,.05,0,.2)),10.450583572185565,abs_tol=1e-12):raise RuntimeError('Known benchmark failed')
    checks=dict(same_seed_reproduced=True,finite_positive_terminal_prices=bool(np.all(np.isfinite(terminal)&(terminal>0))),
                payoff_nonnegative=bool(np.all(payoffs>=0)),
                terminal_mean_zscore=float((terminal.mean()-terminal_target)/terminal_mcse),
                put_call_parity_error=black_scholes(option)-black_scholes(option,'put')-(option.spot*math.exp(-option.dividend_yield*option.maturity)-option.strike*math.exp(-option.rate*option.maturity)))
    cv=control_variate(option)
    cv_comparison=[control_variate(option,n=100000,pilot_n=20000,seed=s) for s in range(SEED,SEED+5)]
    summary=dict(control_variate=cv,control_variate_seed_comparison=cv_comparison,ticker='AAPL',valuation_date=rows[-1]['date'],first_observation=rows[0]['date'],
                 price_observations=len(rows),return_observations=len(returns),data_metadata=metadata,
                 volatility_estimator='sample sd (ddof=1) of adjacent adjusted-close log returns * sqrt(252)',
                 daily_log_return_sd=float(returns.std(ddof=1)),historical_annual_volatility=sigma,
                 annualized_historical_log_mean=float(returns.mean()*252),
                 option=asdict(option),assumed_inputs=['strike','maturity','rate','dividend_yield'],
                 black_scholes=benchmark,monte_carlo=convergence,antithetic=antithetic,
                 antithetic_se_ratio=antithetic['standard_error']/plain['standard_error'],
                 seed_comparison=seeds,sensitivity=sensitivity,checks=checks,seed=SEED,
                 terminal_quantiles=dict(zip(['p05','p50','p95'],map(float,np.quantile(terminal,[.05,.5,.95])))),
                 discounted_payoff_zero_fraction=float(np.mean(payoffs==0)),
                 input_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                 source_manifest=json.loads((ROOT/'data/source_manifest.json').read_text()),
                 environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,rng='numpy.random.PCG64'))
    save_csv(rows,out/'prices_used.csv')
    for name,table in [('convergence',convergence),('seed_comparison',seeds),('sensitivity',sensitivity)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([r['date'] for r in rows],[r['adjusted_close'] for r in rows],color='#176B87');axes[0].set_xticks([0,63,126,189,251],[rows[i]['date'] for i in [0,63,126,189,251]],rotation=30,ha='right');axes[0].set(title='AAPL adjusted close | 2024',ylabel='USD, provider-adjusted')
    axes[1].hist(returns*100,bins=35,color='#176B87');axes[1].set(title=f'251 daily log returns | annual sigma {sigma:.2%}',xlabel='Daily log return (%)',ylabel='Count');fig.tight_layout();fig.savefig(out/'01_history.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(terminal,bins=100,color='#176B87');axes[0].axvline(option.strike,color='#E07A26',label='Strike');axes[0].set(title='Simulated terminal stock prices under Q',xlabel='USD per share');axes[0].legend()
    axes[1].hist(payoffs,bins=100,color='#76ABAE');axes[1].set(title='Discounted call payoffs, including zero mass',xlabel='USD per share',ylabel='Count');fig.tight_layout();fig.savefig(out/'02_simulation.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));counts=[s['paths'] for s in convergence];means=[s['price'] for s in convergence];errors=[1.96*s['standard_error'] for s in convergence];ax.errorbar(counts,means,yerr=errors,fmt='o-',capsize=5,color='#176B87',label='MC estimate and approximate 95% CI');ax.axhline(benchmark,color='#E07A26',linestyle='--',label=f'Black–Scholes {benchmark:.4f}');ax.set_xscale('log');ax.set(xlabel='Number of independent draws',ylabel='Call value (USD per share)',title='Monte Carlo convergence under identical assumptions');ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_convergence.png');plt.close(fig)
    grid=np.linspace(-10,10,1_000_001)
    _,grid_payoff=from_draws(option,grid)
    quadrature=float(np.trapz(grid_payoff*np.exp(-.5*grid**2)/np.sqrt(2*np.pi),grid))
    summary['independent_quadrature']=dict(price=quadrature,absolute_difference=abs(quadrature-benchmark))
    if abs(quadrature-benchmark)>1e-6:raise RuntimeError('Independent numerical integration failed')
    method_rows=[dict(method='plain',main_paths=1_000_000,pilot_paths=0,price=plain['price'],standard_error=plain['standard_error']),
                 dict(method='antithetic',main_paths=1_000_000,pilot_paths=0,price=antithetic['price'],standard_error=antithetic['standard_error']),
                 dict(method='control_variate',main_paths=cv['paths'],pilot_paths=cv['pilot_paths'],price=cv['price'],standard_error=cv['standard_error'])]
    with (out/'method_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(method_rows[0]));writer.writeheader();writer.writerows(method_rows)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([x['method'] for x in method_rows],[x['standard_error'] for x in method_rows],color=['#98C1D9','#176B87','#E07A26']);ax.set(title='Monte Carlo standard error | 1M main draws',ylabel='Standard error (USD per share)');fig.text(.5,.01,'Control variate adds 20,000 independent pilot draws',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'04_method_comparison.png');plt.close(fig)
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
