# Summary of findings from forecast models

## Short summary

* At the moment, changing these parameters doesn't have much impact on bonding rate. 

* It does effect emissions, hence yield.

* Here's the effect on trailing 1Y yield by mid-2026:

  * *Do nothing:* 58.7% – 65.8%
  * *Conservative parameter change:* 58.7% (small error)
  * *Aggressive parameter change B:* 54.5% (small error)

* How do these changes set us up for the rest of 2026? We don't want to give you a forecast for the whole year, because we believe that any forecast we could give today has too high a chance of being misleading — the forecast error is too high, and anyway the plan is to revisit the system by mid 2026 which would likely invalidate whatever model we use. 

  But to give you *some* idea of possible outcomes, let's make some very extreme assumptions about the evolution of the system state: let's imagine bonding rate remains at exactly 52% (which is above target) for every single round of 2026, so that the emission rate is adjusted down every 21 hours. Assume also that the DAO ends up making no further adjustments to the system during the year. Then:

  * With a conservative increase in adjustment speed, total dilution for the year reaches 20%, and annual LPT return rate will be 48%.
  * With adjustment speed pushed to the maximum value for which we ran simulations (3x its current value), total dilution for the year would reach 15%, and annual LPT return rate would be 34%.





## Long summary

Just to remind everyone what these terms mean, over a fixed period (a year, or half a year):

* **Nominal yield (%)** is the number of LPT tokens that accrue to a stake position of 100 LPT over the period.
* **Total emissions** is the proportion by which the LPT supply increases over the period.
* **Dilution** is the proportion by which the share of supply represented by a fixed holding of (unstaked) LPT is diminished over the period.

These quantities all have a generally positive relationship to each other: if one goes up, so too do the other two. (The relationship between yield and the other two also depends on bonding rate, but not in a way that will be very significant for this conversation.)

From the perspective of a staker (Orchestrator or Delegator), nominal yield — the number of tokens they receive as rewards — is likely to be the most interesting figure. However, from the perspective of DAO governance it is essential to keep the dilution figure in view. Of these three quantities, it is the one that best captures the fact that the emissions system is fundamentally **subtractive**: it takes funds out of the pockets of some investors and puts them somewhere else. 

Our risk models aim to quantify the effect of certain parameter changes on bonding rate and dilution. Here is what we found:

1. Based on the evidence of the past two years, changing the emissions parameters will likely have **very little effect** on bonding rate. There's some observable effect, but it's much smaller than a generally positive trend and the influence of external market movements. Adjusting these parameters is not associated with any forseeable adverse effect on stake participation.
2. Despite the observable positive trend, the evolution of the bonding rate over the next year is **highly uncertain.** Based on historical data, it is far from a sure bet that it will remain above 50%, or even do so for most of the time. This leaves us with a lot of uncertainty about the future of emissions, dilution, and nominal yield over that period. The range of possible outcomes is very high — from 12% p5 to 14.5% p95. At LPT's current market cap, that 2.5% range represents over $4.5M worth of funds whose destination we can't predict.
3. This uncertainty is not unavoidable. By reducing the `targetBondingRate`, we can reduce it by quite a lot. We only need to reduce the parameter by a few points to achieve massive reduction in forecast error. This is because based on the history of the participation rate, while it's quite likely to dip below the current target of 50% within the next six months, it's quite unlikely to spend much time below, say, 46%. The sizes of historic staking and unstaking events are just not enough to lead us to expect such a large drawdown.
4. And just to reiterate, we don't expect that reducing this setpoint will have much impact on the bonding rate itself. It **only impacts dilution/emissions/yield**, making them more predictable and giving the DAO more control.
5. Once we've reduced the error and are quite sure that emissions will be trending down for most of H1 2026, we can control for the overall dilution for the period by tuning the adjustment speed `inflationChange`. Under those forecasts, increasing `inflationChange` will mean lower overall dilution, emissions, and nominal yield over the period.
6. There is still some uncertainty in the forecasts. The forecasts don't predict that bonding rate *never* dips below the setpoint, only that it doesn't spend *most of the time* below. We can gain more forecast precision about how much time is spent below the setpoint by reducing `targetBondingRate` even further. Alternatively, we can accept uncertainty about this and make up for it by ramping up the adjustment speed. So there's a a small tradeoff between choices of how much we adjust each of the two parameters.
7. If we only allow ourselves to consider parameters that are reasonably close to their current settings, but commit to reducing uncertainty about dilution, the range of feasible values for H1 2026 is 10.5%–12%. To hit an objective below 12%, the DAO would need to increase `inflationChange`. In terms of similarity to current trends, an objective of 12% is the most conservative possible choice.