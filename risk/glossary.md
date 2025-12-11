# Definitions of terms

* **Emissions.** The rate or schedule on which new tokens are issued. If expressed as a number, the number of tokens as a proportion of current supply that will be issued over a specified period.  That is, if $S_0$ and $S_1$ are the outstanding token supply at the beginning and end of the specified period, the emissions rate over that period is defined to be $E = (S_1-S_0) / S_0$.

  In the Livepeer Minter contract, the per-round emissions scheduled is tracked (in units of ppb) by a state variable named `inflation`.

  Compare https://coinmarketcap.com/academy/glossary/emission.

* **Dilution.** The process or rate of reducing the share of outstanding supply represented by a single token, or owned by a particular entity or class of entities, by issuing more tokens. 

  The dilution rate over a specified period is the proportion by which the share of outstanding supply represented by a given holding will be reduced over the course of that period. If $E$ is the emissions rate for a given period, the dilution rate over that period is $1 - 1/(1+E)$.

* **Yield.** The nominal increase in principal received by an investor in a given investment vehicle over a specified period.

* **Annualised yield.** The nominal increase increase in principal that an investor would receive over the course of  a year if the per-round yield rate upheld for the full course of that year. If $Y$ is the per-round yield and $N$ is the number of rounds per year, then the annualised yield is computed as $(1+Y)^N-1$.

* **Dilution-adjusted yield.** The increase in the share of outstanding supply represented by a given principal that the owner of that principal would receive over a specified period. If $Y$ is the yield and $D$ is the dilution rate over a specified period, the dilution-adjusted yield for that period is $(1+Y)*(1-D)-1$.

* **Bonding rate.** Proportion of outstanding LPT supply locked in the staking system. Also called *participation rate* or *staking rate*. Ranges from 0 to 1 (or 0% to 100% if expressed as a percentage). Computed from bonding ratio via the formula $R\mapsto 1/(1+1/R) = R/(R+1)$.

* **Bonding ratio.** Proportion of staked LPT to unstaked LPT. Ranges from 0 to ∞. Can be computed from bonding rate via the formula $P\mapsto P/(1-P)$.

* **Inflation.** In the Livepeer ecosystem, this word is commonly used to refer to the token emission schedule. It is also the name given to the variable that controls the emission rate in the source code for the Minter contract. Because this term has very different meanings in overlapping context, we find its use to be prone to causing confusion, hence avoid it wherever possible.

