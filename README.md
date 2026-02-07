# The Extremity Premium

**Sentiment Regimes and Adverse Selection in Cryptocurrency Markets**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17989810-blue.svg)](https://doi.org/10.5281/zenodo.17989810)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-With_Editor-yellow.svg)](https://doi.org/10.5281/zenodo.17989810)

**Working Paper DAI-2510** | [Dissensus AI](https://dissensus.ai)

## Abstract

Using the Crypto Fear & Greed Index and Bitcoin daily data, we document that sentiment extremity predicts excess uncertainty beyond realized volatility. Extreme fear and extreme greed regimes exhibit significantly higher spreads than neutral periods---a phenomenon we term the "extremity premium." Extended validation on the full Fear & Greed history (February 2018--January 2026, N = 2,896) confirms the finding: within-volatility-quintile comparisons show a significant premium (p < 0.001, Cohen's d = 0.21), Granger causality from uncertainty to spreads is strong (F = 211), and placebo tests reject the null (p < 0.0001). The effect replicates on Ethereum and across 6 of 7 market cycles. However, the premium is sensitive to functional form: comprehensive regression controls absorb regime effects, while nonparametric stratification preserves them. We interpret this as evidence that sentiment extremity captures volatility-regime interactions not fully represented by parametric controls---consistent with, but not conclusively separable from, the F&G Index's embedded volatility component. An agent-based model reproduces the pattern qualitatively. The results suggest that intensity, not direction, drives uncertainty-linked liquidity withdrawal in cryptocurrency markets, though identification of "pure" sentiment effects from volatility remains an open challenge.

## Key Findings

| Finding | Result |
|---------|--------|
| Extremity premium significance | p < 0.001, Cohen's d = 0.21 |
| Granger causality (uncertainty to spreads) | F = 211 |
| Replication across assets | Confirmed on Ethereum |
| Replication across market cycles | 6 of 7 cycles |
| Placebo tests | Reject null (p < 0.0001) |

## Keywords

extremity premium, sentiment regimes, adverse selection, market microstructure, cryptocurrency, agent-based modeling

## Repository Structure

```
sentiment-microstructure-abm/
├── paper/                      # LaTeX source and PDF
│   ├── main.tex               # Paper source
│   ├── main.pdf               # Compiled paper
│   ├── references.bib         # Bibliography
│   ├── figures/               # Paper figures
│   └── tables/                # Paper tables
├── agents/                     # Agent implementations (Market Maker, Informed, Noise, Arbitrageur)
├── simulation/                 # Mesa ABM environment and matching engine
├── signals/                    # Signal processing
├── analysis/                   # Statistical analysis scripts
├── data_ingestion/             # Binance + Reddit API clients
├── feature_engineering/        # Microstructure features and sentiment analyzer
├── monitoring/                 # Dashboard and metrics
├── demo/                       # Demo scripts
├── tests/                      # Test suite
├── config/                     # Configuration files
├── arxiv-submission/           # arXiv submission package
├── CITATION.cff
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@article{farzulla2026extremity,
  author  = {Farzulla, Murad},
  title   = {The Extremity Premium: Sentiment Regimes and Adverse Selection in Cryptocurrency Markets},
  year    = {2026},
  journal = {Dissensus AI Working Paper DAI-2510},
  doi     = {10.5281/zenodo.17989810}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
