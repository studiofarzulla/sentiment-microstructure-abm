# Sentiment-Microstructure Agent-Based Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Real-time crypto market simulator fusing social sentiment with order-book dynamics**

> **Research Project**: Agent-based model combining uncertainty-aware NLP with market microstructure for crypto market simulation. Built as part of exploring the intersection of behavioral finance, market microstructure, and multi-agent systems.

## Overview

This project implements an agent-based market simulator that combines:
- **Microstructure data**: Real-time order books from Binance
- **Sentiment signals**: Reddit crypto communities (r/CryptoCurrency, r/Bitcoin, etc.) with uncertainty-aware NLP
- **Multi-agent dynamics**: Market Makers, Informed Traders, Noise Traders, Arbitrageurs
- **Regime detection**: Dynamic Factor Models extracting latent market states
- **Full streaming pipeline**: Kafka-based real-time data processing

## Architecture

```
Data Layer:         Binance WebSocket → Kafka → Feature Engineering
Sentiment Layer:    Reddit API → DistilRoBERTa (MC Dropout) → Uncertainty Quantification
Regime Layer:       DFM (rolling window) → Latent Factors (mood, volatility, liquidity)
Simulation Layer:   Mesa ABM → 4 Agent Types → Order Matching Engine
Monitoring Layer:   TimescaleDB → Plotly Dash Dashboard
```

## Project Structure

```
sentiment-microstructure-abm/
├── data_ingestion/          # Binance + Reddit API clients, Kafka producers
├── feature_engineering/     # Microstructure features, sentiment analyzer
├── regime_detection/        # Dynamic Factor Models
├── agents/                  # Market Maker, Informed Trader, Noise Trader, Arbitrageur
├── simulation/              # Mesa market environment, order book, matching engine
├── monitoring/              # Dashboard, metrics, database
├── tests/                   # Unit, integration, validation tests
├── config/                  # Configuration files, environment variables
└── docs/                    # Documentation, architecture diagrams
```

## Setup

### Prerequisites
- Python 3.10+
- Docker + Docker Compose (for Kafka, TimescaleDB)
- GPU access (homelab K3s cluster for training)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Kafka + TimescaleDB
docker-compose up -d
```

## Quick Start

```bash
# 1. Start data ingestion
python data_ingestion/binance_client.py &
python data_ingestion/reddit_client.py &

# 2. Run sentiment analyzer service
python feature_engineering/sentiment_service.py &

# 3. Start simulation
python simulation/run_market.py

# 4. Launch dashboard
python monitoring/dashboard.py
```

## Development Timeline

**Phase 1 (Weeks 1-4):** Foundation - Core modules, sentiment model
**Phase 2 (Weeks 5-8):** Data streams - Binance + Reddit + Kafka
**Phase 3 (Weeks 9-12):** Simulation - Mesa ABM, agent calibration
**Phase 4 (Weeks 13-16):** Deployment - K3s, dashboard, validation

## Key Features

- **Uncertainty-aware sentiment**: Monte Carlo Dropout provides epistemic + aleatoric uncertainty estimates
- **Multi-agent realism**: 4 agent archetypes with distinct behavioral rules
- **Regime detection**: DFM extracts latent factors (market mood, volatility regime, liquidity stress)
- **Full streaming**: Kafka pipeline handles real-time order books + sentiment
- **Validation suite**: Stylized facts, calibration tests, flash crash scenarios

## Research Goals

1. Explore sentiment uncertainty impact on market stability
2. Identify regime transitions via DFM factors
3. Calibrate agent parameters to match real Binance microstructure
4. Test crash propagation under sentiment shocks

## Infrastructure

- **Development**: Local Docker Compose
- **Training**: Homelab K3s cluster (PurrPower node with dual GPUs)
- **Deployment**: K3s cluster with NodePort services

## Current Status

🚧 **Phase 1: Foundation (20% Complete)**

- ✅ Data ingestion (Binance WebSocket + Reddit API with threading)
- ✅ Sentiment analyzer (DistilRoBERTa with MC Dropout uncertainty)
- ✅ Configuration management (Pydantic validation)
- ✅ Test suite (82 tests across modules)
- 🚧 Agent implementations (TODO)
- 🚧 Order book + matching engine (TODO)
- 🚧 DFM regime detection (TODO)

See `docs/IMPLEMENTATION_STATUS.md` for detailed progress.

## Contributing

This is a research project. Contributions welcome via:
- Bug reports and feature requests (open an issue)
- Code improvements (submit a PR)
- Research insights and calibration suggestions

## Citation

If you use this code in your research:

```bibtex
@software{sentiment_microstructure_abm,
  title = {Sentiment-Microstructure Agent-Based Model},
  author = {Farzulla Research},
  year = {2025},
  url = {https://github.com/studiofarzulla/sentiment-microstructure-abm}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [Mesa](https://github.com/projectmesa/mesa) for agent-based modeling
- Sentiment analysis using [DistilRoBERTa](https://huggingface.co/distilroberta-base) with Monte Carlo Dropout
- Market data from [Binance](https://binance.com) WebSocket API
- Social sentiment from Reddit via [PRAW](https://praw.readthedocs.io/)

## Contact

Part of **Farzulla Research** - Exploring interdisciplinary approaches to computational finance, market microstructure, and AI/ML applications in financial markets.

- Website: [farzulla.org](https://farzulla.org) (research)
- Portfolio: [farzulla.com](https://farzulla.com) (creative + technical)

---

**Status**: Active development | **Phase**: Foundation (Weeks 1-4) | **Updated**: October 2025
