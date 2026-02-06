"""
Prediction Market Experiment: Three-Scale Sentiment Analysis

This script explores prediction market data from Polymarket/Kalshi via Dome API
to test whether prediction market sentiment provides additional value beyond
macro (Fear & Greed) and micro (social media) sentiment.

Research Questions:
1. Can we get usable prediction market data for crypto?
2. Does prediction market uncertainty (disagreement) correlate with market spreads?
3. How does prediction market sentiment compare to Fear & Greed Index?

Author: Murad Farzulla
Date: January 2026
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.dome_client import DomeAPIClient, PredictionMarket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explore_available_markets(client: DomeAPIClient) -> List[Dict]:
    """
    Try to discover available crypto-related markets.
    """
    print("\n" + "="*70)
    print("EXPLORING AVAILABLE PREDICTION MARKETS")
    print("="*70)
    
    # Try fetching default crypto markets
    found_markets = []
    
    # Common crypto prediction market patterns
    test_slugs = [
        # Bitcoin price targets (2025)
        "will-bitcoin-price-be-above-50000-usd-on-december-31-2025",
        "will-bitcoin-reach-100000-usd-before-2026",
        "bitcoin-above-100k-on-march-31",
        "bitcoin-above-100000-on-march-31",
        # Bitcoin price targets (2026)
        "will-bitcoin-price-be-above-100000-usd-on-december-31-2026",
        "bitcoin-price-above-100000-end-of-2026",
        # Ethereum
        "will-ethereum-price-be-above-3000-usd-on-december-31-2025",
        "ethereum-above-5000-2025",
        # General crypto
        "crypto-total-market-cap-above-3-trillion-2025",
        # ETF related
        "will-bitcoin-etf-approval-happen-in-2025",
        "spot-bitcoin-etf-2024",
        # Political markets (as baseline test)
        "will-gavin-newsom-win-the-2028-us-presidential-election",
        "presidential-election-winner-2024",
        "will-donald-trump-win-the-2024-presidential-election",
    ]
    
    for slug in test_slugs:
        print(f"\nTrying: {slug[:60]}...")
        market = client.get_market(slug)
        if market:
            print(f"  ✓ Found: {market.question[:50]}...")
            print(f"    Price: {market.current_price:.2%} | Sentiment: {market.sentiment:.3f}")
            print(f"    Volume 24h: ${market.volume_24h:,.0f}" if market.volume_24h else "    Volume: N/A")
            found_markets.append({
                'slug': market.market_slug,
                'question': market.question,
                'price': market.current_price,
                'sentiment': market.sentiment,
                'volume_24h': market.volume_24h,
            })
        time.sleep(1.2)  # Rate limiting
    
    return found_markets


def fetch_market_activity(client: DomeAPIClient, market_slug: str) -> Optional[List[Dict]]:
    """
    Fetch recent activity/trades for a market.
    """
    print(f"\nFetching activity for: {market_slug[:50]}...")
    activity = client.get_activity(market_slug, limit=50)
    if activity:
        print(f"  Found {len(activity)} activity records")
        return activity
    else:
        print("  No activity found")
        return None


def fetch_orderbook(client: DomeAPIClient, market_slug: str) -> Optional[Dict]:
    """
    Fetch orderbook for microstructure analysis.
    """
    print(f"\nFetching orderbook for: {market_slug[:50]}...")
    orderbook = client.get_orderbook(market_slug)
    if orderbook:
        print(f"  Orderbook retrieved")
        # Try to extract spread
        if 'bids' in orderbook and 'asks' in orderbook:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            if bids and asks:
                try:
                    best_bid = float(bids[0][0]) if isinstance(bids[0], list) else float(bids[0].get('price', 0))
                    best_ask = float(asks[0][0]) if isinstance(asks[0], list) else float(asks[0].get('price', 0))
                    spread = best_ask - best_bid
                    print(f"  Best Bid: {best_bid:.4f} | Best Ask: {best_ask:.4f} | Spread: {spread:.4f}")
                except (IndexError, ValueError, TypeError) as e:
                    print(f"  Could not parse spread: {e}")
        return orderbook
    else:
        print("  No orderbook found")
        return None


def analyze_prediction_market_uncertainty(markets: List[Dict]) -> Dict:
    """
    Analyze uncertainty from prediction market data.
    
    Key metrics:
    - Cross-market disagreement (std of sentiments)
    - Price extremity (distance from 0.5)
    - Volume-weighted sentiment
    """
    if not markets:
        return {}
    
    print("\n" + "="*70)
    print("PREDICTION MARKET UNCERTAINTY ANALYSIS")
    print("="*70)
    
    sentiments = [m['sentiment'] for m in markets]
    prices = [m['price'] for m in markets]
    volumes = [m['volume_24h'] or 0 for m in markets]
    
    # Basic stats
    mean_sentiment = np.mean(sentiments)
    std_sentiment = np.std(sentiments)
    
    # Volume-weighted sentiment
    total_volume = sum(volumes)
    if total_volume > 0:
        weighted_sentiment = sum(s * v for s, v in zip(sentiments, volumes)) / total_volume
    else:
        weighted_sentiment = mean_sentiment
    
    # Price extremity (how far from 0.5)
    extremities = [abs(p - 0.5) for p in prices]
    mean_extremity = np.mean(extremities)
    
    # Uncertainty metrics
    # Higher std = more disagreement = more uncertainty
    epistemic_from_disagreement = min(1.0, std_sentiment / 0.5)
    
    results = {
        'n_markets': len(markets),
        'mean_sentiment': float(mean_sentiment),
        'std_sentiment': float(std_sentiment),
        'weighted_sentiment': float(weighted_sentiment),
        'mean_price': float(np.mean(prices)),
        'price_range': (float(min(prices)), float(max(prices))),
        'mean_extremity': float(mean_extremity),
        'total_volume_24h': float(total_volume),
        'epistemic_uncertainty': float(epistemic_from_disagreement),
    }
    
    print(f"\nMarkets analyzed: {results['n_markets']}")
    print(f"Mean Sentiment: {results['mean_sentiment']:.3f}")
    print(f"Sentiment Std (disagreement): {results['std_sentiment']:.3f}")
    print(f"Volume-Weighted Sentiment: {results['weighted_sentiment']:.3f}")
    print(f"Price Range: {results['price_range'][0]:.2%} - {results['price_range'][1]:.2%}")
    print(f"Mean Extremity (|p - 0.5|): {results['mean_extremity']:.3f}")
    print(f"Epistemic Uncertainty (from disagreement): {results['epistemic_uncertainty']:.3f}")
    
    return results


def compare_with_fear_greed(prediction_sentiment: float) -> Dict:
    """
    Compare prediction market sentiment with Fear & Greed Index.
    """
    print("\n" + "="*70)
    print("COMPARING WITH FEAR & GREED INDEX")
    print("="*70)
    
    # Load our existing Fear & Greed data
    data_path = Path(__file__).parent.parent / "data" / "datasets" / "btc_sentiment_daily.csv"
    
    if not data_path.exists():
        print(f"Fear & Greed data not found at {data_path}")
        return {}
    
    df = pd.read_csv(data_path)
    
    # Get recent Fear & Greed value
    if 'fear_greed' not in df.columns:
        print("No fear_greed column in data")
        return {}
    
    latest_fg = df['fear_greed'].iloc[-1]
    latest_fg_sentiment = (latest_fg - 50) / 50  # Convert to [-1, 1]
    
    # Historical stats
    fg_mean = df['fear_greed'].mean()
    fg_std = df['fear_greed'].std()
    fg_sentiment_mean = (fg_mean - 50) / 50
    
    divergence = prediction_sentiment - latest_fg_sentiment
    
    results = {
        'latest_fear_greed': float(latest_fg),
        'latest_fg_sentiment': float(latest_fg_sentiment),
        'prediction_sentiment': float(prediction_sentiment),
        'divergence': float(divergence),
        'fg_historical_mean': float(fg_mean),
        'fg_historical_std': float(fg_std),
    }
    
    print(f"\nLatest Fear & Greed Index: {results['latest_fear_greed']:.1f}")
    print(f"Fear & Greed Sentiment: {results['latest_fg_sentiment']:.3f}")
    print(f"Prediction Market Sentiment: {results['prediction_sentiment']:.3f}")
    print(f"Divergence (PM - F&G): {results['divergence']:.3f}")
    
    if abs(divergence) > 0.3:
        print(f"\n⚠️  SIGNIFICANT DIVERGENCE detected between prediction markets and Fear & Greed!")
        if divergence > 0:
            print("   Prediction markets MORE BULLISH than Fear & Greed")
        else:
            print("   Prediction markets MORE BEARISH than Fear & Greed")
    
    return results


def analyze_orderbook_microstructure(orderbook: Dict) -> Dict:
    """
    Analyze orderbook microstructure from prediction market.
    """
    if not orderbook:
        return {}
    
    print("\n" + "="*70)
    print("PREDICTION MARKET MICROSTRUCTURE ANALYSIS")
    print("="*70)
    
    results = {}
    
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    if not bids or not asks:
        print("No orderbook data available")
        return {}
    
    try:
        # Parse orderbook
        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []
        
        for bid in bids[:10]:  # Top 10 levels
            if isinstance(bid, list):
                bid_prices.append(float(bid[0]))
                bid_sizes.append(float(bid[1]) if len(bid) > 1 else 0)
            elif isinstance(bid, dict):
                bid_prices.append(float(bid.get('price', 0)))
                bid_sizes.append(float(bid.get('size', 0)))
        
        for ask in asks[:10]:
            if isinstance(ask, list):
                ask_prices.append(float(ask[0]))
                ask_sizes.append(float(ask[1]) if len(ask) > 1 else 0)
            elif isinstance(ask, dict):
                ask_prices.append(float(ask.get('price', 0)))
                ask_sizes.append(float(ask.get('size', 0)))
        
        if bid_prices and ask_prices:
            best_bid = max(bid_prices)
            best_ask = min(ask_prices)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            relative_spread = spread / mid_price if mid_price > 0 else 0
            
            # Depth
            total_bid_depth = sum(bid_sizes)
            total_ask_depth = sum(ask_sizes)
            
            # Imbalance
            total_depth = total_bid_depth + total_ask_depth
            imbalance = (total_bid_depth - total_ask_depth) / total_depth if total_depth > 0 else 0
            
            results = {
                'best_bid': float(best_bid),
                'best_ask': float(best_ask),
                'spread': float(spread),
                'relative_spread_bps': float(relative_spread * 10000),
                'mid_price': float(mid_price),
                'bid_depth': float(total_bid_depth),
                'ask_depth': float(total_ask_depth),
                'order_imbalance': float(imbalance),
            }
            
            print(f"\nBest Bid: {results['best_bid']:.4f}")
            print(f"Best Ask: {results['best_ask']:.4f}")
            print(f"Spread: {results['spread']:.4f} ({results['relative_spread_bps']:.1f} bps)")
            print(f"Mid Price: {results['mid_price']:.4f}")
            print(f"Bid Depth: ${results['bid_depth']:,.0f}")
            print(f"Ask Depth: ${results['ask_depth']:,.0f}")
            print(f"Order Imbalance: {results['order_imbalance']:.3f} (positive = more bids)")
    
    except Exception as e:
        print(f"Error parsing orderbook: {e}")
    
    return results


def run_three_scale_analysis():
    """
    Main experiment: Test three-scale sentiment framework.
    """
    print("\n" + "="*70)
    print("THREE-SCALE SENTIMENT EXPERIMENT")
    print("="*70)
    print("Testing prediction markets as third sentiment scale")
    print("="*70)
    
    try:
        client = DomeAPIClient()
    except ValueError as e:
        print(f"\n❌ Cannot run experiment: {e}")
        return None
    
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'markets_found': [],
        'uncertainty_analysis': {},
        'fear_greed_comparison': {},
        'microstructure': {},
        'conclusions': [],
    }
    
    # Step 1: Explore available markets
    markets = explore_available_markets(client)
    results['markets_found'] = markets
    
    if not markets:
        print("\n❌ No prediction markets found. Cannot proceed with analysis.")
        results['conclusions'].append("No accessible prediction markets found")
        return results
    
    # Step 2: Analyze prediction market uncertainty
    uncertainty = analyze_prediction_market_uncertainty(markets)
    results['uncertainty_analysis'] = uncertainty
    
    # Step 3: Compare with Fear & Greed
    if uncertainty:
        fg_comparison = compare_with_fear_greed(uncertainty['weighted_sentiment'])
        results['fear_greed_comparison'] = fg_comparison
    
    # Step 4: Microstructure analysis on most liquid market
    if markets:
        # Find market with highest volume
        markets_with_volume = [m for m in markets if m.get('volume_24h', 0) > 0]
        if markets_with_volume:
            best_market = max(markets_with_volume, key=lambda x: x.get('volume_24h', 0))
            orderbook = fetch_orderbook(client, best_market['slug'])
            microstructure = analyze_orderbook_microstructure(orderbook)
            results['microstructure'] = microstructure
    
    # Step 5: Draw conclusions
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    conclusions = []
    
    if len(markets) >= 3:
        conclusions.append(f"✓ Found {len(markets)} crypto-related prediction markets")
    elif len(markets) > 0:
        conclusions.append(f"⚠ Only found {len(markets)} markets - limited for cross-market analysis")
    
    if uncertainty.get('std_sentiment', 0) > 0.2:
        conclusions.append(f"✓ High cross-market disagreement (std={uncertainty['std_sentiment']:.3f}) suggests genuine uncertainty")
    elif uncertainty.get('std_sentiment', 0) > 0.1:
        conclusions.append(f"~ Moderate cross-market disagreement (std={uncertainty['std_sentiment']:.3f})")
    
    if results.get('fear_greed_comparison', {}).get('divergence'):
        div = results['fear_greed_comparison']['divergence']
        if abs(div) > 0.2:
            conclusions.append(f"✓ Significant divergence ({div:.3f}) between prediction markets and Fear & Greed - supports multi-scale framework")
        else:
            conclusions.append(f"~ Low divergence ({div:.3f}) - prediction markets align with Fear & Greed")
    
    if results.get('microstructure', {}).get('spread'):
        spread = results['microstructure']['spread']
        conclusions.append(f"✓ Orderbook data available - spread of {spread:.4f} can be compared to crypto exchange spreads")
    
    results['conclusions'] = conclusions
    for c in conclusions:
        print(c)
    
    # Save results
    output_path = Path(__file__).parent / "results" / "prediction_market_experiment.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    """Run the prediction market experiment."""
    print("\n" + "="*70)
    print("PREDICTION MARKET SENTIMENT EXPERIMENT")
    print("Testing Three-Scale Framework for ABM")
    print("="*70)
    
    results = run_three_scale_analysis()
    
    if results:
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        
        # Summary for paper
        print("\nKey findings for paper:")
        if results.get('markets_found'):
            print(f"  - {len(results['markets_found'])} prediction markets accessible")
        if results.get('uncertainty_analysis'):
            ua = results['uncertainty_analysis']
            print(f"  - Prediction market sentiment: {ua.get('weighted_sentiment', 0):.3f}")
            print(f"  - Cross-market uncertainty: {ua.get('epistemic_uncertainty', 0):.3f}")
        if results.get('fear_greed_comparison'):
            fc = results['fear_greed_comparison']
            print(f"  - Fear & Greed sentiment: {fc.get('latest_fg_sentiment', 0):.3f}")
            print(f"  - Divergence: {fc.get('divergence', 0):.3f}")
    
    return results


if __name__ == '__main__':
    main()
