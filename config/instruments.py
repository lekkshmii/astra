"""Asset Universe Definitions - Astra Trading Platform.
===================================================

Predefined asset universes for trading strategies.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AssetInfo:
    """Information about a tradeable asset."""

    symbol: str
    name: str
    sector: str
    market_cap: str
    currency: str = "USD"
    exchange: str = "NASDAQ"
    active: bool = True

class AssetUniverse:
    """Asset universe definitions and management.

    Provides predefined sets of assets for different trading strategies.
    """

    def __init__(self) -> None:
        """Initialize asset universes."""
        self._load_universes()

    def _load_universes(self) -> None:
        """Load predefined asset universes."""
        # Technology stocks
        self.tech_universe = {
            "AAPL": AssetInfo("AAPL", "Apple Inc.", "Technology", "Large"),
            "MSFT": AssetInfo("MSFT", "Microsoft Corporation", "Technology", "Large"),
            "GOOGL": AssetInfo("GOOGL", "Alphabet Inc.", "Technology", "Large"),
            "AMZN": AssetInfo("AMZN", "Amazon.com Inc.", "Technology", "Large"),
            "TSLA": AssetInfo("TSLA", "Tesla Inc.", "Technology", "Large"),
            "META": AssetInfo("META", "Meta Platforms Inc.", "Technology", "Large"),
            "NFLX": AssetInfo("NFLX", "Netflix Inc.", "Technology", "Large"),
            "NVDA": AssetInfo("NVDA", "NVIDIA Corporation", "Technology", "Large"),
            "CRM": AssetInfo("CRM", "Salesforce Inc.", "Technology", "Large"),
            "ORCL": AssetInfo("ORCL", "Oracle Corporation", "Technology", "Large"),
        }

        # Financial stocks
        self.finance_universe = {
            "JPM": AssetInfo("JPM", "JPMorgan Chase & Co.", "Finance", "Large"),
            "BAC": AssetInfo("BAC", "Bank of America Corp.", "Finance", "Large"),
            "WFC": AssetInfo("WFC", "Wells Fargo & Company", "Finance", "Large"),
            "GS": AssetInfo("GS", "Goldman Sachs Group Inc.", "Finance", "Large"),
            "MS": AssetInfo("MS", "Morgan Stanley", "Finance", "Large"),
            "C": AssetInfo("C", "Citigroup Inc.", "Finance", "Large"),
            "AXP": AssetInfo("AXP", "American Express Company", "Finance", "Large"),
            "BRK-B": AssetInfo("BRK-B", "Berkshire Hathaway Inc.", "Finance", "Large"),
        }

        # Healthcare stocks
        self.healthcare_universe = {
            "JNJ": AssetInfo("JNJ", "Johnson & Johnson", "Healthcare", "Large"),
            "PFE": AssetInfo("PFE", "Pfizer Inc.", "Healthcare", "Large"),
            "MRNA": AssetInfo("MRNA", "Moderna Inc.", "Healthcare", "Medium"),
            "UNH": AssetInfo("UNH", "UnitedHealth Group Inc.", "Healthcare", "Large"),
            "CVS": AssetInfo("CVS", "CVS Health Corporation", "Healthcare", "Large"),
            "ABBV": AssetInfo("ABBV", "AbbVie Inc.", "Healthcare", "Large"),
        }

        # Consumer stocks
        self.consumer_universe = {
            "WMT": AssetInfo("WMT", "Walmart Inc.", "Consumer", "Large"),
            "COST": AssetInfo("COST", "Costco Wholesale Corp.", "Consumer", "Large"),
            "HD": AssetInfo("HD", "Home Depot Inc.", "Consumer", "Large"),
            "MCD": AssetInfo("MCD", "McDonald\'s Corporation", "Consumer", "Large"),
            "SBUX": AssetInfo("SBUX", "Starbucks Corporation", "Consumer", "Large"),
            "NKE": AssetInfo("NKE", "Nike Inc.", "Consumer", "Large"),
        }

        # Energy stocks
        self.energy_universe = {
            "XOM": AssetInfo("XOM", "Exxon Mobil Corporation", "Energy", "Large"),
            "CVX": AssetInfo("CVX", "Chevron Corporation", "Energy", "Large"),
            "COP": AssetInfo("COP", "ConocoPhillips", "Energy", "Large"),
            "SLB": AssetInfo("SLB", "Schlumberger Limited", "Energy", "Large"),
        }

        # Cryptocurrency proxies
        self.crypto_universe = {
            "BTC-USD": AssetInfo("BTC-USD", "Bitcoin USD", "Cryptocurrency", "Large"),
            "ETH-USD": AssetInfo("ETH-USD", "Ethereum USD", "Cryptocurrency", "Large"),
            "COIN": AssetInfo("COIN", "Coinbase Global Inc.", "Cryptocurrency", "Medium"),
            "MSTR": AssetInfo("MSTR", "MicroStrategy Inc.", "Cryptocurrency", "Medium"),
        }

        # ETFs and indices
        self.etf_universe = {
            "SPY": AssetInfo("SPY", "SPDR S&P 500 ETF", "ETF", "Large"),
            "QQQ": AssetInfo("QQQ", "Invesco QQQ Trust", "ETF", "Large"),
            "IWM": AssetInfo("IWM", "iShares Russell 2000 ETF", "ETF", "Large"),
            "VTI": AssetInfo("VTI", "Vanguard Total Stock Market ETF", "ETF", "Large"),
            "TLT": AssetInfo("TLT", "iShares 20+ Year Treasury Bond ETF", "ETF", "Large"),
            "GLD": AssetInfo("GLD", "SPDR Gold Trust", "ETF", "Large"),
            "VIX": AssetInfo("VIX", "CBOE Volatility Index", "ETF", "Large"),
        }

        # Combined universes
        self.sp500_sample = {**self.tech_universe, **self.finance_universe,
                           **self.healthcare_universe, **self.consumer_universe}

        self.all_stocks = {**self.tech_universe, **self.finance_universe,
                          **self.healthcare_universe, **self.consumer_universe,
                          **self.energy_universe}

        self.diversified_portfolio = {
            "AAPL": self.tech_universe["AAPL"],
            "MSFT": self.tech_universe["MSFT"],
            "JPM": self.finance_universe["JPM"],
            "JNJ": self.healthcare_universe["JNJ"],
            "WMT": self.consumer_universe["WMT"],
            "XOM": self.energy_universe["XOM"],
            "SPY": self.etf_universe["SPY"],
        }

    def get_universe(self, name: str) -> Dict[str, AssetInfo]:
        """Get asset universe by name."""
        universes = {
            "tech": self.tech_universe,
            "finance": self.finance_universe,
            "healthcare": self.healthcare_universe,
            "consumer": self.consumer_universe,
            "energy": self.energy_universe,
            "crypto": self.crypto_universe,
            "etf": self.etf_universe,
            "sp500_sample": self.sp500_sample,
            "all_stocks": self.all_stocks,
            "diversified": self.diversified_portfolio,
        }

        if name not in universes:
            msg = f"Unknown universe: {name}. Available: {list(universes.keys())}"
            raise ValueError(msg)

        return universes[name]

    def get_symbols(self, universe_name: str) -> List[str]:
        """Get list of symbols for a universe."""
        universe = self.get_universe(universe_name)
        return list(universe.keys())

    def get_active_symbols(self, universe_name: str) -> List[str]:
        """Get list of active symbols for a universe."""
        universe = self.get_universe(universe_name)
        return [symbol for symbol, info in universe.items() if info.active]

    def get_by_sector(self, sector: str) -> Dict[str, AssetInfo]:
        """Get all assets in a specific sector."""
        result = {}
        for universe in [self.tech_universe, self.finance_universe,
                        self.healthcare_universe, self.consumer_universe,
                        self.energy_universe]:
            for symbol, info in universe.items():
                if info.sector.lower() == sector.lower():
                    result[symbol] = info
        return result

    def get_by_market_cap(self, market_cap: str) -> Dict[str, AssetInfo]:
        """Get all assets with specific market cap."""
        result = {}
        for universe in [self.tech_universe, self.finance_universe,
                        self.healthcare_universe, self.consumer_universe,
                        self.energy_universe]:
            for symbol, info in universe.items():
                if info.market_cap.lower() == market_cap.lower():
                    result[symbol] = info
        return result

    def create_custom_universe(self, symbols: List[str], name: str = "custom") -> Dict[str, AssetInfo]:
        """Create a custom universe from symbol list."""
        result = {}
        all_assets = {**self.all_stocks, **self.crypto_universe, **self.etf_universe}

        for symbol in symbols:
            if symbol in all_assets:
                result[symbol] = all_assets[symbol]
            else:
                # Create minimal asset info for unknown symbols
                result[symbol] = AssetInfo(symbol, f"Unknown {symbol}", "Unknown", "Unknown")

        return result

    def validate_symbols(self, symbols: List[str]) -> tuple:
        """Validate symbols and return valid/invalid lists."""
        all_assets = {**self.all_stocks, **self.crypto_universe, **self.etf_universe}
        valid = [s for s in symbols if s in all_assets]
        invalid = [s for s in symbols if s not in all_assets]
        return valid, invalid

    def get_universe_info(self) -> Dict[str, Dict[str, int]]:
        """Get summary information about all universes."""
        universes = {
            "tech": self.tech_universe,
            "finance": self.finance_universe,
            "healthcare": self.healthcare_universe,
            "consumer": self.consumer_universe,
            "energy": self.energy_universe,
            "crypto": self.crypto_universe,
            "etf": self.etf_universe,
        }

        info = {}
        for name, universe in universes.items():
            info[name] = {
                "total_assets": len(universe),
                "active_assets": len([a for a in universe.values() if a.active]),
                "large_cap": len([a for a in universe.values() if a.market_cap == "Large"]),
                "medium_cap": len([a for a in universe.values() if a.market_cap == "Medium"]),
                "small_cap": len([a for a in universe.values() if a.market_cap == "Small"]),
            }

        return info
