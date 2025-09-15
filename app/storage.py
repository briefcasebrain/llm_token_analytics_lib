"""
Results storage and cache management.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import tempfile
import pandas as pd


class ResultsStorage:
    """Manages storage and retrieval of simulation and collection results."""

    def __init__(self):
        """Initialize the storage system."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def store_collection_result(self, collected_data: pd.DataFrame, stats: Dict) -> str:
        """
        Store collection results and return a collection ID.

        Args:
            collected_data: The collected DataFrame
            stats: Summary statistics

        Returns:
            Collection ID for later retrieval
        """
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.parquet',
            delete=False
        )
        collected_data.to_parquet(temp_file.name)

        # Generate ID and store reference
        collection_id = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._cache[collection_id] = {
            'file': temp_file.name,
            'stats': stats,
            'timestamp': datetime.now().isoformat(),
            'type': 'collection'
        }

        return collection_id

    def store_simulation_result(self, results: Any, config: Any) -> str:
        """
        Store simulation results and return a simulation ID.

        Args:
            results: Simulation results object
            config: Simulation configuration

        Returns:
            Simulation ID for later retrieval
        """
        simulation_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._cache[simulation_id] = {
            'results': results,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'type': 'simulation'
        }

        return simulation_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored result by ID.

        Args:
            result_id: The ID of the result to retrieve

        Returns:
            Stored result data or None if not found
        """
        return self._cache.get(result_id)

    def get_collection_data(self, collection_id: str) -> Optional[pd.DataFrame]:
        """
        Load collection data from storage.

        Args:
            collection_id: The collection ID

        Returns:
            DataFrame or None if not found
        """
        collection = self.get_result(collection_id)
        if collection and 'file' in collection:
            return pd.read_parquet(collection['file'])
        return None

    def result_exists(self, result_id: str) -> bool:
        """Check if a result exists in storage."""
        return result_id in self._cache

    def list_results(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all stored results."""
        return {
            result_id: {
                'type': result['type'],
                'timestamp': result['timestamp']
            }
            for result_id, result in self._cache.items()
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
