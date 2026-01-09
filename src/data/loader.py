import os
import pandas as pd

class Loader:
    def __init__(
        self,
        source,
        cache_path,
        fetch_function=None,
        validate_function=None,
        dedupe_subset=None,
        dedupe_keep="first",
    ):
        """
        Initialize the Loader.

        Args:
            source (str): Path or URL of the primary data source.
            cache_path (str): Path to the cached data file.
            fetch_function (callable): Function to fetch data from the primary source.
            validate_function (callable): Function to validate the data.
        """
        self.source = source
        self.cache_path = cache_path
        self.fetch_function = fetch_function
        self.validate_function = validate_function
        self.data = None
        self.dedupe_subset = dedupe_subset
        self.dedupe_keep = dedupe_keep

    def load(self):
        """
        Load data from cache, fetch new data, combine them, drop duplicates, and save to cache.
        Prints the number of new rows added to the data.
        """
        cached_data = self._load_from_cache()
        new_data = self._fetch_new_data()

        if cached_data is not None and new_data is not None:
            print("Combining cached data with new data...")
            combined_data = pd.concat([cached_data, new_data])
            if self.dedupe_subset:
                combined_data = combined_data.drop_duplicates(subset=self.dedupe_subset, keep=self.dedupe_keep)
            else:
                combined_data = combined_data.drop_duplicates()
            new_rows_count = len(combined_data) - len(cached_data)
            print(f"{new_rows_count} new rows added to the cache.")
            self.data = combined_data
        elif cached_data is not None:
            self.data = cached_data
            print("No new rows added; using cached data.")
        elif new_data is not None:
            self.data = new_data
            print(f"All rows are new: {len(new_data)} rows added.")
        else:
            raise ValueError("No data available from cache or source.")

        self._save_to_cache()
        return self

    def _load_from_cache(self):
        """
        Load data from the cache if it exists.
        """
        if os.path.exists(self.cache_path):
            print("Loading data from cache...")
            return pd.read_csv(self.cache_path)
        return None

    def _fetch_new_data(self):
        """
        Fetch new data using the fetch function.
        """
        if self.fetch_function:
            print("Fetching new data from the primary source...")
            return self.fetch_function(self.source)
        return None

    def _save_to_cache(self):
        """
        Save the current data to the cache.
        """
        if self.data is not None:
            print(f"Saving combined data to cache at {self.cache_path}...")
            self.data.to_csv(self.cache_path, index=False)

    def validate(self):
        """
        Validate the data using the provided validation function.
        """
        if self.validate_function:
            print("Validating data...")
            self.validate_function(self.data)
        return self

    def get_data(self):
        """
        Return the loaded data.
        """
        return self.data
