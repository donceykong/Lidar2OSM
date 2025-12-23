#!/usr/bin/env python3

# External
import os
import json
from typing import List, Dict


class L2O_Logging:
    """A simple logging class to print warnings, info, and errors with colored output."""

    # ANSI color codes
    COLORS = {
        'info': "\033[92m",     # Green
        'warn': "\033[93m",     # Yellow
        'error': "\033[91m",    # Red
    }
    RESET = "\033[0m"           # Reset color to default

    verbose = True

    def _log_decorator(prelog_msg: str, log_type: str):
        """Decorator to apply color and prefix to logging functions."""
        def apply_log_wrapper(func):
            def log_wrapper(self, message: str, include_prefix: bool = True, return_str = False, new_line = False):
                prefix = f"{log_type.upper()}: " if include_prefix else ""
                color = L2O_Logging.COLORS.get(log_type, L2O_Logging.RESET)

                log_msg = f"{color}{prelog_msg}{prefix}{message}{L2O_Logging.RESET}"
                if return_str:
                    return log_msg
                elif self.verbose:
                    print(log_msg)

            return log_wrapper
        return apply_log_wrapper

    @_log_decorator('', 'info')
    def info(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass
    @_log_decorator('\n', 'info')
    def info_nl(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass

    @_log_decorator('', 'warn')
    def warn(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass
    @_log_decorator('\n', 'warn')
    def warn_nl(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass

    @_log_decorator('', 'error')
    def error(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass
    @_log_decorator('\n', 'error')
    def error_nl(self, message: str, include_prefix: bool = True, return_str: bool = False):
        pass

# Usage:
if __name__ == "__main__":
    logger = L2O_Logging()
    logger.info("This is an informational message with prefix.")
    logger.info("This is an informational message without prefix.", include_prefix=False)
    logger.warn("This is a warning message.")
    logger.error("This is an error message.")
    logger.error_nl("This is an error message on a new line.")


class L2O_AblationStudyProcessor:
    def __init__(self, log_dir: str, verbose_output = True):
        self.log_dir = log_dir

        # Set logger and verbosity
        self.logger = L2O_Logging()
        self.logger.verbose = verbose_output

        # Load the metadata file
        self.load_metadata()

        self.ablation_args = {}

    def load_metadata(self):
        """Loads the ablation study's metadata JSON file."""
        self.metadata = []

        ablation_metadata_file = os.path.join(self.log_dir, "ablation_metadata.json")
        if not os.path.exists(ablation_metadata_file):
            err_msg = self.logger.error(f"Metadata file not found: {ablation_metadata_file}", False, True)
            raise FileNotFoundError(err_msg)
        
        with open(ablation_metadata_file, 'r') as file:
            self.metadata = json.load(file)

        self.logger.info_nl(f"\n-------- Loaded metadata for {len(self.metadata)} tests. --------\n", False, return_str=False)

    def initialize_ablation_args(self):
        """Initialize the ablation_args dictionary with keys from metadata and set each to an empty list."""
        # Ensure the ablation's metadata is initialized
        if not self.metadata:
            self.load_metadata()

        first_entry = self.metadata[0]  # Use the first entry in metadata to extract the keys
        self.ablation_args = {key: [] for key in first_entry}   # Initialize all keys with empty lists
        # print(f"\nInitialized ablation_args with keys: {list(self.ablation_args.keys())}")

    def set_ablation_args(self, args: Dict[str, List]):
        """Sets specific ablation arguments dynamically, leaving unspecified ones as empty lists."""
        # Ensure the ablation argument keys are initialized
        if not self.ablation_args:
            self.initialize_ablation_args()

        # Update only the keys provided in the passed dictionary; others stay as empty lists
        for key, value in args.items():
            if key in self.ablation_args:
                self.ablation_args[key] = value
            else:
                self.logger.warn(f"'{key}' is not a valid key in the metadata file!")

        # Print the user's defined ablation_args
        self.logger.info("Ablation Arguments set to:")
        for key, value in self.ablation_args.items():
            self.logger.info(f"    - '{key}': {value if value else 'ANY'}", False)

    def filter_metadata(self) -> List[Dict]:
        """Filters metadata based on the dynamically generated ablation arguments."""
        filtered_metadata = [
            item for item in self.metadata
            if all(
                not self.ablation_args[key] or item[key] in self.ablation_args[key]
                for key in self.ablation_args if key in item
            )
        ]
        self.logger.info_nl(f"Filtered to {len(filtered_metadata)} tests based on criteria.")

        return filtered_metadata

    def process_test(self, test_metadata: Dict):
        """Processes an individual test case and adds trial results from ablation_results.json."""
        test_id = test_metadata['test_id']
        test_trial_count = int(test_metadata['trial_count'])

        result_file_path = os.path.join(self.log_dir, f"test_{test_id:05d}", "ablation_results.json") # Define the path to the ablation_results.json file for this test
        test_data_dict = {} # Initialize a dictionary to store the test's resulting data

        # Check if the result file exists
        if os.path.exists(result_file_path):
            # Load the results from the ablation_results.json file
            with open(result_file_path, 'r') as result_file:
                results = json.load(result_file)

            # Ensure that the number of trials matches what is expected in the metadata
            if len(results) != test_trial_count:
                self.logger.error_nl(f"Expected {test_trial_count} trials in Test_{test_id:05d} directory, but found {len(results)} in ablation_results.json")
            
            # Store each trial's data in the test_data_dict using the trial_id as the key
            for trial in results:
                trial_id = trial['trial_id']
                test_data_dict[trial_id] = trial

            # Add the trial data dictionary to the test_metadata
            test_metadata['results'] = test_data_dict
        else:
            self.logger.error_nl(f"Results file not found for Test ID: {test_id} at {result_file_path}", include_prefix=False)

        return test_metadata

    def organize_test_data(self, test_metadata):
        self.KEY_IGNORE_LIST = ["ground_truth_tf", "estimated_tf", 'trial_id']
        test_results = {}
        test_results['test_id'] = test_metadata.get('test_id', 'unknown')
        
        # Ensure 'results' exist in test_metadata
        if 'results' in test_metadata:
            results = test_metadata['results']
            
            # Initialize empty lists for each key in the results (excluding ignored ones)
            sample_trial = next(iter(results.values()), {})
            for test_key in sample_trial:
                if test_key not in self.KEY_IGNORE_LIST:
                    test_results[f'{test_key}'] = []

            # Populate results for each trial
            for trial_results in results.values():
                for test_key, value in trial_results.items():
                    keyname = f'{test_key}'
                    if keyname in test_results:
                        test_results[keyname].append(value)
                        
        return test_results

    def get_comparison_dict(self, test_metadata, independent_var, dependent_var):
        """Extract comparison data for independent and dependent variables."""
        self.KEY_KEEP_LIST = [independent_var, dependent_var]
        comparison_results = {}
        
        # Ensure 'results' exist in test_metadata
        if 'results' in test_metadata:
            results = test_metadata['results']
            
            # Initialize empty lists for each key in the results (keeping only relevant ones)
            sample_trial = next(iter(results.values()), {})
            for test_key in sample_trial:
                if test_key in self.KEY_KEEP_LIST:
                    comparison_results[f'{test_key}'] = []

            # Populate results for each trial
            for trial_results in results.values():
                for test_key, value in trial_results.items():
                    if test_key in comparison_results:
                        comparison_results[test_key].append(value)

        return comparison_results

    # def process_filtered_tests(self, filtered_metadata: List[Dict]):
    #     """Processes all filtered test cases."""
    #     # Initialize test_results_list if needed
    #     organized_test_results_all = {}

    #     for test_metadata in filtered_metadata:
    #         # Process the individual test and get results dict ordered by trial_id
    #         test_result = self.process_test(test_metadata)

    #         # Check if test_result is None
    #         if test_result is None:
    #             print(f"Error: process_test returned None for test_metadata {test_metadata}")
    #             continue

    #         # Order by results and append all trials into lists
    #         organized_test_result = self.organize_test_data(test_result)

    #         # Check if organize_test_data is None
    #         if organized_test_result is None:
    #             print(f"Error: organize_test_data returned None for test_result {test_result}")
    #             continue

    #         # Merge test_results_list with test_results_list_new
    #         for key, value in organized_test_result.items():
    #             if key != "test_id":
    #                 if key in organized_test_results_all:
    #                     organized_test_results_all[key].extend(value)
    #                 else:
    #                     organized_test_results_all[key] = value

    #     return organized_test_results_all

    # def find_and_process_files(self):
    #     """Filters and processes the metadata based on the ablation arguments."""
    #     filtered_metadata = self.filter_metadata()
    #     return self.process_filtered_tests(filtered_metadata)
    
    def find_and_process_files(self, max_val_key_values: Dict[str, float] = None, min_val_key_values: Dict[str, float] = None):
        """
        Filters and processes the metadata based on the ablation arguments and optional min/max value dictionaries.

        Args:
            max_val_key_values (dict, optional): A dictionary where keys are the test result keys to filter by,
                                                and values are the max allowed values for those keys.
            min_val_key_values (dict, optional): A dictionary where keys are the test result keys to filter by,
                                                and values are the min allowed values for those keys.

        Returns:
            dict: Organized test results filtered by the specified min/max values.
        """
        # Filter the metadata based on ablation arguments
        filtered_metadata = self.filter_metadata()

        # Process the filtered metadata and apply the min/max value filtering
        return self.process_filtered_tests(filtered_metadata, max_val_key_values=max_val_key_values, min_val_key_values=min_val_key_values)


    def process_filtered_tests(self, filtered_metadata: List[Dict], max_val_key_values: Dict[str, float] = None, min_val_key_values: Dict[str, float] = None):
        """
        Processes all filtered test cases and filters based on multiple min/max values for specified keys.

        Args:
            filtered_metadata (List[Dict]): List of filtered metadata for the tests.
            max_val_key_values (dict, optional): A dictionary where keys are the test result keys to filter by,
                                                and values are the max allowed values for those keys.
            min_val_key_values (dict, optional): A dictionary where keys are the test result keys to filter by,
                                                and values are the min allowed values for those keys.

        Returns:
            dict: Organized test results filtered by the specified min/max values.
        """
        # Initialize test_results_list if needed
        organized_test_results_all = {}

        for test_metadata in filtered_metadata:
            # Process the individual test and get results dict ordered by trial_id
            test_result = self.process_test(test_metadata)

            # Check if test_result is None
            if test_result is None:
                print(f"Error: process_test returned None for test_metadata {test_metadata}")
                continue

            # Order by results and append all trials into lists
            organized_test_result = self.organize_test_data(test_result)

            # Check if organize_test_data is None
            if organized_test_result is None:
                print(f"Error: organize_test_data returned None for test_result {test_result}")
                continue

            # If max_val_key_values or min_val_key_values are provided, apply value filtering
            if max_val_key_values or min_val_key_values:
                # Find the indices where values exceed the max or are below the min for each key
                invalid_indices = set()
                
                # Max value filtering
                if max_val_key_values:
                    for max_key, max_val in max_val_key_values.items():
                        if max_key in organized_test_result:
                            invalid_indices.update(
                                idx for idx, val in enumerate(organized_test_result[max_key]) if val > max_val
                            )
                
                # Min value filtering
                if min_val_key_values:
                    for min_key, min_val in min_val_key_values.items():
                        if min_key in organized_test_result:
                            invalid_indices.update(
                                idx for idx, val in enumerate(organized_test_result[min_key]) if val < min_val
                            )

                # Remove values at invalid indices for all keys in the result
                for key in organized_test_result:
                    if key != "test_id":
                        organized_test_result[key] = [
                            val for idx, val in enumerate(organized_test_result[key]) if idx not in invalid_indices
                        ]

            # Merge organized_test_result into organized_test_results_all
            for key, value in organized_test_result.items():
                if key != "test_id":
                    if key in organized_test_results_all:
                        organized_test_results_all[key].extend(value)
                    else:
                        organized_test_results_all[key] = value

        return organized_test_results_all