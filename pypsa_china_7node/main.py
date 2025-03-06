#!/usr/bin/env python3
"""
PyPSA-China 7-Node Model Main Entry Point

This module serves as the entry point for the entire project,
parsing command line arguments, loading configuration, and
coordinating the workflow.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
import time

# Import project modules
from pypsa_china_7node.data import DataProcessor
from pypsa_china_7node.network import NetworkBuilder
from pypsa_china_7node.optimization import Optimizer
from pypsa_china_7node.analysis import LOLEAnalyzer
from pypsa_china_7node.visualization import ResultVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyPSA-China 7-Node Model')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file path (default: config/config.yaml)')
    
    parser.add_argument('--years', type=int, nargs='+',
                        help='Years to run (default: all years defined in config)')
    
    parser.add_argument('--output-dir', type=str,
                        help='Results output directory (default: as defined in config)')
    
    parser.add_argument('--solver', type=str,
                        help='Solver name (default: as defined in config)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from: {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def setup_directories(config):
    """Ensure required directories exist"""
    # Create results directory
    output_dir = Path(config['results']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create year subdirectories
    for year in config['model']['years']:
        year_dir = output_dir / str(year)
        year_dir.mkdir(exist_ok=True)
    
    return output_dir

def run_workflow(config, args):
    """Run the complete workflow"""
    # Determine which years to run
    years = args.years if args.years else config['model']['years']
    logger.info(f"Will run models for the following years: {years}")
    
    # Set up results directory
    if args.output_dir:
        config['results']['output_dir'] = args.output_dir
    output_dir = setup_directories(config)
    
    # Override solver if specified
    if args.solver:
        config['model']['solver']['name'] = args.solver
    
    # Create data processor
    data_processor = DataProcessor(config)
    
    # Store networks and results for each year
    networks = {}
    lole_results = {}
    
    # Run model for each year
    for year in years:
        logger.info(f"Starting model for year {year}")
        start_time = time.time()
        
        # 1. Build network
        network_builder = NetworkBuilder(config, year)
        network = network_builder.create_network(data_processor)
        
        # 2. Solve optimization
        optimizer = Optimizer(config, network, year)
        solved_network = optimizer.solve()
        
        # 3. LOLE analysis
        analyzer = LOLEAnalyzer(config, solved_network)
        lole_result = analyzer.calculate_lole()
        
        # Store results
        networks[year] = solved_network
        lole_results[year] = lole_result
        
        # 4. Save year-specific results
        if config['results']['save_networks']:
            network_file = output_dir / str(year) / f"network_{year}.nc"
            solved_network.export_to_netcdf(network_file)
            logger.info(f"Network model saved to: {network_file}")
        
        lole_file = output_dir / str(year) / f"lole_{year}.csv"
        lole_result.to_csv(lole_file)
        logger.info(f"LOLE results saved to: {lole_file}")
        
        # Calculate runtime
        elapsed_time = time.time() - start_time
        logger.info(f"Year {year} model completed in {elapsed_time:.2f} seconds")
    
    # 5. Visualize results
    if config['results']['create_plots']:
        logger.info("Generating visualization plots...")
        visualizer = ResultVisualizer(config, networks, lole_results)
        visualizer.create_all_plots()
        
        # If 2050 is included, analyze maximum power flows
        if 2050 in networks:
            visualizer.visualize_2050_flows(networks[2050])
    
    # 6. Generate report
    if config['results']['create_report']:
        logger.info("Generating analysis report...")
        visualizer.generate_report(lole_results)
    
    logger.info(f"All processing completed, results saved to {output_dir}")

def main():
    """Main function"""
    logger.info("Starting PyPSA-China 7-Node Model")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Enable debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Run workflow
        run_workflow(config, args)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Program exited normally")

if __name__ == "__main__":
    main()
