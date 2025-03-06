"""
Network Building Module

This module is responsible for creating and configuring PyPSA network objects,
including adding nodes, links, loads, and generators.
"""

import logging
import pandas as pd
import pypsa
from pathlib import Path

logger = logging.getLogger(__name__)

class NetworkBuilder:
    """Network builder class for creating PyPSA network models"""
    
    def __init__(self, config, year):
        """
        Initialize the network builder
        
        Parameters:
            config: Configuration dictionary
            year: Target year
        """
        self.config = config
        self.year = year
        self.regions = config['network']['regions']
        self.links = config['network']['links']
        self.ens_cost = config['optimization']['ens_cost']
        self.transmission_hurdle_cost = config['optimization']['transmission_hurdle_cost']
        
        logger.info(f"Initializing network builder for year {year}")

    def create_network(self, data_processor):
        """
        Create network model
        
        Parameters:
            data_processor: Data processor object for loading data
            
        Returns:
            Configured PyPSA network object
        """
        logger.info(f"Creating network model for year {self.year}")
        
        # Create network object
        network = pypsa.Network()
        
        # Set time index
        timestamps = data_processor.get_timestamps()
        network.set_snapshots(timestamps)
        
        # Add nodes, links and components
        self._add_buses(network)
        self._add_transmission_links(network, data_processor)
        self._add_loads(network, data_processor)
        self._add_generators(network, data_processor)
        self._add_ens_generators(network)
        
        logger.info(f"Network model for year {self.year} created successfully")
        return network
    
    def _add_buses(self, network):
        """Add region nodes"""
        for region in self.regions:
            network.add("Bus", region, carrier="AC")
        
        logger.debug(f"Added {len(self.regions)} region nodes")
    
    def _add_transmission_links(self, network, data_processor):
        """Add inter-regional transmission links"""
        for region1, region2 in self.links:
            # Get transmission capacity data
            if self.year == 2020:
                capacity = data_processor.get_transmission_capacity(region1, region2, self.year)
                extendable = False
                p_nom_max = None
            elif self.year == 2030:
                capacity = data_processor.get_transmission_capacity(region1, region2, 2020)
                max_capacity = data_processor.get_transmission_capacity(region1, region2, self.year)
                extendable = True
                p_nom_max = max_capacity
            else:  # 2050
                capacity = data_processor.get_transmission_capacity(region1, region2, 2020)
                extendable = True
                p_nom_max = float("inf")  # No capacity limit
            
            # Add forward link
            link_name = f"{region1}-{region2}"
            network.add("Link",
                       link_name,
                       bus0=region1,
                       bus1=region2,
                       p_nom=capacity,
                       p_min_pu=-1,  # Allow bidirectional transmission
                       p_max_pu=1,
                       marginal_cost=self.transmission_hurdle_cost,
                       p_nom_extendable=extendable)
            
            # Set capacity limit (if applicable)
            if extendable and p_nom_max is not None:
                network.links.at[link_name, "p_nom_max"] = p_nom_max
            
            # Add reverse link
            reverse_link_name = f"{region2}-{region1}"
            network.add("Link",
                       reverse_link_name,
                       bus0=region2,
                       bus1=region1,
                       p_nom=capacity,
                       p_min_pu=-1,
                       p_max_pu=1,
                       marginal_cost=self.transmission_hurdle_cost,
                       p_nom_extendable=extendable)
            
            if extendable and p_nom_max is not None:
                network.links.at[reverse_link_name, "p_nom_max"] = p_nom_max
        
        logger.debug(f"Added {len(self.links)*2} transmission links (bidirectional)")
    
    def _add_loads(self, network, data_processor):
        """Add regional loads"""
        load_data = data_processor.get_demand_data()
        scale_factors = data_processor.get_demand_scale_factors(self.year)
        
        for region in self.regions:
            # Scale load according to target year
            if region in load_data.columns and region in scale_factors:
                region_load = load_data[region] * scale_factors[region]
                
                # Add load object
                network.add("Load", 
                           f"{region}-load",
                           bus=region,
                           p_set=region_load)
            else:
                logger.warning(f"Region {region} is missing load data or scaling factor")
        
        logger.debug(f"Added {len(self.regions)} regional loads")
    
    def _add_generators(self, network, data_processor):
        """Add generators"""
        generators_data = data_processor.get_generators_data(self.year)
        min_output_params = self.config['optimization']['min_technical_output']
        marginal_costs = self.config['optimization']['marginal_costs']
        
        for generator in generators_data:
            region = generator["region"]
            tech_type = generator["type"]
            capacity = generator["capacity"]
            
            # Set generator parameters
            p_min_pu = min_output_params.get(tech_type, 0)
            marginal_cost = marginal_costs.get(tech_type, 100)
            
            # Add generator
            gen_name = f"{region}-{tech_type}"
            network.add("Generator",
                       gen_name,
                       bus=region,
                       carrier=tech_type,
                       p_nom=capacity,
                       marginal_cost=marginal_cost,
                       p_min_pu=p_min_pu,
                       p_max_pu=1.0)
            
            # For renewable energy, add availability profile
            if tech_type in ["wind", "solar", "hydro"]:
                availability = data_processor.get_renewable_profile(region, tech_type)
                if availability is not None:
                    network.generators_t.p_max_pu[gen_name] = availability
        
        logger.debug(f"Added {len(generators_data)} generators")
    
    def _add_ens_generators(self, network):
        """Add ENS virtual generators (energy not served)"""
        for region in self.regions:
            network.add("Generator",
                       f"{region}-ENS",
                       bus=region,
                       carrier="ENS",
                       p_nom=float("inf"),  # Infinite capacity
                       marginal_cost=self.ens_cost,  # Very high cost, ensure used as last resort
                       p_min_pu=0,
                       p_max_pu=1.0)
        
        logger.debug(f"Added {len(self.regions)} ENS virtual generators")
