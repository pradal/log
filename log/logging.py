import os
import sys
import time
import pickle
import timeit
import xarray as xr
import pandas as pd
import numpy as np
from dataclasses import fields
import pyvista as pv
import matplotlib.pyplot as plt
#from gudhi import bottleneck_distance

import openalea.plantgl.all as pgl
from openalea.mtg.traversal import pre_order, post_order
from log.visualize import plot_mtg, plot_mtg_alt


class Logger:
    def __init__(self, model_instance, outputs_dirpath="", 
                 output_variables={}, scenario={"default":1}, time_step_in_hours=1, 
                 logging_period_in_hours=1, 
                 recording_sums=True, recording_raw=True, recording_mtg=True, recording_images=True, recording_performance=True,
                 plotted_property="hexose_exudation",
                 echo=True):
        self.g = model_instance.g
        self.props = self.g.properties()
        
        self.models = model_instance.models
        self.outputs_dirpath = outputs_dirpath
        self.output_variables = output_variables
        self.scenario = scenario
        self.summable_output_variables = []
        self.meanable_output_variables = []
        self.time_step_in_hours = time_step_in_hours
        self.logging_period_in_hours = logging_period_in_hours
        self.recording_sums = recording_sums
        self.recording_raw = recording_raw
        self.recording_mtg = recording_mtg
        self.recording_images = recording_images
        self.plotted_property = plotted_property
        self.recording_performance = recording_performance
        self.echo = echo
        # TODO : add a scenario named folder
        self.root_images_dirpath = os.path.join(self.outputs_dirpath, "root_images")
        self.MTG_files_dirpath = os.path.join(self.outputs_dirpath, "MTG_files")
        self.MTG_properties_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties")
        self.MTG_properties_summed_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_summed")
        self.MTG_properties_raw_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_raw")
        self.create_or_empty_directory(self.outputs_dirpath)
        self.create_or_empty_directory(self.root_images_dirpath)
        self.create_or_empty_directory(self.MTG_files_dirpath)
        self.create_or_empty_directory(self.MTG_properties_dirpath)
        self.create_or_empty_directory(self.MTG_properties_summed_dirpath)
        self.create_or_empty_directory(self.MTG_properties_raw_dirpath)

        if self.output_variables == {}:
            for model in self.models:
                self.summable_output_variables += model.extensive_variables
                self.meanable_output_variables += model.intensive_variables
                available_inputs = [i for i in model.inputs if i in self.props.keys()] # To prevent getting inputs that are not proveided neither from another model nor mtg
                self.output_variables.update({f.name:f.metadata for f in fields(model) if f.name in model.state_variables + available_inputs})

        if self.recording_sums:
            self.plant_scale_properties = pd.DataFrame(columns=self.summable_output_variables + self.meanable_output_variables)

        if self.recording_raw:
            self.log_xarray = []
        
        if self.recording_performance:
            self.simulation_performance = pd.DataFrame(columns=["time_step_duration"])

        if recording_images:
            self.plotter = pv.Plotter(off_screen=not self.echo, window_size=[1900, 1080], lighting="three lights")
            self.plotter.set_background("brown")
            self.plotter.camera_position = [(0.2771029327675725, 0.26559962985945457, 0.22821718563975485),
                                            (0.0475656802374161, 0.03795701550406494, -0.09353044576347432),
                                            (-0.5353537107754003, -0.4607303882419462, 0.7079010620909072)]
            framerate = 30
            self.plotter.open_movie(os.path.join(self.root_images_dirpath, "root_movie.mp4"), framerate)
            self.plotter.show(interactive_update=True)
            plot_mtg(self.g, prop_cmap=self.plotted_property)
            root_system_mesh = plot_mtg_alt(self.g, cmap_property=self.plotted_property)
            self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap="jet", show_edges=False)
            self.plot_text = self.plotter.add_text(f" t = 0 h", position="upper_left")

        self.start_time = timeit.default_timer()
        self.previous_step_start_time = self.start_time
        self.simulation_time_in_hours = 0
        
    def create_or_empty_directory(self, directory=""):
        if not os.path.exists(directory):
        # We create it:
            os.makedirs(directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    @property
    def elapsed_time(self):
        return timeit.default_timer() - self.start_time
    
    def __call__(self):
        self.current_step_start_time = self.elapsed_time
        
        if self.echo:
            print(f"[RUNNING] {self.simulation_time_in_hours} hours | step took {round(self.current_step_start_time - self.previous_step_start_time, 1)} s | {int(self.elapsed_time)} s of simulation until now", end='\r', flush=True)

        if self.recording_performance:
            self.recording_step_performance()
        
        if self.simulation_time_in_hours % self.logging_period_in_hours == 0:
            if self.recording_sums:
                self.recording_summed_MTG_properties_to_csv()
            if self.recording_raw:
                self.recording_raw_MTG_properties_in_xarray()
            if self.recording_mtg:
                self.recording_mtg_files()
            if self.recording_images:
                self.recording_images_from_plantgl()
        
        self.simulation_time_in_hours += self.time_step_in_hours
        self.previous_step_start_time = self.current_step_start_time
    
    def recording_step_performance(self):
        step_elapsed = pd.DataFrame({"time_step_duration":self.current_step_start_time - self.previous_step_start_time}, columns=["time_step_duration"], 
                                    index=[self.simulation_time_in_hours])
        self.simulation_performance = pd.concat([self.simulation_performance, step_elapsed])

    def recording_summed_MTG_properties_to_csv(self):
        step_plant_scale = {}
        for var in self.summable_output_variables:
            step_plant_scale.update({var:sum(self.props[var].values())})
        for var in self.meanable_output_variables:
            emerged_only_list = [v for v in self.props[var].values() if v > 0]
            if len(emerged_only_list) > 0:
                step_plant_scale.update({var:np.mean(emerged_only_list)})
            else:
                step_plant_scale.update({var:None})
        step_sum = pd.DataFrame(step_plant_scale, columns=self.summable_output_variables + self.meanable_output_variables, 
                                index=[self.simulation_time_in_hours])
        self.plant_scale_properties = pd.concat([self.plant_scale_properties, step_sum])

    def recording_raw_MTG_properties_in_xarray(self):
        self.log_xarray += [self.mtg_to_dataset(variables=self.output_variables, time=self.simulation_time_in_hours)]
        if sys.getsizeof(self.log_xarray) > 10000:
            print("")
            print("[INFO] Merging stored properties data in one xarray dataset...", flush=True)
            self.write_to_disk(self.log_xarray)
            # Check save maybe
            self.log_xarray = []

    def mtg_to_dataset(self, variables, 
                       coordinates=dict(vid=dict(unit="adim", value_example=1, description="Root segment identifier index"),
                                        t=dict(unit="h", value_example=1, description="Model time step")),
                       description="Model local root MTG properties over time", 
                       time=0):
        # convert dict to dataframe with index corresponding to coordinates in topology space
        # (not just x, y, z, t thanks to MTG structure)
        props_dict = {k:v for k, v in self.props.items() if type(v)==dict}
        props_df = pd.DataFrame.from_dict(props_dict)
        props_df["vid"] = props_df.index
        props_df["t"] = [time for k in range(props_df.shape[0])]
        props_df = props_df.set_index(list(coordinates.keys()))

        # Select properties actually used in the current version of the target model
        props_df = props_df[list(variables.keys())]

        # Filter duplicated indexes
        props_df = props_df[~props_df.index.duplicated()]

        # Remove false root segments created just for branching regularity issues (vid 0, 2, 4, etc)
        props_df = props_df[props_df["struct_mass"] > 0]

        # Convert to xarray with given dimensions to spatialize selected properties
        props_ds = props_df.to_xarray()

        # Dataset global attributes
        props_ds.attrs["description"] = description
        # Dataset coordinates' attribute metadata
        for k, v in coordinates.items():
            getattr(props_ds, k).attrs.update(v)

        # Dataset variables' attribute metadata
        for k, v in variables.items():
            getattr(props_ds, k).attrs.update(v)

        return props_ds
    
    def recording_mtg_files(self):
        with open(os.path.join(self.MTG_files_dirpath, f'root_{self.simulation_time_in_hours}.pckl'), "wb") as f:
            pickle.dump(self.g, f)


    def recording_images_from_plantgl(self):
        # TODO : step back according to max(||x2-x1||, ||y2-y1||, ||z2-z1||)
        #Updates positions with turtle
        plot_mtg(self.g, prop_cmap=self.plotted_property)
        root_system_mesh = plot_mtg_alt(self.g, cmap_property=self.plotted_property)

        self.plotter.remove_actor(self.current_mesh)
        self.plotter.remove_actor(self.plot_text)
        self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap="jet", show_edges=False, specular=1.)
        self.plot_text = self.plotter.add_text(f" t = {self.simulation_time_in_hours} h", position="upper_left")
        self.plotter.update()
        self.plotter.write_frame()
        # Usefull to set new camera angle
        #print(self.plotter.camera_position)

        #pgl.Viewer.display()
        # If needed, we wait for a few seconds so that the graph is well positioned:
        #time.sleep(0.1)
        #image_name = os.path.join(self.root_images_dirpath, f'root_{self.simulation_time_in_hours}.png')
        #pgl.Viewer.saveSnapshot(image_name)

    def write_to_disk(self, xarray_list):
        interstitial_dataset = xr.concat(xarray_list, dim="t")
        interstitial_dataset.to_netcdf(os.path.join(self.MTG_properties_raw_dirpath, f't={self.simulation_time_in_hours}.nc'))
    
    def mtg_persistent_homology(self, g):
        props = g.properties()
        root_gen = g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order(g, root):
            if vid == 1:
                g.node(vid).dist_to_collar = 0
                g.node(vid).order = 1
            else:
                parent = g.parent(vid)
                g.node(vid).dist_to_collar = g.node(parent).dist_to_collar + g.node(parent).length
                if self.props["edge_type"][vid] == "+":
                    g.node(vid).order = g.node(parent).order + 1
                else:
                    g.node(vid).order = g.node(parent).order
        
        prop = "order"

        geodesic_sorting = sorted(props["dist_to_collar"], key=props["dist_to_collar"].get, reverse=True)

        captured_vertices = []
        homology_barcode = []
        colored_prop = []
        for vid in geodesic_sorting[1:]:
            captured = False
            if len(captured_vertices) > 0:
                for axis in captured_vertices:
                    if vid in axis:
                        captured = True
            if not captured:
                new_group = g.Ancestors(vid, RestrictedTo="SameAxis")
                if len(new_group) > 1:
                    captured_vertices += [new_group]
                    homology_barcode += [[props["dist_to_collar"][v] for v in new_group]]
                    colored_prop += [plt.cm.cool(np.mean([props[prop][v] for v in new_group])/5)]

        persitent_diagram = np.array([[min(axs), max(axs)] for axs in homology_barcode])

        fig, ax = plt.subplots(2)
        
        for k in range(len(homology_barcode)):
            line = [-k for i in range(len(homology_barcode[k]))]
            ax[0].plot(homology_barcode[k], line, c=colored_prop[k], linewidth=2)
        
        ax[1].scatter(persitent_diagram[:,0], persitent_diagram[:,1], c=colored_prop)

        # TODO move out
        #print(bottleneck_distance(persitent_diagram, persitent_diagram, 0.))

        plt.show()

        return persitent_diagram


    def stop(self):
        if self.echo:
            elapsed_at_simulation_end = self.elapsed_time
            print("") # to receive the flush
            print(f"[INFO] Simulation ended after {round(elapsed_at_simulation_end/60, 1)} min without error")
            print("[INFO] Now proceeding to data writing on disk...")

        if self.recording_sums:
            # Saving in memory summed properties
            self.plant_scale_properties.to_csv(os.path.join(self.MTG_properties_summed_dirpath, "plant_scale_properties.csv"))

        if self.recording_raw:
            # For saved xarray datasets
            if len(self.log_xarray) > 0:
                print("[INFO] Merging stored properties data in one xarray dataset...")
                self.write_to_disk(self.log_xarray)
                del self.log_xarray
            
            time_step_files = [os.path.join(self.MTG_properties_raw_dirpath, name) for name in os.listdir(self.MTG_properties_raw_dirpath)]
            time_dataset = xr.open_mfdataset(time_step_files)
            time_dataset = time_dataset.assign_coords(coords=self.scenario).expand_dims(dim=dict(zip(list(self.scenario.keys()), [1 for k in self.scenario])))
            time_dataset.to_netcdf(self.MTG_properties_raw_dirpath + '/merged.nc')
            del time_dataset
            for file in os.listdir(self.MTG_properties_raw_dirpath):
                if '.nc' in file and file != "merged.nc":
                    os.remove(self.MTG_properties_raw_dirpath + '/' + file)
        
        if self.recording_performance:
            self.simulation_performance.to_csv(os.path.join(self.outputs_dirpath, "simulation_performance.csv"))

        if self.echo:
            time_writing_on_disk = self.elapsed_time - elapsed_at_simulation_end
            print(f"[INFO] Successfully wrote data on disk after {round(time_writing_on_disk/60, 1)} minutes")
            print("[LOGGER CLOSES]")

        # self.mtg_persistent_homology(g=self.g)

def test_logger():
    return Logger()


