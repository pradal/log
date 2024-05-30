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
import inspect
import logging
from gudhi import bottleneck_distance

from openalea.mtg.traversal import pre_order, post_order
from openalea.mtg import turtle as turt
from log.visualize import plot_mtg, plot_mtg_alt, soil_voxels_mesh, shoot_plantgl_to_mesh


class Logger:

    light_log = dict(recording_images=False, recording_off_screen=True, plotted_property="import_Nm", show_soil=True,
                    recording_mtg=False,
                    recording_raw=False,
                    recording_sums=True,
                    recording_performance=False,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=False,
                    on_performance=False,
                    animate_raw_logs=False)
    
    medium_log_focus_images = dict(recording_images=True, recording_off_screen=True, plotted_property="import_Nm", show_soil=True,
                    recording_mtg=False,
                    recording_raw=False,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=False)
    
    medium_log_focus_properties = dict(recording_images=False, recording_off_screen=True, plotted_property="nitrate_transporters_affinity_factor", flow_property=False, show_soil=False,
                    recording_mtg=False,
                    recording_raw=True,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True)
    
    heavy_log = dict(recording_images=True, recording_off_screen=False, plotted_property="import_Nm", flow_property=True, show_soil=False,
                    recording_mtg=False,
                    recording_raw=True,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=True, compare_to_ref_barcode=False, 
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True)

    def __init__(self, model_instance, outputs_dirpath="",
                 output_variables={}, scenario={"default": 1}, time_step_in_hours=1,
                 logging_period_in_hours=1,
                 recording_sums=False, recording_raw=False, recording_mtg=False, recording_images=False, recording_off_screen=False,
                 recording_performance=False,
                 recording_shoot=False,
                 plotted_property="hexose_exudation", flow_property=False, show_soil=False,
                 recording_barcodes=False, compare_to_ref_barcode=False, barcodes_path="inputs/persistent_barcodes.pckl",
                 echo=True, **kwargs):

        # First Handle exceptions
        self.exceptions = []
        self.model_instance = model_instance
        self.data_structures = model_instance.data_structures
        self.props = {}
        for name, data_structure in self.data_structures.items():
            # If we have to extract properties from a mtg instance
            if str(type(data_structure)) == "<class 'openalea.mtg.mtg.MTG'>":
                if name == "root":
                    self.props[name] = data_structure.properties()
                elif name == "shoot":
                    self.props[name] = data_structure.get_vertex_property(2)["roots"]
                else:
                    print("ERROR, unknown MTG")
            # Elif a dict of properties have already been provided
            elif str(type(data_structure)) == "<class 'dict'>":
                self.props[name] = data_structure
            else:
                print("[WARNING] Unknown data structure has been passed to logger")

        self.models = model_instance.models
        self.outputs_dirpath = outputs_dirpath
        self.output_variables = output_variables
        self.scenario = scenario
        self.summable_output_variables = []
        self.meanable_output_variables = []
        self.plant_scale_state = []
        self.units_for_outputs = {}
        self.time_step_in_hours = time_step_in_hours
        self.logging_period_in_hours = logging_period_in_hours
        self.recording_sums = recording_sums
        self.recording_raw = recording_raw
        self.recording_mtg = recording_mtg
        self.recording_shoot = recording_shoot
        if self.recording_shoot:
            self.shoot = model_instance.shoot
        if "root" not in self.data_structures.keys():
            recording_images = False
        self.recording_images = recording_images
        self.recording_off_screen = recording_off_screen
        self.show_soil = show_soil
        self.plotted_property = plotted_property
        self.flow_property = flow_property
        self.recording_performance = recording_performance
        self.recording_barcodes = recording_barcodes
        self.compare_to_ref_barcode = compare_to_ref_barcode
        
        if self.compare_to_ref_barcode:
            self.recording_barcodes = True
            with open(barcodes_path, "rb") as f:
                self.ref_persitent_barcodes = pickle.load(f)

        if self.recording_barcodes:
            self.persistent_barcodes = {}
                
        self.echo = echo
        # TODO : add a scenario named folder
        self.root_images_dirpath = os.path.join(self.outputs_dirpath, "root_images")
        self.MTG_files_dirpath = os.path.join(self.outputs_dirpath, "MTG_files")
        self.MTG_barcodes_dirpath = os.path.join(self.outputs_dirpath, "MTG_barcodes")
        self.MTG_properties_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties")
        self.MTG_properties_summed_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_summed")
        self.MTG_properties_raw_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_raw")
        self.shoot_properties_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/shoot_properties")
        self.create_or_empty_directory(self.outputs_dirpath)
        self.create_or_empty_directory(self.root_images_dirpath)
        self.create_or_empty_directory(self.MTG_files_dirpath)
        self.create_or_empty_directory(self.MTG_properties_dirpath)
        self.create_or_empty_directory(self.MTG_properties_summed_dirpath)
        if self.recording_raw:
            self.create_or_empty_directory(self.MTG_properties_raw_dirpath)
        if self.recording_barcodes:
            self.create_or_empty_directory(self.MTG_barcodes_dirpath)
        if self.recording_shoot:
            self.create_or_empty_directory(self.shoot_properties_dirpath)

        if self.output_variables == {}:
            for model in self.models:
                self.summable_output_variables += model.extensive_variables
                self.meanable_output_variables += model.intensive_variables
                self.plant_scale_state += model.plant_scale_state
                available_inputs = [i for i in model.inputs if
                                    i in self.props.keys()]  # To prevent getting inputs that are not provided neither from another model nor mtg
                self.output_variables.update(
                    {f.name: f.metadata for f in fields(model) if f.name in model.state_variables + available_inputs})
                self.units_for_outputs.update({f.name: f.metadata["unit"] for f in fields(model) if
                                               f.name in self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state})

        if self.recording_sums:
            self.plant_scale_properties = pd.DataFrame(
                columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state)
            unit_row = pd.DataFrame(self.units_for_outputs,
                                    columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state,
                                    index=["unit"])
            self.plant_scale_properties = pd.concat([self.plant_scale_properties, unit_row])

        if self.recording_raw:
            self.log_xarray = []

        if self.recording_performance:
            self.simulation_performance = pd.DataFrame()

        if recording_images:
            self.log_mtg_coordinates()
            self.init_images_plotter()

        logging.basicConfig(filename=os.path.join(outputs_dirpath, '[RUNNING] simulation.log'), filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        self.start_time = timeit.default_timer()
        self.previous_step_start_time = self.start_time
        self.simulation_time_in_hours = 0


    def init_images_plotter(self):
        self.prop_mins = [None for k in range(9)] + [min(self.props["root"][self.plotted_property].values())]
        self.prop_maxs = [None for k in range(9)] + [max(self.props["root"][self.plotted_property].values())]
        self.all_times_low, self.all_times_high = self.prop_mins[-1], self.prop_mins[-1]
        if self.all_times_low == 0:
            self.all_times_low = self.all_times_high / 1000

        sizes = {"landscape": [1920, 1080], "portrait": [1088, 1920], "square": [1080, 1080],
                    "small_height": [960, 1280]}
        
        if self.recording_off_screen:
            pv.start_xvfb()

        self.plotter = pv.Plotter(off_screen=not self.echo, window_size=sizes["portrait"], lighting="three lights")
        self.plotter.set_background("brown")

        step_back_coefficient = 0.5
        camera_coordinates = (step_back_coefficient, 0., 0.)
        move_up_coefficient = 0.1
        horizontal_aiming = (0., 0., 1.)
        collar_position = (0., 0., -move_up_coefficient)
        self.plotter.camera_position = [camera_coordinates,
                                        collar_position,
                                        horizontal_aiming]

        framerate = 10
        self.plotter.open_movie(os.path.join(self.root_images_dirpath, "root_movie.mp4"), framerate)
        self.plotter.show(interactive_update=True)

        # NOTE : Not necessary since voxels provide the scale information :
        # First plot a 1 cm scale bar
        # self.plotter.add_mesh(pv.Line((0, 0.08, 0), (0, 0.09, 0)), color='k', line_width=7)
        # self.plotter.add_text("1 cm", position="upper_right")

        # Then add initial states of plotted compartments
        root_system_mesh, color_property = plot_mtg_alt(self.data_structures["root"], cmap_property=self.plotted_property)
        self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap="jet", clim=[1e-10, 6e-9], show_edges=False, log_scale=True)
        self.plot_text = self.plotter.add_text(f"Simulation starting...", position="upper_left")
        if "soil" in self.data_structures.keys() and self.show_soil:
            soil_grid = soil_voxels_mesh(self.data_structures["root"], self.data_structures["soil"],
                                            cmap_property="C_hexose_soil")
            self.soil_grid_in_scene = self.plotter.add_mesh(soil_grid, cmap="hot", show_edges=False, specular=1.,
                                                            opacity=0.1)
        if "shoot" in self.data_structures.keys():
            self.shoot_current_meshes = {}
            shoot_mesh = shoot_plantgl_to_mesh(self.data_structures["shoot"])
            for vid in shoot_mesh.keys():
                self.shoot_current_meshes[vid] = self.plotter.add_mesh(shoot_mesh[vid], color="green",
                                                                        show_edges=False, specular=1.)

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
        self.log_mtg_coordinates()

        if self.simulation_time_in_hours > 0:
            log = f"\r[RUNNING] {self.simulation_time_in_hours} hours | step took {round(self.current_step_start_time - self.previous_step_start_time, 1)} s | {time.strftime('%H:%M:%S', time.gmtime(int(self.elapsed_time)))} since simulation started"
            if self.echo:
                sys.stdout.write(log)
            logging.info(log)

        if self.recording_sums:
            self.recording_summed_MTG_properties_to_csv()
            
        if self.recording_raw:
            self.recording_raw_MTG_properties_in_xarray()

        if self.recording_barcodes:
            self.barcode_from_mtg()

        if self.recording_images and self.simulation_time_in_hours % 100 == 0.:
            self.plotter.screenshot(os.path.join(self.outputs_dirpath, f"root_images/snapshot_{self.simulation_time_in_hours}.png"))

        # Only the costly logging operations are restricted here
        if self.simulation_time_in_hours % self.logging_period_in_hours == 0:
            if self.recording_mtg:
                self.recording_mtg_files()
            if self.recording_images:
                self.recording_images_with_pyvista()

        self.simulation_time_in_hours += self.time_step_in_hours
        self.previous_step_start_time = self.current_step_start_time

    def run_and_monitor_model_step(self):
        if self.recording_performance:
            # Also runs the time step!!
            self.recording_step_performance()
        else:
            self.model_instance.run()

    def time_and_run(self, func_name="run"):
        simulation_time_in_hours = self.simulation_time_in_hours

        self = self.model_instance
        steps = inspect.getsource(getattr(self, func_name)).replace("    ", "").split("\n")[1:]
        steps = [step for step in steps if "#" not in step and len(step) != 0]
        steps_times = {k: 0. for k in steps}
        loop_start = time.time()
        for step in steps:
            t_start = time.time()
            exec (step)
            steps_times[step] = time.time() - t_start
        total_time = time.time() - loop_start
        for step in steps:
            steps_times[step] /= total_time
        # To visualize first which step is the most costly
        steps_times = dict(sorted(steps_times.items(), key=lambda item: item[1]))
        step_elapsed = pd.DataFrame(
            steps_times,
            columns=steps,
            index=[simulation_time_in_hours])

        return step_elapsed

    def recording_step_performance(self):
        step_elapsed = self.time_and_run()
        self.simulation_performance = pd.concat([self.simulation_performance, step_elapsed])

    def recording_summed_MTG_properties_to_csv(self):
        # We init the dict that will capture all recorded properties of the current time-step
        step_plant_scale = {}

        # Fist we log from both mtgs:
        for compartment in self.props.keys():
            if compartment == "root":
                prop = self.props[compartment]
                emerged_vids = [k for k, v in prop["struct_mass"].items() if v > 0]
                emerged_vids.remove(1)
                for var in self.summable_output_variables:
                    if var in prop.keys():
                        step_plant_scale.update({var: sum([prop[var][v] for v in emerged_vids])})
                for var in self.meanable_output_variables:
                    if var in prop.keys():
                        if len(emerged_vids) > 0:
                            step_plant_scale.update({var: np.mean([prop[var][v] for v in emerged_vids if v in prop[var].keys()])})
                        else:
                            step_plant_scale.update({var: None})
                for var in self.plant_scale_state:
                    if var in prop.keys():
                        step_plant_scale.update({var: sum(prop[var].values())})

            elif compartment == "shoot":
                prop = self.props[compartment]
                for var in self.summable_output_variables:
                    if var in prop.keys():
                        step_plant_scale.update({var: prop[var]})

            # Then we log from the soil grid (if available)
            elif compartment == "soil":
                for var in self.summable_output_variables:
                    if var in self.props["soil"].keys():
                        step_plant_scale.update({var: np.sum(self.props["soil"][var])})

                for var in self.meanable_output_variables:
                    if var in self.props["soil"].keys():
                        step_plant_scale.update({var: np.mean(self.props["soil"][var])})

        step_sum = pd.DataFrame(step_plant_scale,
                                columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state,
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
                       coordinates=dict(
                           vid=dict(unit="adim", value_example=1, description="Root segment identifier index"),
                           t=dict(unit="h", value_example=1, description="Model time step")),
                       description="Model local root MTG properties over time",
                       time=0):
        # convert dict to dataframe with index corresponding to coordinates in topology space
        # (not just x, y, z, t thanks to MTG structure)
        props_dict = {k: v for k, v in self.props["root"].items() if type(v) == dict and k in variables}
        props_df = pd.DataFrame.from_dict(props_dict)
        props_df["vid"] = props_df.index
        props_df["t"] = [time for k in range(props_df.shape[0])]
        props_df = props_df.set_index(list(coordinates.keys()))

        # Select properties actually used in the current version of the target model
        #props_df = props_df[list(variables.keys())]

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
        for k in props_dict.keys():
            getattr(props_ds, k).attrs.update(variables[k])

        return props_ds

    def recording_mtg_files(self):
        with open(os.path.join(self.MTG_files_dirpath, f'root_{self.simulation_time_in_hours}.pckl'), "wb") as f:
            pickle.dump(self.data_structures["root"], f)

    def recording_images_with_pyvista(self):
        if "root" in self.data_structures.keys():
            # TODO : step back according to max(||x2-x1||, ||y2-y1||, ||z2-z1||)
            root_system_mesh, color_property = plot_mtg_alt(self.data_structures["root"], cmap_property=self.plotted_property, flow_property=self.flow_property)
            if 0. in color_property:
                color_property.remove(0.)
            self.flow_property
            self.plotting_root_hairs=True
            #if self.plotting_root_hairs:
                
            # Accounts for smooth color bar transitions for videos.
            self.prop_mins = self.prop_mins[1:] + [min(color_property)]
            self.prop_maxs = self.prop_maxs[1:] + [max(color_property)]

            mean_mins = np.mean([e for e in self.prop_mins if e is not None])
            mean_maxs = np.mean([e for e in self.prop_maxs if e is not None])

            if self.prop_mins[-1] < self.all_times_low and self.prop_mins[-1] != 0:
                self.all_times_low = self.prop_mins[-1]
            elif mean_mins > 1.1 * self.all_times_low:
                self.all_times_low = mean_mins
            if self.prop_maxs[-1] > self.all_times_high:
                self.all_times_high = self.prop_maxs[-1]
            elif mean_maxs < 0.9 * self.all_times_high:
                self.all_times_high = mean_maxs

            self.plotter.remove_actor(self.current_mesh)
            self.plotter.remove_actor(self.plot_text)
            # self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap="jet",
            #                                           clim=[self.all_times_low, self.all_times_high], show_edges=False,
            #                                           specular=1., log_scale=True)
            # TP, just to have a stable scale.
            #clim = [1e-10, 1]
            clim = [1e-10, 3e-9]
            log_scale = True
            self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap="jet",
                                                      clim=clim, show_edges=False,
                                                      specular=1., log_scale=log_scale)
            # TODO : Temporary, just because the meteo file begins at PAR peak
            self.plot_text = self.plotter.add_text(
                f" day {int((self.simulation_time_in_hours + 12) / 24)} : {(self.simulation_time_in_hours + 12) % 24} h",
                position="upper_left")
            if "soil" in self.data_structures.keys() and self.show_soil:
                soil_grid = soil_voxels_mesh(self.data_structures["root"], self.data_structures["soil"],
                                             cmap_property="C_mineralN_soil")
                self.plotter.remove_actor(self.soil_grid_in_scene)
                self.soil_grid_in_scene = self.plotter.add_mesh(soil_grid, cmap="hot", show_edges=False, specular=1.,
                                                                opacity=0.1)
            # Usefull to set new camera angle
            #print(self.plotter.camera_position)

            #pgl.Viewer.display()
            # If needed, we wait for a few seconds so that the graph is well positioned:
            #time.sleep(0.1)
            #image_name = os.path.join(self.root_images_dirpath, f'root_{self.simulation_time_in_hours}.png')
            #pgl.Viewer.saveSnapshot(image_name)

        if "shoot" in self.data_structures.keys():
            shoot_meshes = shoot_plantgl_to_mesh(self.data_structures["shoot"])
            for vid in shoot_meshes.keys():
                if vid in self.shoot_current_meshes:
                    self.plotter.remove_actor(self.shoot_current_meshes[vid])
                self.shoot_current_meshes[vid] = self.plotter.add_mesh(shoot_meshes[vid], color="lightgreen",
                                                                       show_edges=False, specular=1.)

        self.plotter.update()
        self.plotter.write_frame()

    def write_to_disk(self, xarray_list):
        interstitial_dataset = xr.concat(xarray_list, dim="t")
        interstitial_dataset.to_netcdf(
            os.path.join(self.MTG_properties_raw_dirpath, f't={self.simulation_time_in_hours}.nc'))

    def barcode_from_mtg(self):
        props = self.props["root"]
        g = self.data_structures["root"]
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order(g, root):
            if vid == 1:
                g.node(vid).dist_to_collar = 0
                g.node(vid).order = 1
            else:
                parent = g.parent(vid)
                g.node(vid).dist_to_collar = g.node(parent).dist_to_collar + g.node(parent).length
                if props["edge_type"][vid] == "+":
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
                    colored_prop += [plt.cm.cool(np.mean([props[prop][v] for v in new_group]) / 5)]

        self.persistent_barcodes[self.time_step_in_hours] = np.array([[min(axs), max(axs)] for axs in homology_barcode])


    def plot_persistent_diagram(self, persistent_diagram):

        fig, ax = plt.subplots(2)

        for k in range(len(homology_barcode)):
            line = [-k for i in range(len(homology_barcode[k]))]
            ax[0].plot(homology_barcode[k], line, c=colored_prop[k], linewidth=2)

        ax[1].scatter(persitent_diagram[:, 0], persitent_diagram[:, 1], c=colored_prop)

        plt.show()


    def log_mtg_coordinates(self):

        def root_visitor(g, v, turtle, gravitropism_coefficient=0.06):
            n = g.node(v)

            # For displaying the radius or length X times larger than in reality, we can define a zoom factor:
            zoom_factor = 1.
            # We look at the geometrical properties already defined within the root element:
            radius = n.radius * zoom_factor
            length = n.length * zoom_factor
            angle_down = n.angle_down
            angle_roll = n.angle_roll

            # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
            position1 = turtle.getPosition()
            n.x1 = position1[0] / zoom_factor
            n.y1 = position1[1] / zoom_factor
            n.z1 = position1[2] / zoom_factor

            # The direction of the turtle is changed:
            turtle.down(angle_down)
            turtle.rollL(angle_roll)

            # Tropism is then taken into account:
            # diameter = 2 * n.radius * zoom_factor
            # elong = n.length * zoom_factor
            # alpha = tropism_intensity * diameter * elong
            # turtle.rollToVert(alpha, tropism_direction)
            # if g.edge_type(v)=='+':
            # diameter = 2 * n.radius * zoom_factor
            # elong = n.length * zoom_factor
            # alpha = tropism_intensity * diameter * elong
            turtle.elasticity = gravitropism_coefficient * (n.original_radius / g.node(1).original_radius)
            turtle.tropism = (0, 0, -1)

            # The turtle is moved:
            turtle.setId(v)
            if n.type != "Root_nodule":
                # We define the radius of the cylinder to be displayed:
                turtle.setWidth(radius)
                # We move the turtle by the length of the root segment:
                turtle.F(length)
            else: # SPECIAL CASE FOR NODULES
                # We define the radius of the sphere to be displayed:
                turtle.setWidth(radius)
                # We "move" the turtle, but not according to the length (?):
                turtle.F()

            # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
            position2 = turtle.getPosition()
            n.x2 = position2[0] / zoom_factor
            n.y2 = position2[1] / zoom_factor
            n.z2 = position2[2] / zoom_factor


        # We initialize a turtle in PlantGL:
        
        turtle = turt.PglTurtle()
        # We make the graph upside down:
        turtle.down(180)
        # We initialize the scene with the MTG g:
        turt.TurtleFrame(self.data_structures["root"], visitor=root_visitor, turtle=turtle, gc=False)


    def stop(self):

        if self.echo:
            logging.shutdown()
            elapsed_at_simulation_end = self.elapsed_time
            printed_time = time.strftime('%H:%M:%S', time.gmtime(int(elapsed_at_simulation_end)))
            print("")  # to receive the flush
            if len(self.exceptions) == 0:
                print(f"[INFO] Simulation ended after {printed_time} min without error")
                os.rename(os.path.join(self.outputs_dirpath, "[RUNNING] simulation.log"), os.path.join(self.outputs_dirpath, "[FINISHED] simulation.log"))
            else:
                print(f"[INFO] Simulation ended after {printed_time} min, INTERRUPTED BY THE FOLLOWING ERRORS : ")
                for error in self.exceptions:
                    print(" - ", error)
                os.rename(os.path.join(self.outputs_dirpath, "[RUNNING] simulation.log"), os.path.join(self.outputs_dirpath, "[STOPPED] simulation.log"))
            print("[INFO] Now proceeding to data writing on disk...")

        if self.recording_sums:
            if self.compare_to_ref_barcode:
                barcodes_distances = {t: bottleneck_distance(self.ref_persitent_barcodes[t], barcode, 0) for t, barcode in self.persistent_barcodes.items()}
                print(barcodes_distances)
                self.plant_scale_properties["bottleneck_distances"] = pd.Series(barcodes_distances)
                print(self.plant_scale_properties)

            # Saving in memory summed properties
            self.plant_scale_properties.to_csv(
                os.path.join(self.MTG_properties_summed_dirpath, "plant_scale_properties.csv"))

        if not self.recording_images:
            print("[INFO] Saving a final snapshot...")
            self.log_mtg_coordinates()
            self.init_images_plotter()
            self.recording_images_with_pyvista()

        if not self.recording_mtg:
            print("[INFO] Saving the final state of the MTG...")
            self.recording_mtg_files()

        if self.recording_raw:
            # For saved xarray datasets
            if len(self.log_xarray) > 0:
                print("[INFO] Merging stored properties data in one xarray dataset...")
                self.write_to_disk(self.log_xarray)
                del self.log_xarray

            time_step_files = [os.path.join(self.MTG_properties_raw_dirpath, name) for name in
                               os.listdir(self.MTG_properties_raw_dirpath)]
            time_dataset = xr.open_mfdataset(time_step_files)
            time_dataset = time_dataset.assign_coords(coords=self.scenario).expand_dims(
                dim=dict(zip(list(self.scenario.keys()), [1 for k in self.scenario])))
            time_dataset.to_netcdf(self.MTG_properties_raw_dirpath + '/merged.nc')
            del time_dataset
            for file in os.listdir(self.MTG_properties_raw_dirpath):
                if '.nc' in file and file != "merged.nc":
                    os.remove(self.MTG_properties_raw_dirpath + '/' + file)

        if self.recording_barcodes and not self.compare_to_ref_barcode:
            with open(os.path.join(self.MTG_barcodes_dirpath, f'persistent_barcodes.pckl'), "wb") as f:
                pickle.dump(self.persistent_barcodes, f)

        if self.recording_shoot:
            # convert list of outputs into dataframes
            for outputs_df_list, outputs_filename, index_columns in (
                    (self.shoot.axes_all_data_list, "axes_outputs.csv", ['t', 'plant', 'axis']),
                    (self.shoot.organs_all_data_list, "organs_outputs.csv", ['t', 'plant', 'axis', 'organ']),
                    (
                    self.shoot.hiddenzones_all_data_list, "hiddenzones_outputs.csv", ['t', 'plant', 'axis', 'metamer']),
                    (self.shoot.elements_all_data_list, "elements_outputs.csv",
                     ['t', 'plant', 'axis', 'metamer', 'organ', 'element']),
                    (self.shoot.soils_all_data_list, "soil_outputs.csv", ['t', 'plant', 'axis'])
            ):
                outputs_filepath = os.path.join(self.shoot_properties_dirpath, outputs_filename)
                outputs_df = pd.concat(outputs_df_list, keys=self.shoot.all_simulation_steps, sort=False)
                outputs_df.reset_index(0, inplace=True)
                outputs_df.rename({'level_0': 't'}, axis=1, inplace=True)
                outputs_df = outputs_df.reindex(index_columns + outputs_df.columns.difference(index_columns).tolist(),
                                                axis=1, copy=False)
                outputs_df.fillna(value=np.nan, inplace=True)  # Convert back None to NaN
                outputs_df.to_csv(outputs_filepath)

        if self.recording_performance:
            self.simulation_performance.to_csv(os.path.join(self.outputs_dirpath, "simulation_performance.csv"))

        if self.echo:
            time_writing_on_disk = self.elapsed_time - elapsed_at_simulation_end
            print(f"[INFO] Successfully wrote data on disk after {round(time_writing_on_disk / 60, 1)} minutes")
            print("[LOGGER CLOSES]")

        # self.mtg_persistent_homology(g=self.g)


def test_logger():
    return Logger()
