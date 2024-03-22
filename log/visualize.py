
import numpy as np

from openalea.mtg.plantframe import color
from openalea.mtg import turtle as turt
import openalea.plantgl.all as pgl
import pyvista as pv

from math import cos, sin, floor
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import numpy as np
from time import sleep
from openalea.mtg.traversal import pre_order


def get_root_visitor():
    """
    This function describes the movement of the 'turtle' along the MTG for creating a graph on PlantGL.
    :return: root_visitor
    """
    
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

    return root_visitor

def my_colormap(g, property_name, cmap='jet', vmin=None, vmax=None, lognorm=True):
    """
    This function computes a property 'color' on a MTG based on a given MTG's property.
    :param g: the investigated MTG
    :param property_name: the name of the property of the MTG that will be displayed
    :param cmap: the type of color map
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param lognorm: a Boolean describing whether the scale is logarithmic or not
    :return: the MTG with the corresponding color
    """

    # We make sure that the user did not accidently switch between vmin and vmax:
    if vmin >= vmax:
        raise Exception("Sorry, the vmin and vmax values of the color scale of the graph are wrong!")
    if lognorm and (vmin <=0 or vmax <=0):
        raise Exception("Sorry, it is not possible to represent negative values in a log scale - check vmin and vmax!")

    prop = g.property(property_name)
    keys = prop.keys()
    values = list(prop.values())
    # m, M = int(values.min()), int(values.max())

    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)

    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=np.int).tolist()

    g.properties()['color'] = dict(zip(keys, colors))

    return g

def custom_colorbar(min=0, max=1, unit='Some Units'):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=min, vmax=max)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label(unit)
    plt.ion()
    fig.show()

def prepareScene(scene, width=1200, height=1200, scale=0.8, x_center=0., y_center=0., z_center=0.,
                 x_cam=0., y_cam=0., z_cam=-1.5, grid=False):
    """
    This function returns the scene that will be used in PlantGL to display the MTG.
    :param scene: the scene to start with
    :param width: the width of the graph (in pixels)
    :param height: the height of the graph (in pixels)
    :param scale: a dimensionless factor for zooming in or out
    :param x_center: the x-coordinate of the center of the graph
    :param y_center: the y-coordinate of the center of the graph
    :param z_center: the z-coordinate of the center of the graph
    :param x_cam: the x-coordinate of the camera looking at the center of the graph
    :param y_cam: the y-coordinate of the camera looking at the center of the graph
    :param z_cam: the z-coordinate of the camera looking at the center of the graph
    :param grid: a Boolean describing whether grids should be displayed on the graph
    :return: scene
    """

    # We define the coordinates of the point cam_target that will be the center of the graph:
    cam_target = pgl.Vector3(x_center * scale,
                             y_center * scale,
                             z_center * scale)
    # We define the coordinates of the point cam_pos that represents the position of the camera:
    cam_pos = pgl.Vector3(x_cam * scale,
                          y_cam * scale,
                          z_cam * scale)
    # We position the camera in the scene relatively to the center of the scene:
    pgl.Viewer.camera.lookAt(cam_pos, cam_target)
    # We define the dimensions of the graph:
    pgl.Viewer.frameGL.setSize(width, height)
    # We define whether grids are displayed or not:
    pgl.Viewer.grids.set(grid, grid, grid, grid)

    return scene

def plot_mtg(g, prop_cmap='C_hexose_root', cmap='brg', lognorm=True, vmin=1e-6, vmax=1e-0,
             root_hairs_display=True,
             width=1200, height=910,
             x_center=0., y_center=0., z_center=-0.1,
             x_cam=0.2, y_cam=0., z_cam=-0.2):
    
    """
    This function creates a graph on PlantGL that displays a MTG and color it according to a specified property.
    :param g: the investigated MTG
    :param prop_cmap: the name of the property of the MTG that will be displayed in color
    :param cmap: the type of color map
    :param lognorm: a Boolean describing whether the scale is logarithmic or not
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param x_center: the x-coordinate of the center of the graph
    :param y_center: the y-coordinate of the center of the graph
    :param z_center: the z-coordinate of the center of the graph
    :param x_cam: the x-coordinate of the camera looking at the center of the graph
    :param y_cam: the y-coordinate of the camera looking at the center of the graph
    :param z_cam: the z-coordinate of the camera looking at the center of the graph
    :return: the updated scene
    """

    # Consider: https://learnopengl.com/In-Practice/Text-Rendering

    # DISPLAYING ROOTS:
    #------------------
    visitor = get_root_visitor()
    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()
    # We make the graph upside down:
    turtle.down(180)
    # We initialize the scene with the MTG g:
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)
    # We update the scene with the specified position of the center of the graph and the camera:
    #prepareScene(scene, width=width, height=height,
    #             x_center=x_center, y_center=y_center, z_center=z_center, x_cam=x_cam, y_cam=y_cam, z_cam=z_cam)
    # We compute the colors of the graph:
    # my_colormap(g, prop_cmap, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
    # # We get a list of all shapes in the scene:
    # shapes = dict((sh.id, sh) for sh in scene)
    # # We use the property 'color' of the MTG calculated by the function 'my_colormap':
    # colors = g.property('color')
    # # We cover each node of the MTG:
    # for vid in colors:
    #     if vid in shapes:
    #         n = g.node(vid)
    #         # If the element is not dead:
    #         if n.type != "Dead":
    #             # We color it according to the property cmap defined by the user:
    #             shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.0)
    #         else:
    #             # Otherwise, we print it in black in a semi-transparent way:
    #             shapes[vid].appearance = pgl.Material([0, 0, 0], transparency=0.8)
    #         # property=g.property(prop_cmap)
    #         # if n.property <=0:
    #         #     shapes[vid].appearance = pgl.Material([0, 0, 200])

    #         # SPECIAL CASE: If the element is a nodule, we transform the cylinder into a sphere:
    #         if n.type == "Root_nodule":
    #             # We create a sphere corresponding to the radius of the element:
    #             s = pgl.Sphere(n.radius * 1.)
    #             # We transform the cylinder into the sphere:
    #             shapes[vid].geometry.geometry = pgl.Shape(s).geometry
    #             # We select the parent element supporting the nodule:
    #             index_parent = g.Father(vid, EdgeType='+')
    #             parent = g.node(index_parent)
    #             # We move the center of the sphere on the circle corresponding to the external envelop of the
    #             # parent:
    #             angle = parent.angle_roll
    #             circle_x = parent.radius * cos(angle)
    #             circle_y = parent.radius * sin(angle)
    #             circle_z = 0
    #             shapes[vid].geometry.translation += (circle_x, circle_y, circle_z)

    # # DISPLAYING ROOT HAIRS:
    # #-----------------------
    # if root_hairs_display:
    #     visitor_for_hair = get_root_visitor()
    #     # We initialize a turtle in PlantGL:
    #     turtle_for_hair = turt.PglTurtle()
    #     # We make the graph upside down:
    #     turtle_for_hair.down(180)
    #     # We initialize the scene with the MTG g:
    #     # scene_for_hair = turt.TurtleFrame(g, visitor=visitor_for_hair, turtle=turtle_for_hair, gc=False)
    #     scene_for_hair = turt.TurtleFrame(g, visitor=visitor, turtle=turtle_for_hair, gc=False)
    #     # We update the scene with the specified position of the center of the graph and the camera:
    #     prepareScene(scene_for_hair, width=width, height=height,
    #                  x_center=x_center, y_center=y_center, z_center=z_center, x_cam=x_cam, y_cam=y_cam, z_cam=z_cam)
    #     # We get a list of all shapes in the scene:
    #     shapes_for_hair = dict((sh.id, sh) for sh in scene_for_hair)

    #     # We cover each node of the MTG:
    #     for vid in colors:
    #         if vid in shapes_for_hair:
    #             n = g.node(vid)
    #             # If the element has no detectable root hairs:
    #             if n.root_hair_length<=0.:
    #                 # Then the element is set to be transparent:
    #                 shapes_for_hair[vid].appearance = pgl.Material(colors[vid], transparency=1)
    #             else:
    #                 # We color the root hairs according to the proportion of living and dead root hairs:
    #                 dead_transparency = 0.9
    #                 dead_color_vector=[0,0,0]
    #                 dead_color_vector_Red=dead_color_vector[0]
    #                 dead_color_vector_Green=dead_color_vector[1]
    #                 dead_color_vector_Blue=dead_color_vector[2]

    #                 living_transparency = 0.8
    #                 living_color_vector=colors[vid]
    #                 living_color_vector_Red = colors[vid][0]
    #                 living_color_vector_Green = colors[vid][1]
    #                 living_color_vector_Blue = colors[vid][2]

    #                 living_fraction = n.living_root_hairs_number/n.total_root_hairs_number
    #                 # print("Living fraction is", living_fraction)

    #                 transparency = dead_transparency + (living_transparency - dead_transparency) * living_fraction
    #                 color_vector_Red = floor(dead_color_vector_Red
    #                                          + (living_color_vector_Red - dead_color_vector_Red) * living_fraction)
    #                 color_vector_Green = floor(dead_color_vector_Green
    #                                            + (living_color_vector_Green - dead_color_vector_Green) * living_fraction)
    #                 color_vector_Blue = floor(dead_color_vector_Blue
    #                                           + (living_color_vector_Blue - dead_color_vector_Blue) * living_fraction)
    #                 color_vector = [color_vector_Red,color_vector_Green,color_vector_Blue]

    #                 shapes_for_hair[vid].appearance = pgl.Material(color_vector, transparency=transparency)

    #                 # We finally transform the radius of the cylinder:
    #                 if vid > 1:
    #                     # For normal cases:
    #                     shapes_for_hair[vid].geometry.geometry.geometry.radius = n.radius + n.root_hair_length
    #                 else:
    #                     # For the base of the root system [don't ask why this has not the same formalism..!]:
    #                     shapes_for_hair[vid].geometry.geometry.radius = n.radius + n.root_hair_length

    # # CREATING THE ACTUAL SCENE:
    # #---------------------------
    # # Finally, we update the scene with shapes from roots and, if specified, shapes from root hairs:
    # new_scene = pgl.Scene()
    # for vid in shapes:
    #     new_scene += shapes[vid]
    # if root_hairs_display:
    #     for vid in shapes_for_hair:
    #         new_scene += shapes_for_hair[vid]

    #return new_scene

def plot_xr(datasets, vertice=[], summing=0, selection=[], supplementary_legend=[""]):
    # TODO : convert to class
    root = tk.Tk()
    root.title(f'2D data from vertices {str(vertice)[1:-1]}')
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=10)
    root.columnconfigure(1, weight=10)
    root.columnconfigure(2, weight=1)

    # Listbox widget to add plots
    lb = tk.Listbox(root)
    for k in range(len(selection)):
        lb.insert(k, selection[k])

    # to avoid double window popup
    plt.ioff()
    # Check the number of plots for right subplot divisions
    if len(vertice) in (0, 1):
        fig, ax = plt.subplots()
        ax = [ax]
    else:
        fig, ax = plt.subplots(len(vertice), 1)

    # Embed figure in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)

    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()

    toolbar.grid(row=1, column=1, sticky="NSEW")
    canvas.get_tk_widget().grid(row=2, column=1, sticky="NSEW")
    lb.grid(row=2, column=2, sticky="NSEW", columnspan=2)

    if supplementary_legend == [""]:
        datasets = [datasets]
    for d in range(len(datasets)):
        # If we plot global properties
        if len(vertice) == 0:
            # If properties are spatialized but we want an overall root system summary
            if summing != 0:
                datasets[d] = datasets[d].sum(dim="vid")*summing
            text_annot = [[]]
            if summing != 0:
                for prop in selection:
                    getattr(datasets[d], prop).plot.line(x='t', ax=ax[0], label=prop + supplementary_legend[d])
                    text_annot[0] += [ax[0].text(0, 0, ""), ax[0].text(0, 0, "")]
            else:
                v_extract = datasets[d].stack(stk=[dim for dim in datasets[d].dims if dim not in ("vid", "t")]).sel(vid=1)
                # automatic legends from xarray are structured the following way : modalities x properties
                legend = []
                unit = []
                for combination in np.unique(v_extract.coords["stk"]):
                    combination_extract = v_extract.sel(stk=combination)
                    for prop in selection:
                        getattr(combination_extract, prop).plot.line(x='t', ax=ax[0],
                                                                     label=prop + supplementary_legend[d],
                                                                     add_legend=False)
                        text_annot[0] += [ax[0].text(0, 0, ""), ax[0].text(0, 0, "")]
                        if len(np.unique(v_extract.coords["stk"])) > 1:
                            legend += [combination]
                        else:
                            legend += [""]
                        unit += [getattr(combination_extract, prop).attrs["unit"]]
                        ax[0].get_lines()[-1].set_label('_' + ax[0].get_lines()[-1].get_label() + ' (' + unit[-1] + ')')
                        ax[0].get_lines()[-1].set_visible(False)

            ax[0].set_ylabel("")
            ax[0].set_title("")

        # If we plot local properties
        else:
            text_annot = [[] for k in range(len(vertice))]
            for k in range(len(vertice)):
                v_extract = datasets[d].stack(stk=[dim for dim in datasets[d].dims if dim not in ("vid", "t")]).sel(vid=vertice[k])
                # automatic legends from xarray are structured the following way : modalities x properties
                legend = []
                unit = []
                for combination in np.unique(v_extract.coords["stk"]):
                    combination_extract = v_extract.sel(stk=combination)
                    for prop in selection:
                        getattr(combination_extract, prop).plot.line(x='t', ax=ax[k], label=prop + supplementary_legend[d], add_legend=False)
                        text_annot[k] += [ax[k].text(0, 0, ""), ax[k].text(0, 0, "")]
                        if len(np.unique(v_extract.coords["stk"])) > 1:
                            legend += [combination]
                        else:
                            legend += [""]
                        unit += [getattr(combination_extract, prop).attrs["unit"]]
                        ax[k].get_lines()[-1].set_label('_' + ax[k].get_lines()[-1].get_label() + ' (' + unit[-1]+')')
                        ax[k].get_lines()[-1].set_visible(False)

                ax[k].set_ylabel("")
                ax[k].set_title("")

    if len(vertice) == 0:
        def hover_global(event):
            if event.inaxes == ax[0]:
                # At call remove all annotations to prevent overlap
                for k in text_annot[0]: k.set_visible(False)
                lines = ax[0].get_lines()
                # for all variables lines in the axe
                for l in range(len(lines)):
                    # if the mouse pointer is on the line
                    cont, ind = lines[l].contains(event)
                    if cont and lines[l].get_visible():
                        # get the position
                        posx, posy = [lines[l].get_xdata()[ind['ind'][0]], lines[l].get_ydata()[ind['ind'][0]]]
                        # get variable name
                        label = "{}_{}\n{},{}".format(lines[l].get_label(),
                                                          ["{:.2e}".format(s) for s in legend[l]],
                                                          posx,
                                                          "{:.2e}".format(posy) + " " + unit[l])
                        # add text annotation to the axe and refresh
                        text_annot[0] += [ax[0].text(x=posx, y=posy, s=label)]
                        fig.canvas.draw_idle()
            sleep(1)

        fig.canvas.mpl_connect("motion_notify_event", hover_global)
    else:
        def hover_local(event):
            # for each row
            for axe in range(len(ax)):
                # if mouse event is in the ax
                if event.inaxes == ax[axe]:
                    # At call remove all annotations to prevent overlap
                    for k in text_annot[axe]: k.set_visible(False)
                    # for all variables lines in the axe
                    lines = ax[axe].get_lines()
                    for l in range(len(lines)):
                        # if the mouse pointer is on the line
                        cont, ind = lines[l].contains(event)
                        if cont and lines[l].get_visible():
                            # get the position
                            posx, posy = [lines[l].get_xdata()[ind['ind'][0]], lines[l].get_ydata()[ind['ind'][0]]]
                            # get variable name
                            label = "{}_{}\n{},{}".format(lines[l].get_label(),
                                                          ["{:.2e}".format(s) for s in legend[l]],
                                                          posx,
                                                          "{:.2e}".format(posy) + " " + unit[l])
                            # add text annotation to the axe and refresh
                            text_annot[axe] += [ax[axe].text(x=posx, y=posy, s=label)]
                            fig.canvas.draw_idle()
            sleep(1)

        fig.canvas.mpl_connect("motion_notify_event", hover_local)

    def on_click(event):
        if event.button is MouseButton.LEFT:
            # for each row
            for axe in range(len(ax)):
                # if mouse event is in the ax
                if event.inaxes == ax[axe]:
                    # for all variables lines in the axe
                    for line in ax[axe].get_lines():
                        # if the mouse pointer is on the line
                        cont, ind = line.contains(event)
                        if cont:
                            line.set_visible(False)
                            line.set_label('_'+line.get_label())
                            ax[axe].relim(visible_only=True)
                            ax[axe].autoscale()
                            ax[axe].legend()
            canvas.draw()

    def on_lb_select(event):
        # TODO maybe add possibility to normalize-add a plot for ease of reading
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        # for each row
        for axe in range(len(ax)):
            for line in ax[axe].get_lines():
                if value in line.get_label():
                    line.set_visible(True)
                    if line.get_label()[0] == '_':
                        line.set_label(line.get_label()[1:])
            ax[axe].relim(visible_only=True)
            ax[axe].autoscale()
            ax[axe].legend()
        canvas.draw()

    lb.bind('<<ListboxSelect>>', on_lb_select)

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Finally show figure
    root.update()

def plot_mtg_alt(g, cmap_property):
    props = g.properties()
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    plotted_vids = []
    tubes = []
    for vid in pre_order(g, root):
        if vid not in plotted_vids:
            root = g.Axis(vid)
            plotted_vids += root
            if vid != 1:
                parent = g.Father(vid)
                grandparent = g.Father(parent)
                # We need a minimum of two anchors for the new axis
                root = [grandparent, parent] + root

            points = np.array([[props["x2"][v], props["y2"][v], props["z2"][v]] for v in root])
            spline = pv.Spline(points)
            spline[cmap_property] = [props[cmap_property][v] for v in root]
            # Adjust radius of each element
            spline["radius"] = [props["radius"][v] for v in root]
            tubes += [spline.tube(scalars="radius", absolute=True)]
    
    root_system = pv.MultiBlock(tubes)
    return root_system


    