from ipywidgets import *
from ipyfilechooser import FileChooser
from skimage.transform import rescale
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def plot_aligned_data(reconmetadata, alignmentmetadata, importmetadata, generalmetadata, current_tomo, alignmentdata):

    extend_description_style = {"description_width" :"auto"}
    skip_theta = IntSlider()
    projection_range_x = IntRangeSlider(description="Projection X Range:",layout=Layout(width="70%"),style=extend_description_style)
    projection_range_y = IntRangeSlider(description="Projection Y Range:",layout=Layout(width="70%"),style=extend_description_style)
    projection_range_theta = IntRangeSlider(description="Projection Theta Range:",layout=Layout(width="70%"),style=extend_description_style)
    
    def load_data_for_plot(self):
            if use_aligned_or_raw.value == "Last Recon:" :
                projection_range_x.description="Recon X Range:"
                projection_range_y.description="Recon Y Range:"
                projection_range_Z.description="Recon Z Range:"
                skip_theta.description="Skip z:"
                projection_range_x.max=current_tomo.shape[2] - 1
                projection_range_x.value = [0, current_tomo.shape[2] - 1]
                projection_range_y.max=current_tomo.shape[1] - 1
                projection_range_y.value=[0, current_tomo.shape[1] - 1]  
                projection_range_theta.value=[0, current_tomo.shape[0] - 1]
                projection_range_theta.max=current_tomo.shape[0] - 1

            else:
                projection_range_x.description="Projection X Range:"
                projection_range_y.description="Projection Y Range:"
                projection_range_theta.description="Projection Z Range:"
                skip_theta.description="Skip theta:"
                projection_range_x.max=current_tomo.prj_imgs.shape[2] - 1
                projection_range_x.value = [0, current_tomo.prj_imgs.shape[2] - 1]
                projection_range_y.max=current_tomo.prj_imgs.shape[1] - 1
                projection_range_y.value=[0, current_tomo.prj_imgs.shape[1] - 1]  
                projection_range_theta.value=[0, current_tomo.prj_imgs.shape[0] - 1]
                projection_range_theta.max=current_tomo.prj_imgs.shape[0] - 1


            projection_range_x.min=0
            projection_range_x.step=1
            projection_range_x.disabled=False
            projection_range_x.continuous_update=False
            projection_range_x.orientation="horizontal"
            projection_range_x.readout=True
            projection_range_x.readout_format="d"
            projection_range_x.layout=Layout(width="70%")
            projection_range_x.style=extend_description_style
            
            projection_range_y.min=0
            projection_range_y.step=1
            projection_range_y.disabled=False
            projection_range_y.continuous_update=False
            projection_range_y.orientation="horizontal"
            projection_range_y.readout=True
            projection_range_y.readout_format="d"
            projection_range_y.layout=Layout(width="70%")
            projection_range_y.style=extend_description_style
                    
            projection_range_theta.min=0
            projection_range_theta.step=1
            projection_range_theta.disabled=False
            projection_range_theta.continuous_update=False
            projection_range_theta.orientation="horizontal"
            projection_range_theta.readout=True
            projection_range_theta.readout_format="d"
            projection_range_theta.layout=Layout(width="70%")
            projection_range_theta.style=extend_description_style
            
            skip_theta.value=20
            skip_theta.min=1
            skip_theta.max=50
            skip_theta.step=1
            skip_theta.disabled=False
            skip_theta.continuous_update=False
            skip_theta.orientation="horizontal"
            skip_theta.readout=True
            skip_theta.readout_format="d"
            skip_theta.layout=Layout(width="70%")
            skip_theta.style=extend_description_style

    load_data_button = Button(
        description="Click to load the selected data.", layout=Layout(width="auto")
    )
    load_data_button.on_click(load_data_for_plot)

    # Radio for use of raw/normalized data, or normalized + aligned.
    def update_before_after(self):
        global current_tomo
        reconmetadata["tomo"]["aligned_unaligned"] = self["new"]
        if self["new"] == "Aligned":
            current_tomo = alignmentdata[-1].tomo
            reconmetadata["wd"] = alignmentdata[-1].alignment_wd_child
            reconmetadata["tomo"] = current_tomo
        if self["new"] == "Unaligned":
            current_tomo = tomo_norm_mlog
            reconmetadata["wd"] = importmetadata["tomo"]["fpath"]
            reconmetadata["tomo"] = current_tomo
        if self["new"] == "Last Recon":
            current_tomo = alignmentdata[-1].recon


    # stuff needed for importing a new file from here. not necessary.
    radio_import_options = ["tiff", "tiff stack", "h5", "numpy array"]

    def create_filetype_radio(
        description, options=radio_import_options, value="tiff", disabled=False
    ):
        radio = RadioButtons(
            options=options,
            value=value,
            description=description,
            disabled=disabled,
            style={"description_width": "auto"},
        )
        return radio

    use_aligned_or_raw = create_filetype_radio("Data Source:", options=[ "Unaligned","Aligned", "Last Recon"], value = "Unaligned")
    use_aligned_or_raw.observe(update_before_after, names="value")
    def update_datatype(self):
        reconmetadata["tomo"]["imgtype"] = self["new"]
    tomo_radio = create_filetype_radio("Tomo Image Type:")
    tomo_radio.observe(update_datatype, names="value")

    plot_output = Output()
    movie_output = Output()

    def plot_projection_movie(tomodata, range_x, range_y, range_z, skip, scale_factor):

        frames = []
        animSliceNos = range(range_z[0], range_z[1], skip)
        volume = tomodata.prj_imgs[
            range_z[0] : range_z[1] : skip,
            range_y[0] : range_y[1] : 1,
            range_x[0] : range_x[1] : 1,
        ]
        volume_rescaled = rescale(
            volume, (1, scale_factor, scale_factor), anti_aliasing=False
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(animSliceNos)):
            frames.append([ax.imshow(volume_rescaled[i], cmap="viridis")])
        ani = animation.ArtistAnimation(
            fig, frames, interval=50, blit=True, repeat_delay=100
        )
        # plt.close()
        display(HTML(ani.to_jshtml()))

    def create_projection_movie_on_click(button_click):
        movie_output.clear_output()
        with movie_output:
            create_movie_button.button_style = "info"
            create_movie_button.icon = "gear"
            create_movie_button.description = "Making a movie."
            plot_projection_movie(
                current_tomo,
                projection_range_x.value,
                projection_range_y.value,
                projection_range_theta.value,
                skip_theta.value,
                0.1,
            )
            create_movie_button.button_style = "success"
            create_movie_button.icon = "square-check"
            create_movie_button.description = "Do it again?"
    
    # Making a movie button
    create_movie_button = Button(
        description="Click me to create a movie", layout=Layout(width="auto")
    )
    create_movie_button.on_click(create_projection_movie_on_click)
    movie_output = Output()
    movie_output.layout = Layout(width="100%", height="100%", align_items="center")

    grid_movie = GridBox(
        children=[create_movie_button, movie_output],
        layout=Layout(
            width="100%",
            grid_template_rows="auto",
            grid_template_columns="15% 84%",
            grid_template_areas="""
            "create_movie_button movie_output"
            """,
        ),
    )

    plot_box_layout = Layout(
        border="3px solid blue",
        width="100%",
        height="auto",
        align_items="center",
        justify_content="center",
    )

    plot_vbox = VBox(
        [   
            load_data_button,
            use_aligned_or_raw,
            projection_range_x,
            projection_range_y,
            projection_range_theta,
            skip_theta,
            grid_movie,
        ],
        layout=plot_box_layout,
    )

    return plot_vbox


# widget_linker["projection_range_x_movie"] = projection_range_x
# widget_linker["projection_range_y_movie"] = projection_range_y
# widget_linker["projection_range_theta_movie"] = projection_range_theta
# widget_linker["skip_theta_movie"] = skip_theta


#--------File chooser for new tomodata object, if desired.------------#
    # tomofc = FileChooser(path=generalmetadata["starting_wd"])

    # def update_tomofname(self):
    #     reconmetadata["tomo"]["fpath"] = self.selected_path
    #     reconmetadata["tomo"]["fname"] = self.selected_filename
    # tomofc.register_callback(update_tomofname)

    # # Load original data's location.
    # def load_location_original_data(self):
    #     tomofc.reset(path=importmetadata["tomo"]["fpath"])
    # load_location_orig_data_button = Button(
    #     description="Load orig. location", layout=Layout(width="auto")
    # )
    # load_location_orig_data_button.on_click(load_location_original_data)

    # # Load aligned data's location.
    # def load_location_aligned_data(self):
    #     tomofc.reset(path=alignmentdata[-1].alignment_wd_child)
    # load_location_aligned_data_button = Button(
    #     description="Load aligned location", layout=Layout(width="auto")
    # )
    # load_location_aligned_data_button.on_click(load_location_aligned_data)