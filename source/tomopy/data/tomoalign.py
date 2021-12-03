from tqdm.notebook import tnrange, tqdm
from joblib import Parallel, delayed
from time import process_time, perf_counter, sleep
from skimage.registration import phase_cross_correlation
from skimage import transform
from tomopy.recon import wrappers
from tomopy.prep.alignment import scale as scale_tomo
from contextlib import nullcontext
from tomopy.recon import algorithm
from skimage.transform import rescale
from tomopy.misc.corr import circ_mask
from copy import deepcopy, copy

import tomopy.data.tomodata as td
import matplotlib.pyplot as plt
import datetime
import time
import json
import astra
import os
import tifffile as tf
import cupy as cp
import tomopy
import numpy as np


class TomoAlign:
    """
    Class for performing alignments.

    Parameters
    ----------
    tomo : TomoData object.
        Normalize the raw tomography data with the TomoData class. Then,
        initialize this class with a TomoData object.
    metadata : metadata from setup in widget-based notebook.
    """

    def __init__(
        self,
        tomo,
        metadata,
        alignment_wd=None,
        alignment_wd_child=None,
        prj_aligned=None,
        shift=None,
        sx=None,
        sy=None,
        recon=None,
    ):

        self.tomo = tomo  # tomodata object
        self.metadata = metadata
        self.prj_range_x = metadata["opts"]["prj_range_x"]
        self.prj_range_y = metadata["opts"]["prj_range_y"]
        self.shift = shift
        self.sx = sx
        self.sy = sy
        self.conv = None
        self.recon = recon
        self.alignment_wd = alignment_wd
        self.alignment_wd_child = alignment_wd_child

        # setting up output callback context managers
        if "callbacks" in metadata:
            if "methodoutput" in metadata["callbacks"]:
                self.method_bar_cm = metadata["callbacks"]["methodoutput"]
            else:
                self.method_bar_cm = nullcontext()
            if "output1" in metadata["callbacks"]:
                self.output1_cm = metadata["callbacks"]["output1"]
            else:
                self.output1_cm = nullcontext()
            if "output2" in metadata["callbacks"]:
                self.output2_cm = metadata["callbacks"]["output2"]
            else:
                self.output2_cm = nullcontext()
        else:
            self.method_bar_cm = nullcontext()
            self.output1_cm = nullcontext()
            self.output2_cm = nullcontext()

        # creates working directory based on time
        # creates multiple alignments based on
        if self.metadata["alignmultiple"] == True:
            self.make_wd_and_go()
            self.align_multiple()
        else:
            if self.alignment_wd is None:
                self.make_wd_and_go()
            self.align()

    def make_wd_and_go(self):
        now = datetime.datetime.now()
        os.chdir(self.metadata["generalmetadata"]["workingdirectorypath"])
        dt_string = now.strftime("%Y%m%d-%H%M-")
        os.mkdir(dt_string + "alignment")
        os.chdir(dt_string + "alignment")
        self.save_align_metadata()
        self.alignment_wd = os.getcwd()

    def align_multiple(self):

        metadata_list = []
        for key in self.metadata["methods"]:
            d = self.metadata["methods"]
            keys_to_remove = set(self.metadata["methods"].keys())
            keys_to_remove.remove(key)
            _d = {k: d[k] for k in set(list(d.keys())) - keys_to_remove}
            _metadata = self.metadata.copy()
            _metadata["methods"] = _d
            _metadata["alignmultiple"] = False
            metadata_list.append(_metadata)

        for metadata in metadata_list:
            self.metadata["callbacks"]["button"].description = (
                "Starting" + " " + list(metadata["methods"].keys())[0]
            )
            self.__init__(self.tomo, metadata, alignment_wd=self.alignment_wd)

    def align(self):
        """
        Aligns a TomoData object using options in GUI.
        """
        proj_range_x_low = self.metadata["opts"]["prj_range_x"][0]
        proj_range_x_high = self.metadata["opts"]["prj_range_x"][1]
        proj_range_y_low = self.metadata["opts"]["prj_range_y"][0]
        proj_range_y_high = self.metadata["opts"]["prj_range_y"][1]
        self.prj_aligned = self.tomo.prj_imgs[
            :,
            proj_range_y_low:proj_range_y_high:1,
            proj_range_x_low:proj_range_x_high:1,
        ].copy()

        tic = time.perf_counter()
        self.joint_astra_cupy()
        toc = time.perf_counter()

        self.metadata["alignment_time"] = {
            "seconds": toc - tic,
            "minutes": (toc - tic) / 60,
            "hours": (toc - tic) / 3600,
        }

        self.save_align_data()

    def save_align_metadata(self):
        # from https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("overall_alignment_metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)

    def save_align_data(self):
        # if on the second alignment, go into the directory most recently saved
        if self.metadata["align_number"] > 0:
            os.chdir(self.alignment_wd_child)
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M-")
        method_str = list(self.metadata["methods"].keys())[0]

        if (
            "SIRT_CUDA" in self.metadata["methods"]
            and "Faster" in self.metadata["methods"]["SIRT_CUDA"]
        ):
            if self.metadata["methods"]["SIRT_CUDA"]["Faster"] == True:
                method_str = method_str + "-faster"
            if self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == True:
                method_str = method_str + "-fastest"
        os.mkdir(dt_string + method_str)
        os.chdir(dt_string + method_str)

        # save child working directory for use in multiple alignments
        self.alignment_wd_child = os.getcwd()

        # https://stackoverflow.com/questions/51674222/how-to-make-json-dumps-in-python-ignore-a-non-serializable-field
        def safe_serialize(obj, f):
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            return json.dump(obj, f, default=default, indent=4)

        with open("metadata.json", "w+") as f:
            a = safe_serialize(self.metadata, f)
        if self.metadata["opts"]["downsample"] == True:
            np.save("aligned_tomo_data_downsampled", self.prj_aligned)
        else:
            np.save("aligned_tomo_data", self.prj_aligned)
        np.save("sx", self.sx)
        np.save("sy", self.sy)
        np.save("conv", self.conv)
        np.save("last_recon", self.recon)
        if self.metadata["align_number"] == 0:
            os.chdir(self.alignment_wd)
        else:
            os.chdir(self.alignment_wd_child)

    def joint_astra_cupy(
        self,
        blur=True,
        rin=0.5,
        rout=0.8,
    ):
        # Needs scaling for skimage float operations.

        self.prj_aligned, scl = scale_tomo(self.prj_aligned)
        # Shift arrays
        self.sx = np.zeros((self.prj_aligned.shape[0]))
        self.sy = np.zeros((self.prj_aligned.shape[0]))
        self.conv = np.zeros((self.metadata["opts"]["num_iter"]))

        # Pad images BEFORE scaling. This is so sx, sy can easily be
        # rescaled. Increase padding in settings when downsampling - 10 pixels
        # padding will turn into 1 pixel for downsample_factor = 0.1.
        pad = self.metadata["opts"]["pad"]
        npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
        self.prj_aligned = np.pad(
            self.prj_aligned, npad, mode="constant", constant_values=0
        )

        if self.metadata["opts"]["downsample"] == True:
            downsample_factor = self.metadata["opts"]["downsample_factor"]
            self.prj_aligned = rescale(
                self.prj_aligned,
                (1, downsample_factor, downsample_factor),
                anti_aliasing=True,
            )
            pad_downsampled = tuple([int(downsample_factor*x) for x in pad])
        # at this point padding is pad*downsample_factor

        # Initialization of reconstruction dataset
        tomo_shape = self.prj_aligned.shape
        self.recon = np.empty(
            (tomo_shape[1], tomo_shape[2], tomo_shape[2]), dtype=np.float32
        )

        # Options go into kwargs which go into recon()
        kwargs = {}
        options = {
            "proj_type": "cuda",
            "method": list(self.metadata["methods"].keys())[0],
            "num_iter": 1,
        }
        kwargs["options"] = options

        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        batchsize = self.metadata["opts"]["batch_size"]

        with self.method_bar_cm:
            method_bar = tqdm(
                total=self.metadata["opts"]["num_iter"],
                desc=options["method"],
                display=True,
            )

        for n in range(self.metadata["opts"]["num_iter"]):
            tic = perf_counter()
            if np.mod(n, 1) == 0:
                _rec = self.recon
            if self.metadata["methods"]["SIRT_CUDA"]["Faster"] == True:
                self.recon = self.recon_sirt_3D(self.prj_aligned)
            elif self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == True:
                self.recon = self.recon_sirt_3D_allgpu(self.prj_aligned)
            else:
                self.recon = tomopy.recon(
                    self.prj_aligned,
                    self.tomo.theta,
                    algorithm=wrappers.astra,
                    init_recon=_rec,
                    ncore=None,
                    **kwargs,
                )
            method_bar.update()

            self.recon = np.array_split(self.recon, batchsize, axis=0)
            shift_cpu = []
            sim = []

            with self.output1_cm:
                self.metadata["callbacks"]["output1"].clear_output()
                for i in tnrange(len(self.recon), desc="Re-projection", leave=True):
                    _rec = self.recon[i].copy()
                    vol_geom = astra.create_vol_geom(
                        _rec.shape[1], _rec.shape[1], _rec.shape[0]
                    )
                    phantom_id = astra.data3d.create("-vol", vol_geom, data=_rec)
                    proj_geom = astra.create_proj_geom(
                        "parallel3d",
                        1,
                        1,
                        _rec.shape[0],
                        _rec.shape[1],
                        self.tomo.theta,
                    )
                    projections_id, _sim = astra.creators.create_sino3d_gpu(
                        phantom_id, proj_geom, vol_geom
                    )
                    # if self.metadata["methods"]["SIRT_CUDA"]["Faster"] and self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == False

                    _sim = _sim.swapaxes(0, 1)

                    sim.append(_sim)
                    astra.data3d.delete(projections_id)
                    astra.data3d.delete(phantom_id)
            del _sim
            del _rec
            sim = np.concatenate(sim, axis=1)
            main_logger = self.metadata["generalmetadata"]["main_logger"]

            if (
                self.metadata["methods"]["SIRT_CUDA"]["Faster"] == False
                and self.metadata["methods"]["SIRT_CUDA"]["Fastest"] == False
            ):
                sim = np.flip(sim, axis=0)
            self.recon = np.concatenate(self.recon, axis=0)

            # Blur edges.
            # TODO: come up with options for rin, rout (blur radius)
            if blur:
                _prj = tomopy.prep.alignment.blur_edges(self.prj_aligned, rin, rout)
                _sim = tomopy.prep.alignment.blur_edges(sim, rin, rout)
            else:
                _prj = self.prj_aligned
                _sim = sim

            # Cut the data up into batchsize (along projection axis) so that the GPU can handle it.
            # This number will change depending on your GPU memory.
            
            _prj = np.array_split(_prj, batchsize, axis=0)
            _sim = np.array_split(_sim, batchsize, axis=0)
            shift_cpu = []

            with self.output1_cm:
                for i in tnrange(len(_prj), desc="Cross-correlation", leave=True):
                    # use unpadded regions to do the correlation
                    shift_gpu = phase_cross_correlation(
                        _prj[i][:,pad_downsampled[1]:-pad_downsampled[1]:1,pad_downsampled[0]:-pad_downsampled[0]:1],
                        _sim[i][:,pad_downsampled[1]:-pad_downsampled[1]:1,pad_downsampled[0]:-pad_downsampled[0]:1],
                        upsample_factor=self.metadata["opts"]["upsample_factor"],
                        return_error=False,
                    )
                    shift_cpu.append(cp.asnumpy(shift_gpu))
            self.shift = np.concatenate(shift_cpu, axis=1)
            with self.output1_cm:
                self.prj_aligned, self.sx, self.sy, err = transform_parallel(
                    self.prj_aligned, self.sx, self.sy, self.shift, self.metadata
                )
                self.conv[n] = np.linalg.norm(err)

            with self.output2_cm:
                plt.clf()
                self.plotIm(sim)
                self.plotSxSy()
                main_logger.info(self.sx)
                print(f"Error = {np.linalg.norm(err):3.3f}.")
                self.metadata["callbacks"]["output2"].clear_output(wait=True)

        # Re-normalize data
        method_bar.close()
        self.prj_aligned *= scl

        # rescale change in x, y, and shift based on downsampling
        # TODO: better to have other variables _sx, _sy, _shift inside loop
        # above.
        new_prj_imgs = deepcopy(self.tomo.prj_imgs)
        if self.metadata["opts"]["downsample"] == True:
            downsample_factor = self.metadata["opts"]["downsample_factor"]
            self.sx = self.sx / downsample_factor
            self.sy = self.sy / downsample_factor
            self.shift = self.shift / downsample_factor

            # make new dataset and pad/shift it for the next round
            pad = self.metadata["opts"]["pad"]
            npad = ((0, 0), (pad[1], pad[1]), (pad[0], pad[0]))
            new_prj_imgs = np.pad(new_prj_imgs, npad, mode="constant", constant_values=0)

        new_prj_imgs = warp_projections(new_prj_imgs, self.sx, self.sy, self.metadata)
        new_prj_imgs = trim_padding(new_prj_imgs)
        self.tomo = td.TomoData(
            prj_imgs=new_prj_imgs, metadata=self.metadata["importmetadata"]["tomo"]
        )
        self.recon = circ_mask(self.recon, 0)

        return self

    def plotIm(self, sim, projection_num=50):
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(self.prj_aligned[projection_num], cmap="gray")
        ax1.set_axis_off()
        ax1.set_title("Projection Image")
        ax2.imshow(sim[projection_num], cmap="gray")
        ax2.set_axis_off()
        ax2.set_title("Re-projected Image")
        plt.show()

    def plotSxSy(self):
        plotrange = range(self.prj_aligned.shape[0])
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax1.cla()
        ax2.cla()
        ax1.plot(plotrange, self.sx)
        ax1.set_title("Sx")
        ax2.plot(plotrange, self.sy)
        ax2.set_title("Sy")
        plt.show()

    def recon_sirt_3D(self, prj):
        # Init tomo in sinogram order
        sinograms = algorithm.init_tomo(prj, 0)
        num_proj = sinograms.shape[1]
        num_y = sinograms.shape[0]
        num_x = sinograms.shape[2]
        # assume angles used are the same as parent tomography
        angles = self.tomo.theta
        proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
        vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
        projector = astra.create_projector("cuda3d", proj_geom, vol_geom)
        astra.plugin.register(astra.plugins.SIRTPlugin)
        W = astra.OpTomo(projector)
        rec_sirt = W.reconstruct("SIRT-PLUGIN", sinograms, 1)
        return rec_sirt

    def recon_sirt_3D_allgpu(self, prj):
        # Init tomo in sinogram order
        sinograms = algorithm.init_tomo(prj, 0)
        num_proj = sinograms.shape[1]
        num_y = sinograms.shape[0]
        num_x = sinograms.shape[2]
        # assume angles used are the same as parent tomography
        angles = self.tomo.theta
        proj_geom = astra.create_proj_geom("parallel3d", 1, 1, num_y, num_x, angles)
        vol_geom = astra.create_vol_geom(num_x, num_x, num_y)
        sinograms_id = astra.data3d.create("-sino", proj_geom, sinograms)
        rec_id = astra.data3d.create("-vol", vol_geom)
        reco_alg = "SIRT3D_CUDA"
        cfg = astra.astra_dict(reco_alg)
        cfg["ProjectionDataId"] = sinograms_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 2)
        rec_sirt = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinograms_id)
        return rec_sirt


def transform_parallel(prj, sx, sy, shift, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1] * metadata["opts"]["downsample_factor"]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] * metadata["opts"]["downsample_factor"]
    )

    def transform_algorithm(prj, shift, sx, sy, m):
        shiftm = shift[:, m]
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m] + shiftm[1]) < shift_x_condition
            and np.absolute(sy[m] + shiftm[0]) < shift_y_condition
        ):
            sx[m] += shiftm[1]
            sy[m] += shiftm[0]
            err[m] = np.sqrt(shiftm[0] * shiftm[0] + shiftm[1] * shiftm[1])

            # similarity transform shifts in (x, y)
            tform = transform.SimilarityTransform(translation=(shiftm[1], shiftm[0]))
            prj[m] = transform.warp(prj[m], tform, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm)(prj, shift, sx, sy, m)
        for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj, sx, sy, err


def warp_projections(prj, sx, sy, metadata):
    num_theta = prj.shape[0]
    err = np.zeros((num_theta + 1, 1))
    shift_y_condition = (
        metadata["opts"]["pad"][1]
    )
    shift_x_condition = (
        metadata["opts"]["pad"][0] 
    )

    def transform_algorithm_warponly(prj, sx, sy, m):
        # don't let it shift if the value is larger than padding
        if (
            np.absolute(sx[m]) < shift_x_condition
            and np.absolute(sy[m]) < shift_y_condition
        ):
            # similarity transform shifts in (x, y)
            tform = transform.SimilarityTransform(translation=(sx[m], sy[m]))
            prj[m] = transform.warp(prj[m], tform, order=5)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(transform_algorithm_warponly)(prj, sx, sy, m)
        for m in tnrange(num_theta, desc="Transformation", leave=True)
    )
    return prj


def init_new_from_prior(prior_tomoalign, metadata):
    prj_imgs = deepcopy(prior_tomoalign.tomo.prj_imgs)
    new_tomo = td.TomoData(
        prj_imgs=prj_imgs, metadata=metadata["importmetadata"]["tomo"]
    )
    new_align_object = TomoAlign(
        new_tomo,
        metadata,
        alignment_wd=prior_tomoalign.alignment_wd,
        alignment_wd_child=prior_tomoalign.alignment_wd_child,
    )
    return new_align_object


def trim_padding(prj):
    # https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
    xs, ys, zs = np.where(prj != 0)

    minxs = np.min(xs)
    maxxs = np.max(xs)
    minys = np.min(ys)
    maxys = np.max(ys)
    minzs = np.min(zs)
    maxzs = np.max(zs)

    # extract cube with extreme limits of where are the values != 0
    result = prj[minxs : maxxs + 1, minys : maxys + 1, minzs : maxzs + 1]
    # not sure why +1 here.

    return result