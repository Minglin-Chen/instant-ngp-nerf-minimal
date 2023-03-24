import sys
import time
import argparse
import os
import commentjson as json
import numpy as np
from tqdm import tqdm
import pandas as pd

from common import *
import pyngp as ngp


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. A path to the training data.")
	parser.add_argument("--network", default="", help="Path to the network config.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR")

	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train.")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	testbed = ngp.Testbed()

	if args.load_snapshot:
		# load network
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		# build network
		assert os.path.exists(args.network)
		testbed.load_model_config(args.network)
		# load dataset
		testbed.load_training_data(args.scene, 0.33, [0.5,0.5,0.5])

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0 and not args.load_snapshot:
		n_steps = 100000

	if n_steps > 0:
		assert args.load_snapshot or args.save_snapshot
		output_dir = os.path.dirname(args.load_snapshot if args.load_snapshot else args.save_snapshot)

		start_t = time.perf_counter()
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.training_buffer.i_step < n_steps:
				testbed.train(16, 2**18)
				t.update(testbed.training_buffer.i_step - old_training_step)
				t.set_postfix(loss=testbed.training_buffer.loss)
				old_training_step = testbed.training_buffer.i_step
		elapsed_t = time.perf_counter() - start_t

		with open(os.path.join(output_dir, 'train_timing.txt'), 'w') as f:
			f.writelines(f'{elapsed_t} s')

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		assert args.load_snapshot or args.save_snapshot
		output_dir = os.path.dirname(args.load_snapshot if args.load_snapshot else args.save_snapshot)
		image_dir = os.path.join(output_dir, 'image')
		if not os.path.exists(image_dir): os.makedirs(image_dir)
		depth_dir = os.path.join(output_dir, 'depth')
		if not os.path.exists(depth_dir): os.makedirs(depth_dir)

		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)

		# configuration
		spp = 8
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]
		testbed.rendering_buffer.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"]

		# Evaluate metrics
		filename_list, psnr_list, ssim_list, lpips_list, time_list = [], [], [], [], []
		for i, frame in enumerate(tqdm(test_transforms["frames"])):
			# 1. load reference image
			p = frame["file_path"] + ".png"
			ref_image = read_image(os.path.join(data_dir, p))
			# NeRF blends with background colors in sRGB space, rather than first
			# transforming to linear space, blending there, and then converting back.
			# (See e.g. the PNG spec for more information on how the `alpha` channel
			# is always a linear quantity.)
			# The following lines of code reproduce NeRF's behavior (if enabled in
			# testbed) in order to make the numbers comparable.
			if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
				# Since sRGB conversion is non-linear, alpha must be factored out of it
				ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
				ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
				ref_image[...,:3] *= ref_image[...,3:4]
				ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
				ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

			# 2. render image from NeRF
			testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])

			# rgb
			start_t = time.perf_counter()
			image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, False)
			elapsed_t = time.perf_counter() - start_t

			# depth
			testbed.rendering_buffer.render_mode = ngp.RenderMode.Depth
			depth = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, False)
			testbed.rendering_buffer.render_mode = ngp.RenderMode.Shade

			# diffimg = np.absolute(image - ref_image)
			# diffimg[...,3:4] = 1.0

			# 3. metric
			A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
			R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)

			psnr_v = mse2psnr(float(compute_error("MSE", A, R)[0]))
			ssim_v = float(compute_error("SSIM", A, R)[0])
			lpips_v = float(compute_error("LPIPS", A, R)[0])

			filename_list.append(os.path.basename(p))
			psnr_list.append(psnr_v)
			ssim_list.append(ssim_v)
			lpips_list.append(lpips_v)
			time_list.append(elapsed_t)
			
			write_image(os.path.join(image_dir, os.path.basename(p)), image[...,:3])
			write_depth(
				os.path.join(depth_dir, os.path.basename(p).split('.')[0]+'.png'), 
				depth[...,0], 1./6., cm='jet')

		psnr_mean, ssim_mean, lpips_mean = np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)
		time_mean = np.mean(time_list)
		pd.DataFrame(
			{
				''      	: [ fn for fn in filename_list ] + ['mean'],
				'PSNR'  	: psnr_list + [psnr_mean],
				'SSIM'  	: ssim_list + [ssim_mean],
				'LPIPS' 	: lpips_list + [lpips_mean],
				'Time (s)'	: time_list + [time_mean]
			}
    	).to_excel(os.path.join(output_dir, 'eval_stat.xls'), index=False)
		print(f"PSNR={psnr_mean:.3f} SSIM={ssim_mean:.3f} LPIPS(VGG)={lpips_mean:.3f} FPS={1./time_mean:.2f}")