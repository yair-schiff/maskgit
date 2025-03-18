import argparse
import os

import jax
import numpy as np
import rich.syntax
import rich.tree
from tqdm.auto import tqdm
from PIL import Image

from maskgit.inference import ImageNet_class_conditional_generator


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
	"""
	python gen_imgs.py \
		--output_image_path ./outputs/images \
		--output_label_path ./outputs/labels \
		--output_start_index 0 \
		--seed 42 \
		--num_batches 2 \
		--batch_size 8 \
		--image_size 256 \
		--decoding_strategy remdm \
		--num_iter 16 \
		--mask_scheduling_method uniform
	"""
	parser = argparse.ArgumentParser(description="Generate images from a trained model")
	parser.add_argument("--output_image_path", type=str, default="./outputs/images",
											help="Path to save the generated images")
	parser.add_argument("--output_label_path", type=str, default="./outputs/labels",
											help="Path to save the generated labels")
	parser.add_argument("--output_start_index", type=int, default=0,
											help="Starting index for the output files")
	parser.add_argument("--seed", type=int, default=42,
											help="Random seed")
	parser.add_argument("--num_batches", type=int, default=1,
											help="Number of batches to generate")
	parser.add_argument("--batch_size", type=int, default=8,
											help="Batch size")
	parser.add_argument("--image_size", type=int, default=256,
											help="Size of the generated images")
	parser.add_argument("--decoding_strategy", type=str, default="maskgit",
											help="Decoding strategy",
											choices=["maskgit", "mdlm", "remdm_conf", "remdm_rescale", "remdm_cap"])
	parser.add_argument("--remdm_eta", type=float, default=0.0,
											help="REMDM eta (used for REMDM constant / looping decoding")
	parser.add_argument("--num_iter", type=int, default=16,
											help="Number of iterations")
	parser.add_argument("--mask_scheduling_method", type=str, default="uniform",
											help="Mask scheduling method")
											# choices=["uniform", "pow", "cosine", "log", "exp"])
	parser.add_argument("--sampling_temperature", type=float, default=1.0,
											help="Sampling temperature")
	parser.add_argument("--sampling_temperature_annealing", type=str, default="False",
											help="Use sampling temperature annealing.",
											choices=["True", "False"])
	args = parser.parse_args()

	style = "dim"
	tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

	fields = vars(args).keys()
	for field in fields:
		branch = tree.add(field, style=style, guide_style=style)
		config_section = vars(args).get(field)
		branch_content = str(config_section)
		branch.add(rich.syntax.Syntax(branch_content, "yaml"))
	rich.print(tree)

	output_image_path = args.output_image_path
	output_label_path = args.output_label_path
	index = args.output_start_index
	image_size = args.image_size
	num_batches = args.num_batches
	batch_size = args.batch_size
	generator = ImageNet_class_conditional_generator(image_size=image_size)
	arbitrary_seed = args.seed + index
	decoding_strategy = args.decoding_strategy
	remdm_eta = args.remdm_eta
	num_iter = args.num_iter
	mask_scheduling_method = args.mask_scheduling_method
	sampling_temperature = args.sampling_temperature
	sampling_temperature_annealing = str2bool(args.sampling_temperature_annealing)
	print(f"Generating {num_batches * batch_size:,d} images ({num_batches:,d} batches * {batch_size:,d} images per batch)"
				f" - Indices [{index:06d}, {index + num_batches * batch_size - 1:06d}]\n")

	os.makedirs(output_image_path, exist_ok=True)
	os.makedirs(output_label_path, exist_ok=True)
	rng = jax.random.PRNGKey(arbitrary_seed)
	for i in tqdm(range(num_batches), desc="Generating images"):
		# Skip batches that have completed
		if all([os.path.exists(f"{output_image_path}/{index + i * batch_size + j:06d}.png") \
						and os.path.exists(f"{output_label_path}/{index + i * batch_size + j:06d}.txt")
						for j in range(batch_size)]):
			print(f"Skipping batch {i}. Images [{index + i * batch_size:06d} - {index + i * batch_size + batch_size - 1:06d}] already generated")
			continue


		# Sample label \in [0, 999]
		rng, label_rng = jax.random.split(rng)
		label = jax.random.randint(label_rng, (batch_size, 1), minval=0, maxval=1000)

		# prep the input tokens based on the chosen label
		input_tokens = generator.create_input_tokens_normal(label, batch_size=batch_size)

		rng, sample_rng = jax.random.split(rng)
		results = generator.generate_samples(
			input_tokens, sample_rng,
			decoding_strategy=decoding_strategy,
			remdm_eta=remdm_eta,
			num_iter=num_iter,
			mask_scheduling_method=mask_scheduling_method,
			sampling_temperature_annealing=sampling_temperature_annealing,
			sampling_temperature=sampling_temperature)

		# Save images
		for j in range(batch_size):
			img = np.clip(results[j], 0, 1) * 255
			Image.fromarray(img.astype(np.uint8)).save(f"{output_image_path}/{index + i*batch_size + j:06d}.png")
			with open(f"{output_label_path}/{index + i*batch_size + j:06d}.txt", "wt") as f:
				f.write(f"{label[j].item()}")
