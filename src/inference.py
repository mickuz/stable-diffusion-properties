import torch
import argparse

from daam import trace
from diffusers import DiffusionPipeline


def setup_model(model_id, device):
    model = DiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    return model


def run_inference(model, prompt, device, iteration=0):
    generator = torch.Generator(device=device)
    generator.manual_seed(iteration)

    output_dir_name = prompt.lower().replace(" ", "-") + f"-{iteration}"

    with torch.no_grad():
        with trace(model) as tc:
            model(prompt=prompt, generator=generator)
            experiment = tc.to_experiment(output_dir_name)
            experiment.save()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--number_of_samples", type=int, required=True)
    args = parser.parse_args()

    model = setup_model(args.model_id, device=device)

    for i in range(args.number_of_samples):
        run_inference(model, args.prompt, device=device, iteration=i)
