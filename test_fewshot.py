import requests
import torch
import transformers
import json
from PIL import Image
# from otter.modeling_otter import OtterForConditionalGeneration
import argparse
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from otter.modeling_otter import OtterForConditionalGeneration



def get_formatted_prompt(prompt: str) -> str:
    return f"<image> User: {prompt} GPT: <answer>"


def get_response(path: str, prompt: str) -> str:
    query_image = Image.open(path)
    vision_x = (
        image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
        .unsqueeze(1)
        .unsqueeze(0)
    )
    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=256,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_path_or_name",
    #     type=str,
    #     default="luodian/otter-9b-hf",
    #     help="Path or name of the model (HF format)",
    # )
    # parser.add_argument(
    #     "--model_version_or_tag",
    #     type=str,
    #     default="apr25_otter",
    #     help="Version or tag of the model",
    # )
    # parser.add_argument(
    #     "--input_file",
    #     type=str,
    #     default="evaluation/sample_questions.json",
    #     help="Path of the input file",
    # )
    # args = parser.parse_args()
    model_path_or_name = "/home/yabin/otter/aiart/checkpoint_4.pt"
    img_path1 = "/home/yabin/datasets/aiart/train2014/COCO_train2014_000000435833.jpg"
    img_path2 = "/home/yabin/datasets/aiart/sdft/0033107.jpg"
    img_path3 = "/home/yabin/datasets/aiart/sdft/0070108.jpg"

    model = FlamingoForConditionalGeneration.from_pretrained(
        "luodian/openflamingo-9b-hf", device_map="auto"
    )

    # model = OtterForConditionalGeneration.from_pretrained(
    #     'luodian/otter-9b-hf', device_map="auto"
    # )

    tokenizer = model.text_tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]}
    )
    model.lang_encoder.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(model_path_or_name, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], False)
    model.text_tokenizer.padding_side = "left"
    image_processor = transformers.CLIPImageProcessor()


    demo_image_one = Image.open(img_path1)
    demo_image_two = Image.open(img_path2)
    query_image = Image.open(img_path3)
    vision_x = (
        image_processor.preprocess(
            [demo_image_one, demo_image_two, query_image], return_tensors="pt"
        )["pixel_values"]
        .unsqueeze(1)
        .unsqueeze(0)
    )
    lang_x = model.text_tokenizer(
        [
            "<image>User: Describe this image. \
            GPT:<answer> This is a real image, because it has many details and no mistakes.<|endofchunk|>\
            <image>User: Describe this image. \
            GPT:<answer> This is an ai generated image by StableDiffusion, because it's a little fuzzy in some places.<|endofchunk|>\
            <image>User: Describe this image. \
            GPT:<answer>"
        ],
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=256,
        num_beams=3,
        no_repeat_ngram_size=3,
    )

    print("Generated text: ", model.text_tokenizer.decode(generated_text[0]))

    # response = get_response(img_path, prompt)
    # import pdb;pdb.set_trace()
