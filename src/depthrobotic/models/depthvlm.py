"""models / depthvlm: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.perception.parsing import parse_json_object
from depthrobotic.prompts import DEPTH_PROMPT
from PIL import Image
from transformers import AutoProcessor
import numpy as np
import torch


class DepthVLM:

    def __init__(self, model_path, source_path):
        from .vendor import load_depthvlm

        Qwen3VLForConditionalGeneration = load_depthvlm(source_path)
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()

    @torch.inference_mode()
    def generate_response(self, image, prompt, max_new_tokens=160):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise robot visual grounding assistant.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=[images], padding=True, return_tensors="pt"
        ).to("cuda")
        inputs.pop("mm_token_type_ids", None)
        image_tokens = int((inputs.input_ids == self.model.config.image_token_id).sum())
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        response = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return (response, image_tokens)

    @torch.inference_mode()
    def ask(self, image, prompt, max_new_tokens=160):
        response, image_tokens = self.generate_response(image, prompt, max_new_tokens)
        return (response, parse_json_object(response), image_tokens)

    @torch.inference_mode()
    def predict_depth(self, image, focal_px):
        if focal_px <= 0:
            raise ValueError("focal_px must be positive")
        scale = 1000.0 / focal_px
        canonical = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.BILINEAR
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": canonical},
                    {"type": "text", "text": DEPTH_PROMPT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=[images], padding=True, return_tensors="pt"
        ).to("cuda")
        output = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        if output.depth_pred is None:
            raise RuntimeError("DepthVLM returned no depth_pred")
        depth = output.depth_pred[0].detach().float().cpu().numpy()
        while depth.ndim > 2:
            depth = depth[0]
        return depth.astype(np.float32, copy=False)

    def generate_with_tools(self, messages, tools, max_new_tokens=300):
        """Run one generation step with tool definitions. Returns raw model output."""
        text = self.processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        all_images = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and (
                        item.get("type") == "image" or "image" in item
                    ):
                        img = item.get("image")
                        if img is not None:
                            all_images.append(img)
        if all_images:
            image_inputs, _ = self.process_vision_info(messages)
            if not isinstance(image_inputs, list):
                image_inputs = [image_inputs]
            inputs = self.processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to("cuda")
        else:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(
                "cuda"
            )
        inputs.pop("mm_token_type_ids", None)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0, inputs.input_ids.shape[1] :]
        return self.processor.decode(generated, skip_special_tokens=False)
