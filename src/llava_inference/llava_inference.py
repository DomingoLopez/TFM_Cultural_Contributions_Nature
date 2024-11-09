from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import time


class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images: list):
        """
        Loads a trial in memory to use it, manipulate it and use as input for Llava-1.5 multimodal.
        Args:
            images (dict)
        """
        # Cargar el procesador y el modelo correspondientes (Llava-1.7-7b, es más que bastante. Podemos probar con LlavaNext también)
        processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to("cuda:0")

        # Preparar imagen de ejemplo
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

        # Definir conversación
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this image in one of these categories: Nature, Urban, Rural or Others"},
                {"type": "image", "image": image},  
                ],
            },
        ]

        # Crear el prompt con el template de chat
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Procesar inputs, asegurándote de que están en CUDA
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Autogenerar la respuesta a partir del prompt
        start_time = time.time()
        output = model.generate(**inputs, max_new_tokens=100)
        end_time = time.time()


        inference_time = end_time - start_time
        print(f"Tiempo de inferencia: {inference_time:.2f} segundos")

        # Decodificar y mostrar el resultado
        print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    pass
