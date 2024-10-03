import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img = 'images/test.webp'

raw_image = Image.open(img).convert('RGB')

# conditional image captioning
#text = "a photography of"
#inputs = processor(raw_image, text, return_tensors="pt")

#out = model.generate(**inputs, max_new_tokens=30)
#print(processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
