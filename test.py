from inference.layer_b_infer import infer_image

result = infer_image("dataset/ai/stevejobs.png")
print(result)

result = infer_image("dataset/real/real_0001.jpg")
print(result)
