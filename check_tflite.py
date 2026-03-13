import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='models/expense_model.tflite')
interpreter.allocate_tensors()
print('--- Inputs ---')
for detail in interpreter.get_input_details():
    print(f"Index: {detail['index']}, Name: {detail['name']}, Shape: {detail['shape']}")
print('--- Outputs ---')
for detail in interpreter.get_output_details():
    print(f"Index: {detail['index']}, Name: {detail['name']}, Shape: {detail['shape']}")
