import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import string
import time
import argparse
import os

characters = ' '+ string.digits + string.ascii_lowercase+string.punctuation


def my_ctc_decode(y_pred):
	decode =(np.ones((1,16))*(-1)).astype('int64');
	idx=0;prev=69;
	for i in range(16):
		t=np.argmax(y_pred[0][i])
		if t!=69:
			if t!=prev:
				decode[0][idx]=t;idx=idx+1;
		prev=t
	return decode


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Path to the model', type=str)
	parser.add_argument('--inputDir', help='Where to read the captchas to break. E.g. data/', type=str)
	parser.add_argument('--output', help='File where the classifications should be saved', type=str)
	args = parser.parse_args()

	if args.model is None:
		print("Please specify the model path")
		exit(1)

	if args.inputDir is None:
		print("Please specify the directory with captchas to break")
		exit(1)

	if args.output is None:
		print("Please specify the path to the output file")
		exit(1)


	interpreter = tflite.Interpreter(model_path=args.model)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	input_shape = input_details[0]['shape']

	path=args.inputDir;files=[]
	for file in os.listdir(path):
		if file.endswith('.png'):
			files.append(file)
	files.sort()
	time_start=time.time()

	with open(args.output, 'w') as output_file:
		for file in files:
			f = path+file;ff = Image.open(f);dd = np.array(ff).astype('float32');
			dd = np.array([dd/255.0])
			interpreter.set_tensor(input_details[0]['index'], dd)
			interpreter.invoke()
			y_pred = interpreter.get_tensor(output_details[0]['index'])
			out=my_ctc_decode(y_pred)[:,:6];#print(out)
			out = ''.join([characters[x] for x in out[0]])
			#print(file+" "+out.strip(' '))
			output_file.write(file + "," + out.strip(' ') + "\n")
	time_end=time.time()
	print('time cost',time_end-time_start,'s')


if __name__ == '__main__':
	main()
