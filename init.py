# license:  Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#           Licensed under the CC BY-NC-SA 4.0 license
#           (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

from google_drive_downloader import GoogleDriveDownloader  as gdd
import os, json, glob
import pickle
import pdb
import parser
import argparse
import sys

def batch_download(keys, file_dir):
	# download the dataset
	for key in keys:
		folder = '/'.join(key.split('/')[0:-1])+'/'
		for i in range(1,len(key.split('/'))-1):
			folder = '/'.join(key.split('/')[0:i])+'/'
			if not os.path.exists(folder):
				os.mkdir(folder)
		if not os.path.isfile(key):
			gdd.download_file_from_google_drive(
				file_id=file_dir[key]['file_id'],
				dest_path=key,
				unzip=True)

	return 

def main():
	# parsing the input argument
	parser = argparse.ArgumentParser(\
		'determine the part of FLAT dataset to download.'
	)
	parser.add_argument(\
		'-n','--name', 
		metavar='dataset_name',
		type=str, 
		default = ['test'],
		nargs = '+',
		help='list the part of FLAT dataset to download'
	)
	parser.add_argument(\
		'-c','--category', 
		metavar='category_name',
		type=str, 
		default = 'kinect',
		help='list the hardware or category of FLAT dataset to download'
	)
	args = parser.parse_args()

	# download the parameters for cameras first
	param_id = '1qXvprK-vmS4eJJA4GimjjuqNoNPuAvpw'
	folder = './params/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	if not os.path.isfile(folder+'params.zip'):
		gdd.download_file_from_google_drive(
			file_id=param_id,
			dest_path=folder+'params.zip',
			unzip=True,
		)

	# download the trained models for kinect
	param_id = '1gVFmJ4mXkcnjjNHfgQ_BKM4v7woMUYWa'
	os.chdir('./pipe/')
	folder = './models/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	if not os.path.isfile(folder+'params.zip'):
		gdd.download_file_from_google_drive(
			file_id=param_id,
			dest_path=folder+'kinect.zip',
			unzip=True
		)
	os.chdir('../')

	# create the local folder for the dataset
	folder = './FLAT/'
	if not os.path.exists(folder):
		os.mkdir(folder)

	# load the directory list of the flat dataset
	file_dir_name = 'file_dir.pickle'
	with open(file_dir_name, 'rb') as f:
		file_dir = pickle.load(f)

	# inidicate the hardware or trans_render one wants to download	
	flat_flg = args.category
	lists = [key for key in file_dir.keys() if (flat_flg in key)*('.txt' in key)]
	if 'all' in args.name:
		list_flgs = lists
	else:
		list_flgs = []
		for i in range(len(args.name)):
			list_flgs += [key for key in file_dir.keys() \
			if (flat_flg in key)*('.txt' in key)*(args.name[i] in key)]

	os.chdir(folder)
	lists = []

	# if one needs to download the trans_rendering file
	if flat_flg == 'trans_render':
		# download the files in the datafolder
		keys = [key for key in file_dir.keys() \
			if (key.split('/')[1] == flat_flg)
		]
		batch_download(keys, file_dir)
	else:
		# download the certain list indicated by the flg
		for i in range(len(list_flgs)):
			filename = list_flgs[i]
			if filename in file_dir.keys():
				batch_download([filename],file_dir)

				# load the file, and read stuffs
				f = open(filename,'r')
				message = f.read()
				files = message.split('\n')
				data_list = files[0:-1]

				# download the images in the list folder
				filename = filename[:-4]+'/'
				keys = [key for key in file_dir.keys() if filename in key]
				batch_download(keys, file_dir)

				# download the files in the datafolder
				keys = [key for key in file_dir.keys() \
					if (key.split('/')[-1] in data_list) \
					and (key.split('/')[1] == flat_flg)
				]
				batch_download(keys, file_dir)

	# download a trans_render model for data augmentation
	param_id = '1CEghNRT-Y_uNzFTkIUXQHaB61kXN0Kt6'
	key = './trans_render/static/'
	for i in range(1,len(key.split('/'))):
		folder = '/'.join(key.split('/')[0:i])+'/'
		if not os.path.exists(folder):
			os.mkdir(folder)
	if not os.path.isfile(folder+'1499455750460059.pickle'):
		gdd.download_file_from_google_drive(
			file_id=param_id,
			dest_path=folder+'1499455750460059.pickle',
			unzip=True,
		)

if __name__ == "__main__":
	main()
